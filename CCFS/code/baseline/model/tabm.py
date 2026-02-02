import os
import math
import numpy as np
from typing import Dict, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.tools import cls_metrics, reg_metrics, split_and_encode
from tabm import TabM


def gen_tabm(
    fe,
    seed: int = 324,
    device: str = "cuda",
    *,
    valid_size: float = 0.2,
    batch_size: int = 512,
    max_epochs: int = 500,
    patience: int = 80,
    lr: float = 3e-3,
    weight_decay: float = 1e-5,
    max_cat_card: int = 50,
):
    """
    使用你提供的 split_and_encode 将列拆成 (x_f, x_c) 并喂给 TabM。
    fe.task_type ∈ {"cls","reg"}；最后一列为 label。
    返回: {"train","test","yhat_tr","yhat_te"}
    """
    # 设备 & 种子

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 拆分 + 编码（用你的函数） ---
    label_col = fe.train.columns[-1]
    enc = split_and_encode(fe.train, fe.test, label_col, max_cat_card=max_cat_card)

    xtr_f, xtr_c = enc["xtr_f"].to(dev), enc["xtr_c"].to(dev)
    xte_f, xte_c = enc["xte_f"].to(dev), enc["xte_c"].to(dev)
    y_tr_all = enc["y_tr"].to(dev)
    y_te = enc["y_te"].to(dev)

    # 训练/验证切分（保持和你其他 baseline 一致：按索引切）
    n_tr = xtr_f.size(0)
    n_val = int(round(n_tr * valid_size))
    idx = torch.randperm(n_tr, device=dev)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr_f, X_tr_c = xtr_f[tr_idx], xtr_c[tr_idx]
    X_val_f, X_val_c = xtr_f[val_idx], xtr_c[val_idx]
    y_tr = y_tr_all[tr_idx]
    y_val = y_tr_all[val_idx]

    # 统计信息
    n_num_features = X_tr_f.size(1)
    cat_cardinalities = list(map(int, enc["categories"])) if len(enc["categories"]) else None

    is_cls = (fe.task_type == "cls")
    if is_cls:
        # 注意：分类标签需要 int64；二分类也按多分类处理（CE）
        y_tr = y_tr.long()
        y_val = y_val.long()
        y_full = enc["y_tr"].cpu().numpy()
        y_te_cpu = enc["y_te"].cpu().numpy()
        n_classes = int(np.unique(np.concatenate([y_full, y_te_cpu])).size)
        d_out = n_classes
    else:
        # 回归 (N,1)
        y_tr = y_tr.float().view(-1, 1)
        y_val = y_val.float().view(-1, 1)
        y_full = enc["y_tr"].cpu().numpy()
        d_out = 1

    # --- 构建 TabM ---
    make_kwargs = dict(n_num_features=n_num_features, d_out=d_out)
    if cat_cardinalities:
        make_kwargs["cat_cardinalities"] = cat_cardinalities  # 会自动做 one-hot

    model = TabM.make(**make_kwargs).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- DataLoader（带 cat/num）---
    def _dl(xf, xc, y, bs, shuffle):
        return DataLoader(TensorDataset(xf, xc, y), batch_size=bs, shuffle=shuffle, drop_last=False)

    tr_loader = _dl(X_tr_f, X_tr_c, y_tr, batch_size, True)

    # --- 训练循环 + early stopping ---
    best_metric = -math.inf if is_cls else math.inf
    best_state = None
    best_epoch = 0
    no_improve = 0

    bar = tqdm(range(1, max_epochs + 1), desc="TabM training", ncols=100, leave=True)
    for epoch in bar:
        model.train()
        losses = []
        for xb_f, xb_c, yb in tr_loader:
            opt.zero_grad(set_to_none=True)
            x_cat = xb_c if xb_c.size(1) > 0 else None
            y_pred_k = model(xb_f, x_cat)  # (B, K, d_out or 1)

            if is_cls:
                B, K, C = y_pred_k.shape
                loss = F.cross_entropy(y_pred_k.reshape(B * K, C), yb.repeat_interleave(K))
            else:
                # 回归，MSE：让 K 个头都学习同一目标
                preds = y_pred_k.squeeze(-1)              # (B, K)
                loss = F.mse_loss(preds, yb.repeat(1, preds.size(1)))

            loss.backward()
            opt.step()
            losses.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            x_val_cat = X_val_c if X_val_c.size(1) > 0 else None
            val_out = model(X_val_f, x_val_cat)  # (Nv, K, d_out/1)
            if is_cls:
                prob = F.softmax(val_out, dim=-1).mean(dim=1)      # (Nv, C)
                val_metric = (prob.argmax(1) == y_val).float().mean().item()
            else:
                pred = val_out.squeeze(-1).mean(dim=1, keepdim=True)  # (Nv,1)
                val_metric = float(torch.sqrt(F.mse_loss(pred, y_val)).item())  # RMSE

        avg_loss = float(np.mean(losses) if losses else 0.0)
        improved = (val_metric > best_metric) if is_cls else (val_metric < best_metric)
        if improved:
            best_metric = val_metric
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        tag = "val_acc" if is_cls else "val_RMSE"
        bar.set_postfix({"loss": f"{avg_loss:.6f}", tag: f"{val_metric:.5f}", "best@": f"{best_epoch}:{best_metric:.5f}"})

        if no_improve >= patience:
            tqdm.write("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # --- 推断（train_full / test，K 头平均）---
    with torch.no_grad():
        X_full_f = enc["xtr_f"].to(dev)
        X_full_c = enc["xtr_c"].to(dev)
        x_full_cat = X_full_c if X_full_c.size(1) > 0 else None
        x_te_cat = xte_c if xte_c.size(1) > 0 else None

        out_tr = model(X_full_f, x_full_cat)
        out_te = model(xte_f, x_te_cat)

        if is_cls:
            prob_tr = F.softmax(out_tr, dim=-1).mean(dim=1).cpu().numpy()
            prob_te = F.softmax(out_te, dim=-1).mean(dim=1).cpu().numpy()
            yhat_tr = prob_tr.argmax(axis=1)
            yhat_te = prob_te.argmax(axis=1)
            m_tr = cls_metrics(y_full, yhat_tr, y_score=prob_tr)
            m_te = cls_metrics(y_te.cpu().numpy(), yhat_te, y_score=prob_te)
        else:
            yhat_tr = out_tr.squeeze(-1).mean(dim=1).cpu().numpy()
            yhat_te = out_te.squeeze(-1).mean(dim=1).cpu().numpy()
            m_tr = reg_metrics(y_full, yhat_tr)
            m_te = reg_metrics(y_te.cpu().numpy(), yhat_te)

    if hasattr(model, "eval"):
        model.eval()
    return {
        "train": m_tr,
        "test":  m_te,
        "yhat_tr": yhat_tr,
        "yhat_te": yhat_te,
        "model": model,
    }
