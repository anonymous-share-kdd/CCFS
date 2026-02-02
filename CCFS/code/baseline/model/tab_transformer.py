# baseline/model/tabtransformer.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tab_transformer_pytorch import TabTransformer

from feature import FeatureEvaluator
from utils.tools import cls_metrics, reg_metrics, split_and_encode


def gen_tabtransformer(
    fe: FeatureEvaluator,
    *,
    max_cat_card: int = 50,
    batch_size: int = 256,
    dim: int = 128,
    depth: int = 6,
    heads: int = 8,
    epochs: int = 2000,
    lr: float = 3e-4,
    wd: float = 1e-5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label = fe.label_col

    # ===== 1) 通用前处理 =====
    data = split_and_encode(fe.train, fe.test, label, max_cat_card=max_cat_card)
    Xc_tr, Xf_tr, Xc_te, Xf_te = data["xtr_c"], data["xtr_f"], data["xte_c"], data["xte_f"]
    y_tr, y_te = data["y_tr"], data["y_te"]
    categories = data["categories"]
    cat_cols, cont_cols = data["cat_cols"], data["cont_cols"]

    # ===== 2) Dataset + Dataloader =====
    if fe.task_type == "cls":
        y_tr_t = y_tr.long()
        y_te_t = y_te.long()
        n_classes = int(np.unique(y_tr.numpy()).shape[0])
        dim_out = 1 if n_classes == 2 else n_classes
    else:
        y_tr_t = y_tr.float()
        y_te_t = y_te.float()
        dim_out = 1

    train_loader = DataLoader(TensorDataset(Xc_tr, Xf_tr, y_tr_t), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xc_te, Xf_te, y_te_t), batch_size=batch_size, shuffle=False)

    # ===== 3) 连续特征均值方差（用于模型内部 normalization）=====
    if len(cont_cols) > 0:
        xtr_f_np = Xf_tr.numpy()
        mean = np.nanmean(xtr_f_np, axis=0)
        std = np.nanstd(xtr_f_np, axis=0)
        std[std == 0] = 1.0
        continuous_mean_std = torch.tensor(np.stack([mean, std], axis=1), dtype=torch.float32)
    else:
        continuous_mean_std = None

    # ===== 4) 模型定义 =====
    model = TabTransformer(
        categories=categories,
        num_continuous=len(cont_cols),
        dim=dim,
        depth=depth,
        heads=heads,
        dim_out=dim_out,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=nn.ReLU(),
        continuous_mean_std=continuous_mean_std
    ).to(device)

    # ===== 5) 损失函数 + 优化器 =====
    if fe.task_type == "cls":
        criterion = nn.BCEWithLogitsLoss() if dim_out == 1 else nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ===== 6) 训练 =====
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb_c, xb_f, yb in train_loader:
            xb_c, xb_f, yb = xb_c.to(device), xb_f.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb_c, xb_f).squeeze(-1)
            if fe.task_type == "cls" and dim_out == 1:
                yb = yb.float()
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if ep % 10 == 0 or ep == 1:
            print(f"[TabT][Epoch {ep:04d}] Loss={total_loss/len(train_loader):.4f}")

    # ===== 7) 预测函数 =====
    @torch.no_grad()
    def _predict(loader):
        model.eval()
        preds, scores, trues = [], [], []
        for xb_c, xb_f, yb in loader:
            xb_c, xb_f = xb_c.to(device), xb_f.to(device)
            out = model(xb_c, xb_f).squeeze(-1)

            if fe.task_type == "cls":
                if dim_out == 1:
                    prob = torch.sigmoid(out)
                    pred = (prob > 0.5).long()
                    score = prob.detach().cpu().numpy()
                else:
                    prob = torch.softmax(out, dim=-1)
                    pred = prob.argmax(dim=-1)
                    score = prob.detach().cpu().numpy()
                preds.append(pred.detach().cpu().numpy())
                scores.append(score)
            else:
                preds.append(out.detach().cpu().numpy())

            trues.append(yb.cpu().numpy())

        y_pred = np.concatenate(preds, axis=0) if preds else np.array([])
        y_true = np.concatenate(trues, axis=0) if trues else np.array([])

        if fe.task_type == "cls":
            y_score = np.concatenate(scores, axis=0) if scores else None
        else:
            y_score = None

        return y_true, y_pred, y_score

    # ===== 8) 计算指标 =====
    y_tr_true, yhat_tr, yscore_tr = _predict(train_loader)
    y_te_true, yhat_te, yscore_te = _predict(test_loader)

    if fe.task_type == "cls":
        m_tr = cls_metrics(y_tr_true, yhat_tr, y_score=yscore_tr)
        m_te = cls_metrics(y_te_true, yhat_te, y_score=yscore_te)
    else:
        m_tr = reg_metrics(y_tr_true, yhat_tr)
        m_te = reg_metrics(y_te_true, yhat_te)

    # ===== 9) 输出结果 =====
    print(f"[TabT] Done")
    print(f"  Train: {m_tr}")
    print(f"  Test:  {m_te}")

    if hasattr(model, "eval"):
        model.eval()

    return {
        "train": m_tr,
        "test": m_te,
        "yhat_tr": yhat_tr,
        "yhat_te": yhat_te,
        "model": model
    }