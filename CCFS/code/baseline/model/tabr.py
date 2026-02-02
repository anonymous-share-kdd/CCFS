import os
import math
from typing import Dict, Any, Optional, Literal, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import faiss

_FAISS_OK = False

def get_d_out(n_classes: Optional[int]) -> int:
    """1 for regression, n_classes for classification."""
    return 1 if (n_classes is None) else int(n_classes)

class Lambda(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x): return self.fn(x)

# ==============================
#           TabR Model
# ==============================
class Model(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        num_embeddings: Optional[dict],  # unused (we pass None)
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.one_hot_encoder = None  # 我们在 runner 里只走 {"num": ...}
        self.num_embeddings = None   # 不用数值嵌入

        # >>> Encoder (E)
        d_in = n_num_features + n_bin_features + sum(cat_cardinalities)
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList([make_block(i > 0) for i in range(encoder_n_blocks)])

        # >>> Retriever (R)
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main),
                Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> Predictor (P)
        self.blocks1 = nn.ModuleList([make_block(True) for _ in range(predictor_n_blocks)])
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)

    def _encode(self, x_: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')

        xs = []
        if x_num is not None:
            xs.append(x_num if self.num_embeddings is None else self.num_embeddings(x_num).flatten(1))
        if x_bin is not None:
            xs.append(x_bin)
        if x_cat is not None:
            raise NotImplementedError("This single-file runner only supports numeric features via {'num': ...}.")
        x = torch.cat(xs, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def _faiss_search(self, k: Tensor, candidate_k: Tensor, topk: int) -> Tensor:
        # returns indices [B, topk]; uses FAISS if available, else PyTorch topk
        if _FAISS_OK:
            try:
                d_main = k.shape[1]
                if k.is_cuda:
                    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                else:
                    index = faiss.IndexFlatL2(d_main)
                index.reset()
                index.add(candidate_k)  # torch tensor supported via contrib.torch_utils
                _, idx = index.search(k, topk)  # type: ignore
                return torch.as_tensor(idx, device=k.device, dtype=torch.long)
            except Exception:
                pass
        # fallback: brute-force via matmul
        # similarity trick: -||a-b||^2 = -||a||^2 + 2 a·b - ||b||^2
        sims = (
            -k.square().sum(-1, keepdim=True)
            + 2 * (k @ candidate_k.T)
            - candidate_k.square().sum(-1).unsqueeze(0)
        )
        return sims.topk(topk, dim=-1).indices

    def forward(
        self,
        *,
        x_: Dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: Dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.memory_efficient):
            candidate_k = self._encode(candidate_x_)[1]
        x, k = self._encode(x_)

        if is_train:
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        B, D = k.shape
        idx = self._faiss_search(k, candidate_k, context_size + (1 if is_train else 0))

        if is_train:
            # remove self index (avoid leakage)
            # distances-based way in orig, here simply drop position equal to self
            self_row = torch.arange(B, device=k.device)[:, None]
            # put +inf mask and re-topk is complicated; instead: gather all then drop first that equals
            # Safer: recompute distances and sort; but simpler: replace exact match with last index
            # We emulate orig behavior by shifting:
            mask = idx == self_row
            if mask.any():
                # move self index to the end position
                for i in range(B):
                    m = mask[i]
                    if m.any():
                        pos = torch.nonzero(m, as_tuple=False)[0, 0]
                        idx[i, pos:-1] = idx[i, pos+1:].clone()
                        idx[i, -1] = 0  # dummy; will be ignored after slice below
            idx = idx[:, :context_size]

        # gather context keys
        context_k = candidate_k[idx]                           # [B, K, D]
        # similarities (same as orig)
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = F.dropout(probs, p=self.dropout.p, training=self.training)

        # context values: label embedding + delta
        if isinstance(self.label_encoder, nn.Linear):  # 回归
            y_neighbors = candidate_y[idx]
            if y_neighbors.dim() == 2:  # [B,K]
                y_neighbors = y_neighbors.unsqueeze(-1)
            context_y_emb = self.label_encoder(y_neighbors)  # [B,K,D]
        else:  # 分类
            context_y_emb = self.label_encoder(candidate_y[idx].long())  # [B, K, D]
        values = context_y_emb + self.T(k[:, None] - context_k)          # [B, K, D]
        context_x = (probs[:, None] @ values).squeeze(1)                 # [B, D]
        x = x + context_x

        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x  # [B, C] (cls) or [B, 1] (reg)


# ==============================
#        Baseline wrapper
# ==============================
from utils.tools import cls_metrics, reg_metrics

def _to_tensor(x, device, is_y: bool, task_type: str):
    if is_y:
        if task_type == "cls":
            return torch.as_tensor(x, dtype=torch.long, device=device)
        x = x.astype(np.float32)
        return torch.as_tensor(x, dtype=torch.float, device=device)
    else:
        return torch.as_tensor(x, dtype=torch.float, device=device)

def gen_tabr(
    fe,
    *,
    seed: int = 324,
    device: str = "cuda",
    # —— 模型超参（贴近原实现的常见设定）——
    context_size: int = 32,
    d_main: int = 256,
    d_multiplier: float = 2.0,
    encoder_n_blocks: int = 2,
    predictor_n_blocks: int = 2,
    context_dropout: float = 0.1,
    dropout0: float = 0.1,
    dropout1: Optional[float] = None,
    normalization: str = "LayerNorm",
    activation: str = "ReLU",
    # —— 优化与训练 ——
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    max_epochs: int = 2000,
    patience: int = 300,
    valid_ratio: float = 0.2,
) -> Dict[str, Any]:
    """
    与你的 baseline 框架对齐：
      - fe.train / fe.test 的最后一列为 label
      - fe.task_type in {"cls","reg"}
    返回:
      {"train": m_tr, "test": m_te, "yhat_tr": yhat_tr, "yhat_te": yhat_te}
    """
    # 设备与随机种子

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # 读取数据
    label_col = fe.train.columns[-1]
    X_full = fe.train.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y_full = fe.train[label_col].to_numpy()
    X_te   = fe.test.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y_te   = fe.test[label_col].to_numpy()

    # 切分 train/val
    from sklearn.model_selection import train_test_split
    strat = y_full if fe.task_type == "cls" else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=valid_ratio, random_state=seed, stratify=strat
    )

    # 张量
    X_tr_t     = _to_tensor(X_tr, dev, False, fe.task_type)
    y_tr_t     = _to_tensor(y_tr, dev, True,  fe.task_type)
    X_val_t    = _to_tensor(X_val, dev, False, fe.task_type)
    y_val_t    = _to_tensor(y_val, dev, True,  fe.task_type)
    X_trfull_t = _to_tensor(X_full, dev, False, fe.task_type)
    y_trfull_t = _to_tensor(y_full, dev, True,  fe.task_type)
    X_te_t     = _to_tensor(X_te, dev, False, fe.task_type)
    y_te_t     = _to_tensor(y_te, dev, True,  fe.task_type)

    # 构造模型（只走 numeric 通道）
    n_classes = None if fe.task_type != "cls" else int(np.unique(y_full).size)
    model = Model(
        n_num_features=X_tr.shape[1],
        n_bin_features=0,
        cat_cardinalities=[],
        n_classes=n_classes,
        num_embeddings=None,
        d_main=d_main,
        d_multiplier=d_multiplier,
        encoder_n_blocks=encoder_n_blocks,
        predictor_n_blocks=predictor_n_blocks,
        mixer_normalization="auto",
        context_dropout=context_dropout,
        dropout0=dropout0,
        dropout1="dropout0" if dropout1 is None else dropout1,
        normalization=normalization,
        activation=activation,
        memory_efficient=False,
        candidate_encoding_batch_size=None,
    ).to(dev)

    # 优化器/损失
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss() if fe.task_type == "cls" else nn.MSELoss()

    def pack_num(x_t: Tensor) -> Dict[str, Tensor]:
        return {"num": x_t}

    def iter_batches(N: int, bs: int):
        idx = torch.randperm(N, device=dev)
        for s in range(0, N, bs):
            yield idx[s:s+bs]

    # 训练循环（与原理一致：候选=全训练集；训练时从候选里去掉当前 batch）
    best_score = -np.inf if fe.task_type == "cls" else np.inf
    best_state = None
    best_epoch = 0
    no_improve = 0

    print("===> Start training TabR (single-file, orig-logic) ...")
    bar = tqdm(range(1, max_epochs + 1), desc="TabR training", ncols=100, leave=False)
    is_better = (lambda cur, best: cur > best) if fe.task_type == "cls" else (lambda cur, best: cur < best)

    for epoch in bar:
        model.train()
        epoch_losses = []
        N = X_tr_t.shape[0]

        for bidx in iter_batches(N, batch_size):
            xb, yb = X_tr_t[bidx], y_tr_t[bidx]
            mask = torch.ones(N, dtype=torch.bool, device=dev)
            mask[bidx] = False
            cand_x, cand_y = X_tr_t[mask], y_tr_t[mask]

            opt.zero_grad(set_to_none=True)
            out = model(
                x_=pack_num(xb),
                y=yb,
                candidate_x_=pack_num(cand_x),
                candidate_y=cand_y,
                context_size=context_size,
                is_train=True,
            ).squeeze(-1)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().item()))

        # 验证
        model.eval()
        with torch.no_grad():
            val_out = model(
                x_=pack_num(X_val_t),
                y=None,
                candidate_x_=pack_num(X_tr_t),
                candidate_y=y_tr_t,
                context_size=context_size,
                is_train=False,
            ).squeeze(-1)
            if fe.task_type == "cls":
                score = (val_out.argmax(1) == y_val_t).float().mean().item()  # val_acc
            else:
                score = float(torch.sqrt(F.mse_loss(val_out, y_val_t)).item())  # val_RMSE

        avg_loss = float(np.mean(epoch_losses) if epoch_losses else 0.0)

        # 更新 best
        if is_better(score, best_score):
            best_score = score
            best_epoch = epoch
            # 保存最佳权重
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # 用进度条覆盖式展示
        bar.set_description(f"Epoch {epoch}/{max_epochs}")
        bar.set_postfix(loss=f"{avg_loss:.6f}", best=f"{best_score:.5f}@{best_epoch}")

        # 早停
        if no_improve >= patience:
            bar.write("Early stopping.")
            break

    bar.close()
    model.eval()
    with torch.no_grad():
        # train(full) / test 都用 train_split 作为候选（与原实现评估阶段一致）
        out_tr = model(
            x_=pack_num(X_trfull_t),
            y=None,
            candidate_x_=pack_num(X_tr_t),
            candidate_y=y_tr_t,
            context_size=context_size,
            is_train=False,
        )
        out_te = model(
            x_=pack_num(X_te_t),
            y=None,
            candidate_x_=pack_num(X_tr_t),
            candidate_y=y_tr_t,
            context_size=context_size,
            is_train=False,
        )

        if fe.task_type == "cls":
            yhat_tr = out_tr.argmax(1).cpu().numpy()
            yhat_te = out_te.argmax(1).cpu().numpy()
            prob_tr = F.softmax(out_tr, dim=1).cpu().numpy()
            prob_te = F.softmax(out_te, dim=1).cpu().numpy()
            m_tr = cls_metrics(y_full, yhat_tr, y_score=prob_tr)
            m_te = cls_metrics(y_te,  yhat_te, y_score=prob_te)
        else:
            yhat_tr = out_tr.squeeze(-1).cpu().numpy()
            yhat_te = out_te.squeeze(-1).cpu().numpy()
            m_tr = reg_metrics(y_full, yhat_tr)
            m_te = reg_metrics(y_te,  yhat_te)

    print("✅ Training finished.")

    if hasattr(model, "eval"):
        model.eval()

    return {"train": m_tr, "test": m_te, "yhat_tr": yhat_tr, "yhat_te": yhat_te, "model": model}
