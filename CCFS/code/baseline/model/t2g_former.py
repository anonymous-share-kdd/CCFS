import math
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    def __init__(self, df, label_col, num_cols, cat_cols, task_type, cat_maps=None):
        self.label_col = label_col
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.task_type = task_type

        X = df.drop(columns=[label_col])
        y = df[label_col]

        self.X_num = torch.tensor(X[num_cols].values, dtype=torch.float32) if num_cols else None

        if cat_cols:
            if cat_maps is None:
                self.cat_maps = {}
                xs = []
                for c in cat_cols:
                    uniq = sorted(X[c].dropna().unique())
                    mapping = {v: i + 1 for i, v in enumerate(uniq)}
                    self.cat_maps[c] = mapping
                    xs.append(X[c].map(mapping).fillna(0).astype(int).values)
                self.X_cat = torch.tensor(np.stack(xs, axis=1), dtype=torch.long)
            else:
                self.cat_maps = cat_maps
                self.X_cat = torch.tensor(
                    np.stack([X[c].map(self.cat_maps[c]).fillna(0).astype(int).values for c in cat_cols], axis=1),
                    dtype=torch.long
                )
        else:
            self.X_cat = None
            self.cat_maps = {}

        if task_type == "cls":
            self.y = torch.tensor(y.values, dtype=torch.long)
        else:
            self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_num[idx] if self.X_num is not None else None,
                self.X_cat[idx] if self.X_cat is not None else None,
                self.y[idx])


class NumericalEmbedding(nn.Module):
    def __init__(self, n_num, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num, d_token))
        self.bias = nn.Parameter(torch.zeros(n_num, d_token))

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class GraphEstimator(nn.Module):
    def __init__(self, n_tokens, d_token, hidden=128):
        super().__init__()
        self.n_tokens = n_tokens
        self.node = nn.Parameter(torch.randn(n_tokens, d_token))
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_token, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self):
        F = self.n_tokens
        vi = self.node.unsqueeze(1).expand(F, F, -1)
        vj = self.node.unsqueeze(0).expand(F, F, -1)
        pair = torch.cat([vi, vj], dim=-1)
        logits = self.mlp(pair).squeeze(-1)
        A = torch.softmax(logits, dim=-1)
        return A


class GraphMHSA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, eps=1e-12):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + attn_bias
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        out = self.proj(out)
        return out


class T2GBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = GraphMHSA(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias):
        x = x + self.attn(self.norm1(x), attn_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossLevelReadout(nn.Module):
    def __init__(self, n_layers, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_layers),
        )

    def forward(self, cls_stack):
        g = self.gate(cls_stack[-1])
        w = torch.softmax(g, dim=-1)
        out = 0.0
        for i in range(cls_stack.shape[0]):
            out = out + cls_stack[i] * w[:, i].unsqueeze(-1)
        return out


class T2GFormer(nn.Module):
    def __init__(
        self,
        n_num,
        cat_cardinalities,
        d_token=128,
        n_heads=8,
        n_layers=3,
        dropout=0.1,
        task_type="cls",
        n_classes=2,
    ):
        super().__init__()
        self.task_type = task_type

        self.num_emb = NumericalEmbedding(n_num, d_token) if n_num > 0 else None
        self.cat_embs = nn.ModuleList([nn.Embedding(card + 1, d_token) for card in cat_cardinalities])

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        self.blocks = nn.ModuleList([T2GBlock(d_token, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.readout = CrossLevelReadout(n_layers, d_token)
        self.final_norm = nn.LayerNorm(d_token)

        self.graph = GraphEstimator(n_tokens=1 + n_num + len(cat_cardinalities), d_token=d_token, hidden=128)

        if task_type == "cls":
            self.head = nn.Linear(d_token, n_classes)
        else:
            self.head = nn.Linear(d_token, 1)

    def _make_tokens(self, x_num, x_cat):
        tokens = []
        if self.num_emb is not None:
            tokens.append(self.num_emb(x_num))
        if len(self.cat_embs) > 0:
            cat_tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)]
            tokens.append(torch.stack(cat_tokens, dim=1))
        x = torch.cat(tokens, dim=1)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x

    def forward(self, x_num, x_cat):
        x = self._make_tokens(x_num, x_cat)
        A = self.graph()
        bias = torch.log(A + 1e-12)
        bias = bias.unsqueeze(0).unsqueeze(0)

        cls_list = []
        for blk in self.blocks:
            x = blk(x, bias)
            cls_list.append(x[:, 0])

        cls_stack = torch.stack(cls_list, dim=0)
        z = self.readout(cls_stack)
        z = self.final_norm(z)

        out = self.head(z)
        if self.task_type == "reg":
            return out.squeeze(-1)
        return out


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for x_num, x_cat, y in loader:
        if x_num is not None:
            x_num = x_num.to(device)
        if x_cat is not None:
            x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x_num, x_cat)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for x_num, x_cat, y in loader:
        if x_num is not None:
            x_num = x_num.to(device)
        if x_cat is not None:
            x_cat = x_cat.to(device)
        y = y.to(device)
        out = model(x_num, x_cat)
        loss = criterion(out, y)
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)


def run_t2g_former(
    train_df,
    test_df,
    label_col,
    task_type,
    num_cols,
    cat_cols,
    n_classes=2,
    d_token=128,
    n_heads=8,
    n_layers=3,
    batch_size=256,
    lr=1e-3,
    epochs=20,
    device="cuda"
):
    train_ds = TabularDataset(train_df, label_col, num_cols, cat_cols, task_type)
    test_ds = TabularDataset(test_df, label_col, num_cols, cat_cols, task_type, train_ds.cat_maps)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    cat_cards = [len(train_ds.cat_maps[c]) for c in cat_cols]

    model = T2GFormer(
        n_num=len(num_cols),
        cat_cardinalities=cat_cards,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.1,
        task_type=task_type,
        n_classes=n_classes,
    ).to(device)

    if task_type == "cls":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        eval_epoch(model, test_loader, criterion, device)

    return model
