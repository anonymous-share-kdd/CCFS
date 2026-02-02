# -*- coding: utf-8 -*-

import os
import json
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

base_path = '../../data'

TASK_DICT = {
    'higgs': 'cls',
    'adult': 'cls',
    'pima_indian': 'cls',
    'brfss_diabetes': 'cls',
    'lifestyle_risk': 'cls',
    'credit_default': 'cls',
    'wine_red': 'cls',
    'wine_white': 'cls',
    'mus_scRNA_seq': 'cls',
    'mice_protein': 'cls',
    'mnist': 'cls',
    'openml_586': 'reg',
    'openml_588': 'reg',
    'openml_589': 'reg',
    'openml_607': 'reg',
    'openml_616': 'reg',
    'openml_618': 'reg',
    'openml_620': 'reg',
    'openml_637': 'reg',
    'parkinson': 'reg',
    'apt_rental_price': 'reg',
    'ct_slice_loc': 'reg',
}

DEFAULT_TEST_SIZE = 0.2


@dataclass
class FeatureEvaluator:
    task_name: str
    task_type: str
    label_col: str
    original: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    processed_hdf_path: str
    context_text_train: Optional[pd.Series] = None
    context_text_test: Optional[pd.Series] = None
    label_mapping: Optional[Dict[Any, int]] = None

    @property
    def ds_size(self) -> int:
        return self.train.shape[1] - 1

    def generate_data(self, choice, flag: str = '') -> pd.DataFrame:
        if hasattr(choice, 'detach'):
            choice = choice.detach().cpu().numpy()
        choice = np.asarray(choice).astype(int).reshape(-1)
        if choice.shape[0] != self.ds_size:
            raise ValueError(f'wrong shape of choice: {choice.shape[0]} vs {self.ds_size}')

        if flag == 'train':
            ds = self.train
        elif flag == 'test':
            ds = self.test
        else:
            ds = self.original

        X = ds.iloc[:, :-1]
        y = ds.iloc[:, -1]
        idx = np.where(choice == 1)[0]
        X_sub = X.iloc[:, idx].astype(np.float64)
        return pd.concat([X_sub, y.astype(np.float64)], axis=1)


def _history_root() -> str:
    p = os.path.join(base_path, 'history')
    os.makedirs(p, exist_ok=True)
    return p


def _cache_dir(task_name: str) -> str:
    d = os.path.join(_history_root(), task_name)
    os.makedirs(d, exist_ok=True)
    return d


def _processed_hdf_path(task_name: str) -> str:
    return os.path.join(_history_root(), f'{task_name}.hdf')


def _infer_label_column(df: pd.DataFrame) -> str:
    for c in ['label', 'price', 'class', 'target', 'y']:
        if c in df.columns:
            return c
    return df.columns[-1]


def _read_hdf_multi(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Cannot find HDF: {path}')
    for key in ['raw', 'df', None]:
        try:
            return pd.read_hdf(path, key=key) if key else pd.read_hdf(path)
        except Exception:
            continue
    raise RuntimeError(f'Failed to read HDF: {path}')


def _standardize_preserve_neg1(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    keep_cols = []
    for col in X.columns:
        s = pd.to_numeric(X[col], errors='coerce')
        if s.isna().all():
            continue
        mask = (s != -1) & (~s.isna())
        if mask.sum() == 0:
            continue
        s = s.astype('float64')
        mu = s[mask].mean()
        sd = s[mask].std(ddof=0)
        if (sd is None) or np.isnan(sd) or sd == 0:
            sd = 1e-6
        s_norm = s.copy()
        s_norm[mask] = (s[mask] - mu) / sd
        s_norm[~mask] = -1
        X[col] = s_norm
        keep_cols.append(col)
    return X[keep_cols] if keep_cols else pd.DataFrame(index=X.index)


def _normalize_labels(y: pd.Series):
    uniques = sorted(y.unique())
    mapping = {old: new for new, old in enumerate(uniques)}
    y_new = y.map(mapping).astype(int)
    return y_new, mapping


def _split_data(X: pd.DataFrame, y: pd.Series, task_type: str,
                context: Optional[pd.Series],
                test_size: float, seed: int):
    stratify = y if task_type == 'cls' else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify
    )
    if context is not None:
        ctx_tr, ctx_te = train_test_split(
            context, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify
        )
    else:
        ctx_tr, ctx_te = None, None
    return X_tr, X_te, y_tr, y_te, ctx_tr, ctx_te


def _save_processed_hdf(out_path: str,
                        raw_df: pd.DataFrame,
                        train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        ctx_train: Optional[pd.Series],
                        ctx_test: Optional[pd.Series],
                        emb_train: Optional[np.ndarray] = None,
                        emb_test: Optional[np.ndarray] = None) -> None:

    if os.path.exists(out_path):
        os.remove(out_path)

    raw_df.to_hdf(out_path, key='raw', format='table')
    train_df.to_hdf(out_path, key='train', format='table')
    test_df.to_hdf(out_path, key='test', format='table')

    if ctx_train is not None:
        pd.DataFrame({"context_text": ctx_train}).to_hdf(out_path, key='context_text_train', format='table')
        pd.DataFrame({"context_text": ctx_test}).to_hdf(out_path, key='context_text_test', format='table')

    if emb_train is not None and emb_test is not None:
        pd.DataFrame(emb_train, index=train_df.index).to_hdf(out_path, key='context_embedding_train', format='table')
        pd.DataFrame(emb_test, index=test_df.index).to_hdf(out_path, key='context_embedding_test', format='table')
    else:
        pd.DataFrame(index=train_df.index).to_hdf(out_path, key='context_embedding_train', format='fixed')
        pd.DataFrame(index=test_df.index).to_hdf(out_path, key='context_embedding_test', format='fixed')


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def _embed_texts(texts, model_name, max_length=128, batch_size=64, device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(map(str, texts[i:i + batch_size]))
            enc = tok(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(device)
            out = mdl(**enc)
            if hasattr(out, "last_hidden_state"):
                vec = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            else:
                vec = out[0][:, 0]
            all_vecs.append(vec.detach().cpu())
    return torch.cat(all_vecs, dim=0).numpy()




if __name__ == '__main__':
    for name in ['pima_indian']:
        fe = build_or_load_feature_env(
            name,
            max_length=128,
            batch_size=128
        )
        print(f'[{name}] task={fe.task_type} shape: train {fe.train.shape}, test {fe.test.shape}, ds_size={fe.ds_size}')
