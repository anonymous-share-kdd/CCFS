import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

def _to_1d(a, dtype=None):
    x = np.asarray(a)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if dtype is not None:
        x = x.astype(dtype)
    return x.ravel()

def relative_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    num = np.abs(y_true - y_pred).sum()
    den = np.abs(y_true - y_true.mean()).sum() + 1e-12
    return num / den

def cls_metrics(y_true, y_pred, y_score=None):

    y_true = _to_1d(y_true, int)
    y_pred = _to_1d(y_pred, int)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Length mismatch between y_true and y_pred")
    classes = np.unique(y_true)
    n_classes = classes.size

    if n_classes == 2:
        prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec_w  = recall_score(y_true, y_pred,   average='weighted', zero_division=0)
        f1_w   = f1_score(y_true, y_pred,       average='weighted', zero_division=0)
        auc_w  = np.nan
        if y_score is not None:
            y_score = np.asarray(y_score)
            try:
                if y_score.ndim == 2 and y_score.shape[1] >= 2:
                    auc_w = roc_auc_score(y_true, y_score[:, 1])
                elif y_score.ndim == 1:
                    auc_w = roc_auc_score(y_true, y_score)
            except Exception:
                auc_w = np.nan
        return {'Prec_w': float(prec_w), 'Rec_w': float(rec_w),
                'F1_w': float(f1_w), 'AUC_w': float(auc_w) if np.isfinite(auc_w) else float('nan')}
    else:
        prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_macro  = recall_score(y_true, y_pred,   average='macro', zero_division=0)
        f1_micro   = f1_score(y_true, y_pred,       average='micro', zero_division=0)
        f1_macro   = f1_score(y_true, y_pred,       average='macro', zero_division=0)
        return {'Prec_macro': float(prec_macro), 'Rec_macro': float(rec_macro),
                'F1_micro': float(f1_micro), 'F1_macro': float(f1_macro)}

def reg_metrics(y_true, y_pred):
    y_true = _to_1d(y_true, float)
    y_pred = _to_1d(y_pred, float)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Length mismatch between y_true and y_pred")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    rae = relative_absolute_error(y_true, y_pred)
    return {
        '1-MAE': float(1.0 - mae),
        '1-MSE': float(1.0 - mse),
        '1-RAE': float(1.0 - rae),
        '1-RMSE': float(1.0 - rmse),
    }


