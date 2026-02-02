import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
BASE_PATH = os.path.abspath(os.path.join(HERE, "..", "..", "data"))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from feature import FeatureEvaluator
from utils.tools import cls_metrics, reg_metrics
from model.modules import CCFS
from utils.datasets import CCFSDataset


def format_metric_dict(m_dict):
    return " | ".join([f"{k}={v:.4f}" for k, v in m_dict.items()])


def process(args):
    task_name = args.task_name

    pkl_path = f'{BASE_PATH}/history/{task_name}/fe.pkl'
    with open(pkl_path, 'rb') as f:
        fe: FeatureEvaluator = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task_type == 'reg':
        output_dim = 1
        criterion = nn.MSELoss()
    else:
        num_classes = fe.train.iloc[:, -1].nunique()
        if num_classes == 2:
            output_dim = 1
            criterion = nn.BCEWithLogitsLoss()
        else:
            output_dim = num_classes
            criterion = nn.CrossEntropyLoss()

    print(f"Task Type: {task_type} | Columns: {num_columns} | Output Dim: {output_dim}")

    batch_size = args.batch_size

    train_ds = CCFSDataset(fe.train, fe.context_emb_train, task_type)
    test_ds = CCFSDataset(fe.test, fe.context_emb_test, task_type)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = CCFS(
        num_columns=num_columns,
        task_type=task_type,
        args=args,
        output_dim=output_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _predict(loader):
        model.eval()
        preds = []
        scores = []
        trues = []

        with torch.no_grad():
            for batch in loader:
                x_tab = batch['x_tab'].to(device)
                x_text = batch['x_text'].to(device)
                yb = batch['y'].to(device)

                logits, gate = model(x_tab, x_text, return_gate=True)

                trues.append(yb.cpu().numpy())
                if task_type == "cls":
                    if output_dim == 1:
                        prob = torch.sigmoid(logits)
                        pred = (prob > 0.5).long()
                        preds.append(pred.squeeze(-1).cpu().numpy())
                        scores.append(prob.squeeze(-1).cpu().numpy())
                    else:
                        prob = torch.softmax(logits, dim=-1)
                        pred = prob.argmax(dim=-1)
                        preds.append(pred.cpu().numpy())
                else:
                    pred = logits.squeeze(-1)
                    preds.append(pred.cpu().numpy())

        y_true = np.concatenate(trues, axis=0) if trues else np.array([])
        y_pred = np.concatenate(preds, axis=0) if preds else np.array([])

        y_score = None
        if task_type == "cls" and output_dim == 1 and len(scores) > 0:
            y_score = np.concatenate(scores, axis=0)

        return y_true, y_pred, y_score

    epochs = args.epochs
    lambda_sparse = args.lambda_sparse

    best_score = -float('inf')
    best_epoch = -1
    best_train_metrics = {}
    best_test_metrics = {}

    start_time_global = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            x_tab = batch['x_tab'].to(device)
            x_text = batch['x_text'].to(device)
            y = batch['y'].to(device)

            optimizer.zero_grad()
            logits, w = model(x_tab, x_text, return_gate=True)

            if task_type == 'reg' or output_dim == 1:
                task_loss = criterion(logits.squeeze(), y.float())
            else:
                task_loss = criterion(logits, y)

            l1_penalty = w.mean()
            loss = task_loss + lambda_sparse * l1_penalty

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        y_tr_true, yhat_tr, yscore_tr = _predict(train_loader)
        y_te_true, yhat_te, yscore_te = _predict(test_loader)

        if task_type == 'cls':
            tr_metrics = cls_metrics(y_tr_true, yhat_tr, y_score=yscore_tr)
            te_metrics = cls_metrics(y_te_true, yhat_te, y_score=yscore_te)
        else:
            tr_metrics = reg_metrics(y_tr_true, yhat_tr)
            te_metrics = reg_metrics(y_te_true, yhat_te)

        tr_str = format_metric_dict(tr_metrics)
        te_str = format_metric_dict(te_metrics)
        epoch_time = time.time() - epoch_start
        print(f"{epoch + 1:5d} | {avg_loss:8.4f} | {tr_str:<55} | {te_str:<55} | {epoch_time:8.2f}s")

        if task_type == 'reg':
            current_score = te_metrics.get('1-rae', te_metrics.get('1-RAE', -float('inf')))
        elif output_dim == 1:
            current_score = te_metrics.get('F1_w', -float('inf'))
        else:
            current_score = te_metrics.get('F1_micro', te_metrics.get('f1_micro', -float('inf')))

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            best_train_metrics = tr_metrics.copy()
            best_test_metrics = te_metrics.copy()

    total_time = time.time() - start_time_global
    print("-" * 150)
    print(f"Training Finished. Best Epoch: {best_epoch} | Best Score: {best_score:.4f} | Total Time(s): {total_time:.2f}")
    print(f"Best Train: {format_metric_dict(best_train_metrics)}")
    print(f"Best  Test: {format_metric_dict(best_test_metrics)}")


if __name__ == '__main__':
    import argparse

    task_list = [
        'adult',
        'pima_indian',
        'brfss_diabetes',
        'lifestyle_risk',
        'credit_default',
        'wine_red',
        'wine_white',
        'mice_protein',
        'mnist',
        'openml_586',
        'openml_588',
        'openml_589',
        'openml_607',
        'openml_616',
        'openml_618',
        'openml_620',
        'openml_637',
        'parkinson',
        'apt_rental_price',
        'ct_slice_loc',
    ]

    parser = argparse.ArgumentParser(description="CCFS Task Runner")
    parser.add_argument('--task_name', type=str, default='adult', choices=task_list)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lambda_sparse', type=float, default=0.0)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--ffn_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()
    process(args)
