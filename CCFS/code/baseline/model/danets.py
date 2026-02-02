import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from baseline.model.danet.DAN_Task import DANetClassifier, DANetRegressor
from utils.tools import cls_metrics, reg_metrics


def gen_danet(fe, seed: int = 324):


    label_col = fe.train.columns[-1]
    X_full = fe.train.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y_full = fe.train[label_col].to_numpy()
    Xte = fe.test.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y_te = fe.test[label_col].to_numpy()

    lr = 8e-3
    weight_decay = 0.0
    layer = 20
    base_outdim = 64
    k = 5
    drop_rate = 0.1
    batch_size = 256
    virtual_batch_size = 256
    max_epochs = 2000
    patience = 500
    valid_size = 0.2

    Xtr, Xval, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=valid_size, random_state=seed,
        stratify=y_full if fe.task_type == "cls" else None
    )

    if fe.task_type == "cls":
        model = DANetClassifier(
            optimizer_params=dict(lr=lr, weight_decay=weight_decay, nus=(0.8, 1.0)),
            scheduler_fn=None, scheduler_params={},
            layer=layer, base_outdim=base_outdim, k=k, drop_rate=drop_rate,
            seed=seed, device_name="auto", verbose=0
        )

        print("===> Start training DANetClassifier ...")

        model.fit(
            X_train=Xtr, y_train=y_tr,
            eval_set=[(Xval, y_val)], eval_name=['valid'], eval_metric=['accuracy'],
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, virtual_batch_size=virtual_batch_size,
            logname=None, resume_dir=None, n_gpu=1
        )


        yhat_tr = model.predict(X_full)
        yhat_te = model.predict(Xte)
        yscore_tr = model.predict_proba(X_full)
        yscore_te = model.predict_proba(Xte)

        m_tr = cls_metrics(y_full, yhat_tr, y_score=yscore_tr)
        m_te = cls_metrics(y_te, yhat_te, y_score=yscore_te)

    else:
        y_full_2d = y_full.reshape(-1, 1).astype(np.float32)
        y_te_2d = y_te.reshape(-1, 1).astype(np.float32)
        Xtr, Xval, y_tr, y_val = train_test_split(
            X_full, y_full_2d, test_size=valid_size, random_state=seed
        )

        model = DANetRegressor(
            std=float(np.std(y_full_2d)) if np.std(y_full_2d) > 0 else 1.0,
            optimizer_params=dict(lr=lr, weight_decay=weight_decay, nus=(0.8, 1.0)),
            scheduler_fn=None, scheduler_params={},
            layer=layer, base_outdim=base_outdim, k=k,
            seed=seed, device_name="auto", verbose=0
        )

        print("===> Start training DANetRegressor ...")
        model.fit(
            X_train=Xtr, y_train=y_tr,
            eval_set=[(Xval, y_val)], eval_name=['valid'], eval_metric=['mse'],
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, virtual_batch_size=virtual_batch_size,
            logname=None, resume_dir=None, n_gpu=1
        )


        yhat_tr = model.predict(X_full).reshape(-1)
        yhat_te = model.predict(Xte).reshape(-1)
        m_tr = reg_metrics(y_full, yhat_tr)
        m_te = reg_metrics(y_te, yhat_te)

    print("✅ Training finished.")

    if hasattr(model, "eval"):
        model.eval()

    return {"train": m_tr, "test": m_te, "yhat_tr": yhat_tr, "yhat_te": yhat_te, "model": model}
