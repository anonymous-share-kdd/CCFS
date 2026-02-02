import numpy as np
import torch
import time
from tabpfn import TabPFNClassifier, TabPFNRegressor
from feature import FeatureEvaluator
from tabpfn_extensions.many_class import ManyClassClassifier

from utils.tools import cls_metrics, reg_metrics



def gen_tabpfn(fe: FeatureEvaluator):
    X_tr = fe.train.iloc[:, :-1].values
    y_tr = fe.train.iloc[:, -1].values
    X_te = fe.test.iloc[:, :-1].values
    y_te = fe.test.iloc[:, -1].values

    print(f"[TabPFN] Train: {X_tr.shape}, Test: {X_te.shape}, Task={fe.task_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()

    # ===== Classification =====
    if fe.task_type == "cls":
        n_classes = len(np.unique(y_tr))
        print(f"[TabPFN] n_classes = {n_classes}")


        base  = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
        try:
            print("[TabPFN] Trying plain TabPFNClassifier...")
            model = base
            model.fit(X_tr, y_tr)
            used_many_class = False
        except ValueError as e:
            msg = str(e)
            if "Number of classes" in msg and "maximal number of classes supported by TabPFN" in msg:
                print("[TabPFN] Too many classes for base TabPFN, "
                      "falling back to ManyClassClassifier wrapper.")
                base2 = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
                model = ManyClassClassifier(
                    estimator=base2,
                    alphabet_size = 10,
                    random_state=42,
                    verbose=0,
                )
                model.fit(X_tr, y_tr)
                used_many_class = True
            else:
                raise

        yhat_tr, yscore_tr = _adaptive_predict_cls(
            model,
            X_tr,
            device=device,
            want_proba=True,
            init_bs=256,
            min_bs=32,
        )
        yhat_te, yscore_te = _adaptive_predict_cls(
            model,
            X_te,
            device=device,
            want_proba=True,
            init_bs=256,
            min_bs=32,
        )

        m_tr = cls_metrics(y_tr, yhat_tr, y_score=yscore_tr)
        m_te = cls_metrics(y_te, yhat_te, y_score=yscore_te)

        # ===== Regression =====
    else:
        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        model.fit(X_tr, y_tr)

        yhat_tr = _adaptive_predict_reg(
            model,
            X_tr,
            device=device,
            init_bs=256,
            min_bs=32,
        )
        yhat_te = _adaptive_predict_reg(
            model,
            X_te,
            device=device,
            init_bs=256,
            min_bs=32,
        )

        m_tr = reg_metrics(y_tr, yhat_tr)
        m_te = reg_metrics(y_te, yhat_te)

    elapsed = time.time() - start
    print(f"[TabPFN] Done | Time={elapsed:.2f}s | Train={m_tr} | Test={m_te}")

    if hasattr(model, "eval"):
        model.eval()

    return {
        "train": m_tr,
        "test": m_te,
        "yhat_tr": np.array(yhat_tr),
        "yhat_te": np.array(yhat_te),
    }
