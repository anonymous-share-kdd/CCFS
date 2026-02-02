from feature import FeatureEvaluator
from sklearn.neural_network import MLPClassifier, MLPRegressor
from utils.tools import cls_metrics, reg_metrics
import numpy as np

def gen_mlp(fe: FeatureEvaluator):


    X_tr = fe.train.iloc[:, :-1].values
    y_tr = fe.train.iloc[:, -1].values
    X_te = fe.test.iloc[:, :-1].values
    y_te = fe.test.iloc[:, -1].values

    print(f"[MLP] Train: {X_tr.shape}, Test: {X_te.shape}, Task={fe.task_type}")

    if fe.task_type == "cls":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=42,
            verbose=False
        )
        model.fit(X_tr, y_tr)

        yhat_tr = model.predict(X_tr)
        yhat_te = model.predict(X_te)

        # 概率输出：用于计算 AUC
        yscore_tr = None
        yscore_te = None
        if hasattr(model, "predict_proba"):
            yscore_tr = model.predict_proba(X_tr)
            yscore_te = model.predict_proba(X_te)
        elif hasattr(model, "decision_function"):
            yscore_tr = model.decision_function(X_tr)
            yscore_te = model.decision_function(X_te)


        m_tr = cls_metrics(y_tr, yhat_tr, y_score=yscore_tr)
        m_te = cls_metrics(y_te, yhat_te, y_score=yscore_te)

        results = {
            "train": m_tr,
            "test": m_te,
            "yhat_tr": yhat_tr,
            "yhat_te": yhat_te,
        }

    else:
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=42,
            verbose=False
        )
        model.fit(X_tr, y_tr)

        yhat_tr = model.predict(X_tr)
        yhat_te = model.predict(X_te)

        m_tr = reg_metrics(y_tr, yhat_tr)
        m_te = reg_metrics(y_te, yhat_te)

    if hasattr(model, "eval"):
        model.eval()

    return {
            "train": m_tr,
            "test": m_te,
            "yhat_tr": yhat_tr,
            "yhat_te": yhat_te,
            "model": model
        }