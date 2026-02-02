import numpy as np
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from utils.tools import cls_metrics, reg_metrics, split_and_encode
from feature import FeatureEvaluator


def gen_tabnet(fe: FeatureEvaluator, valid_size=0.2, random_state=42):
    # ==== 1) 拆分 X / y（直接用 fe 的 DataFrame；需为数值矩阵）====
    label_col = fe.train.columns[-1]
    Xtr = fe.train.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    Xte = fe.test.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y_tr_raw = fe.train[label_col].to_numpy()
    y_te_raw = fe.test[label_col].to_numpy()

    is_cls = (fe.task_type == "cls")

    if is_cls:

        y_tr = y_tr_raw.astype("int64")
        y_te = y_te_raw.astype("int64")

        # ==== 3) 切验证集（分层）====
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            Xtr, y_tr, test_size=valid_size, random_state=random_state, stratify=y_tr
        )

        # ==== 4) 训练（默认参数，不传 metric/verbose）====
        clf = TabNetClassifier(mask_type='entmax')
        clf.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val, y_val)],batch_size=256)

        # ==== 5) 预测 ====
        yhat_tr = clf.predict(Xtr)
        yhat_te = clf.predict(Xte)
        yscore_tr = clf.predict_proba(Xtr)  # (n, C)
        yscore_te = clf.predict_proba(Xte)

        # ==== 6) 指标 ====
        m_tr = cls_metrics(y_tr, yhat_tr, y_score=yscore_tr)
        m_te = cls_metrics(y_te, yhat_te, y_score=yscore_te)

    else:
        # ==== 回归 ====
        # 切验证集（不分层）
        y_tr = y_tr_raw.astype("float32")
        y_te = y_te_raw.astype("float32")
        X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
            Xtr, y_tr, test_size=valid_size, random_state=random_state
        )

        # ✅ 关键：TabNetRegressor 需要二维 targets
        y_tr_sub_2d = y_tr_sub.reshape(-1, 1).astype("float32")
        y_val_2d = y_val.reshape(-1, 1).astype("float32")

        reg = TabNetRegressor(mask_type='entmax')
        reg.fit(X_tr_sub, y_tr_sub_2d, eval_set=[(X_val, y_val_2d)],batch_size=256)

        yhat_tr = reg.predict(Xtr).reshape(-1)
        yhat_te = reg.predict(Xte).reshape(-1)

        m_tr = reg_metrics(y_tr, yhat_tr)
        m_te = reg_metrics(y_te, yhat_te)

    model = clf if is_cls else reg
    if hasattr(model, "eval"):
        model.eval()

    return {
        "train": m_tr,
        "test": m_te,
        "yhat_tr": yhat_tr,
        "yhat_te": yhat_te,
        "model": model
    }