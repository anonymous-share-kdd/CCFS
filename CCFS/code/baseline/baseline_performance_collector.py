import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import traceback
import gc

import torch
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from feature import build_or_load_feature_env
from baseline.model import *
# from baseline.model.tabpfn import gen_tabpfn
from utils.tools import robust_save_model

baseline_name = {
    # 'mlp': gen_mlp,
    # 'tab_transformer': gen_tabtransformer,
    # 'tabpfn': gen_tabpfn,
    # 't2g_former': gen_t2g_former,
    # 'tabnet': gen_tabnet,
    # 'danets': gen_danet,
    # 'tabr': gen_tabr,
    # 'tabm': gen_tabm,
}

BASE_PATH = os.path.abspath(os.path.join(HERE, "..", "..", "data"))


def _fmt(m: dict) -> str:
    if not m:
        return "-"
    parts = [f"{k}={float(v):.4f}" for k, v in m.items()]
    return " | ".join(parts)

def _write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _cuda_memory_summary_safe() -> str:
    try:
        if torch.cuda.is_available():
            return torch.cuda.memory_summary()
    except Exception:
        pass
    return "CUDA not available or memory_summary() failed."


def gen_baseline_performance(fe_, task_name_,args):
    hdf_path = fe_.processed_hdf_path
    log_dir = os.path.join(os.path.dirname(hdf_path), task_name_)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "baseline.txt")
    summary_path = os.path.join(log_dir, "summary.json")

    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            if "models" not in summary:
                summary["models"] = {}
        except Exception:
            summary = {
                "task": task_name_,
                "type": fe_.task_type,
                "time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "models": {}
            }
    else:
        summary = {
            "task": task_name_,
            "type": fe_.task_type,
            "time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "models": {}
        }

    print(f"\n==== Baseline run start | task={task_name_} | type={fe_.task_type} ====")
    t0 = time.time()

    file_exists = os.path.exists(log_path)
    mode = "a" if file_exists else "w"

    with open(log_path, mode, encoding="utf-8") as lf:
        if not file_exists:
            lf.write(f"Baseline run | task={task_name_} | type={fe_.task_type} | "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            lf.write("=" * 120 + "\n")
            lf.write(f"{'Baseline':>16} | {'Train metrics':<45} | "
                     f"{'Test metrics':<45} | {'Time(s)':>8}\n")
            lf.write("-" * 120 + "\n")
        else:
            lf.write("\n" + "=" * 120 + "\n")
            lf.write(f"Resume run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            lf.write("-" * 120 + "\n")

        for name_, func in baseline_name.items():
            baseline_dir = os.path.join(log_dir, "baseline", name_)
            os.makedirs(baseline_dir, exist_ok=True)

            print(f"\n-- [{name_}] start --")
            t1 = time.time()
            try:
                if name_ == "ft_transformer":
                    result = func(
                        fe_,
                        dim=args.dim,
                        depth=args.depth,
                        heads=args.heads,
                        epochs=args.epochs,
                    )
                else:
                    result = func(fe_)
                m_tr, m_te = result["train"], result["test"]
                elapsed = time.time() - t1

                line = (f"{name_:>16} | "
                        f"{_fmt(m_tr):<45} | "
                        f"{_fmt(m_te):<45} | "
                        f"{elapsed:>8.2f}\n")
                lf.write(line)

                print(f"[{name_}] Train: {_fmt(m_tr)}")
                print(f"[{name_}] Test : {_fmt(m_te)}")
                print(f"-- [{name_}] end | elapsed {elapsed:.2f}s --")

                model_obj = result.get("model", None)
                model_path = None
                if model_obj is not None:
                    if isinstance(model_obj, torch.nn.Module):
                        model_obj = model_obj.to("cpu")
                    info = robust_save_model(model_obj, baseline_dir, name_)
                    model_path = info.get("model_path")
                    print(f"[{name_}] 💾 Saved ({info.get('kind')}) → {model_path}")

                manifest = {
                    "task": task_name_,
                    "baseline": name_,
                    "model_path": model_path,
                    "task_type": fe_.task_type,
                    "n_train": len(fe_.train),
                    "n_test": len(fe_.test),
                    "time": time.strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(os.path.join(baseline_dir, "manifest.json"), "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)


                yhat_tr = np.asarray(result.get("yhat_tr", []))
                yhat_te = np.asarray(result.get("yhat_te", []))

                if yhat_tr.size and len(yhat_tr) == len(fe_.train):
                    if fe_.task_type == "cls":
                        ser_tr = pd.Series(yhat_tr.astype("int64"),
                                           index=fe_.train.index, name="pred")
                    else:
                        ser_tr = pd.Series(yhat_tr.astype("float32"),
                                           index=fe_.train.index, name="pred")
                    ser_tr.to_hdf(hdf_path, key=f"{name_}_pred_train",
                                  mode="a", format="table")
                    print(f"[{name_}] ✓ saved train preds → {hdf_path}::{name_}_pred_train")

                if yhat_te.size and len(yhat_te) == len(fe_.test):
                    if fe_.task_type == "cls":
                        ser_te = pd.Series(yhat_te.astype("int64"),
                                           index=fe_.test.index, name="pred")
                    else:
                        ser_te = pd.Series(yhat_te.astype("float32"),
                                           index=fe_.test.index, name="pred")
                    ser_te.to_hdf(hdf_path, key=f"{name_}_pred_test",
                                  mode="a", format="table")
                    print(f"[{name_}] ✓ saved test preds  → {hdf_path}::{name_}_pred_test")

                summary["models"][name_] = {
                    "status": "ok",
                    "elapsed_sec": round(elapsed, 3),
                    "train": m_tr,
                    "test": m_te,
                    "model_path": model_path,
                }
                _write_text(summary_path, json.dumps(summary, indent=2))

            except torch.cuda.OutOfMemoryError as e:
                _log_error(log_dir, name_, e, t1)
                elapsed = time.time() - t1
                err_line = f"{name_:>16} | ERROR: CUDA OOM (after {elapsed:.2f}s)\n"
                lf.write(err_line)
                print(err_line.strip())

                summary["models"][name_] = {
                    "status": "oom",
                    "elapsed_sec": round(elapsed, 3),
                }
                _write_text(summary_path, json.dumps(summary, indent=2))

            except Exception as e:
                _log_error(log_dir, name_, e, t1)
                elapsed = time.time() - t1
                err_line = f"{name_:>16} | ERROR: {e} (after {elapsed:.2f}s)\n"
                lf.write(err_line)
                print(err_line.strip())

                summary["models"][name_] = {
                    "status": "error",
                    "elapsed_sec": round(elapsed, 3),
                    "message": str(e)
                }
                _write_text(summary_path, json.dumps(summary, indent=2))

            finally:
                try:
                    del result
                except Exception:
                    pass
                try:
                    del model_obj
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[{name_}] 🧹 GPU cache cleared after baseline.")

        total = time.time() - t0
        lf.write("-" * 120 + "\n")
        lf.write(f'heads = {args.heads} depth = {args.depth} dim = {args.dim}')
        lf.write("-" * 120 + "\n")
        lf.write(f"Total elapsed: {total:.2f}s\n")

        prev_total = summary.get("total_elapsed_sec", 0.0)
        summary["total_elapsed_sec"] = round(prev_total + total, 3)
        _write_text(summary_path, json.dumps(summary, indent=2))

    print(f"\n==== Baseline run end | total {total:.2f}s ====")
    print(f"[log] saved → {log_path}")
    print(f"[summary] saved → {summary_path}")
    return None


def process(task_name_, args):
    fe = build_or_load_feature_env(task_name_)
    _ = gen_baseline_performance(fe, task_name_,args)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--task", type=str, default="")
    args = parser.parse_args()


    task_list = [
    # 'adult',
    'pima_indian',
    # 'higgs',
    # 'brfss_diabetes',
    # 'lifestyle_risk',
    # 'credit_default',
    # 'wine_red',
    # 'wine_white',
    # 'mus_scRNA_seq',
    # 'mice_protein',
    # 'mnist',



    # 'openml_586',
    # 'openml_588',
    # 'openml_589',
    # 'openml_607',
    # 'openml_616',
    # 'openml_618',
    # 'openml_620',
    # 'openml_637',

    # 'parkinson',
    # 'apt_rental_price',
    # 'ct_slice_loc',
]
    dim_list = [128, 256, 512]
    depth_list = [2,3,4,5,6]
    head_list = [2,3,4,5,5,6,7,8]
    task_name = args.task
    for dim in dim_list:
        for depth in depth_list:
            for head in head_list:
                args.dim = dim
                args.depth = depth
                args.heads = head

                process(task_name, args)
