# -*- coding: utf-8 -*-
"""
Run all models and fuse predictions.
Outputs (kept as before):
- submission_ensemble.csv  : 融合后的提交
- model_comparison_plot.png: 各模型指标对比
This script assumes single-model scripts have already produced their submissions
and optional OOF files under OUT.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ======== Paths (keep original style with robust fallback) ========
BASE_DIR = "/Users/krt/krt files/Github/new_stat/credit_code"
ROOT_DIR = os.path.dirname(BASE_DIR) if os.path.exists(BASE_DIR) else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(OUT, exist_ok=True)

# ======== Helper ========
def ks_score(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))

def read_submission(name):
    p = os.path.join(OUT, f"submission_{name}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        assert {"id","target"}.issubset(df.columns), f"{p} must have id,target"
        return df
    return None

def read_oof(name):
    p = os.path.join(OUT, f"oof_{name}.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        return df
    return None

def safe_metric_from_oof(df):
    try:
        y = df["target"].values
        o = df["oof"].values
        auc = roc_auc_score(y, o)
        ap = average_precision_score(y, o)
        ks = ks_score(y, o)
        return {"AUC": auc, "AP": ap, "KS": ks}
    except Exception:
        return {"AUC": np.nan, "AP": np.nan, "KS": np.nan}

def main():
    print(">>> Run all & ensemble")
    # Load individual submissions
    names = ["logistic", "extratrees", "xgboost"]
    subs = {n: read_submission(n) for n in names}
    subs = {k:v for k,v in subs.items() if v is not None}
    if not subs:
        print("⚠️ 未发现单模型提交文件，请先运行各训练脚本（train_*.py）。")
        return

    # Align by id
    ids = None
    for n, df in subs.items():
        ids = df["id"].values if ids is None else ids
        if not np.array_equal(ids, df["id"].values):
            df.sort_values("id", inplace=True); df.reset_index(drop=True, inplace=True)
    # read sample to keep original order if needed
    sample_path = os.path.join(DATA_DIR, "提交样例.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        order = sample["id"].values
        for n in subs:
            subs[n] = subs[n].set_index("id").loc[order].reset_index()

    # Read OOF for weighting
    metrics = {}
    for n in subs.keys():
        oof_df = read_oof(n)
        if oof_df is not None:
            metrics[n] = safe_metric_from_oof(oof_df)
        else:
            metrics[n] = {"AUC": np.nan, "AP": np.nan, "KS": np.nan}

    # Compute weights based on AUC (fallback to equal)
    aucs = {k: v["AUC"] for k,v in metrics.items() if not np.isnan(v["AUC"])}
    if aucs:
        # emphasize better models using softmax of AUC*10
        score_vec = np.array([aucs[k] for k in subs.keys()])
        w = np.exp(score_vec*10)  # sharp
        w = w / w.sum()
        weights = {k: float(w[i]) for i,k in enumerate(subs.keys())}
    else:
        k = len(subs)
        weights = {n: 1.0/k for n in subs.keys()}

    print("[Weights]", json.dumps(weights, ensure_ascii=False, indent=2))

    # Weighted average ensemble
    ens = None
    for n, df in subs.items():
        p = df["target"].values.astype(float)
        ens = p*weights[n] if ens is None else ens + p*weights[n]

    sub_ens = pd.DataFrame({"id": subs[next(iter(subs))]["id"].values, "target": ens})
    ens_path = os.path.join(OUT, "submission_ensemble.csv")
    sub_ens.to_csv(ens_path, index=False, float_format="%.8f", encoding="utf-8")
    print(f"✅ 已生成融合预测文件：{ens_path}")

    # Draw comparison plot
    dfc = []
    for n, m in metrics.items():
        dfc.append({"model": n, **m})
    dfc = pd.DataFrame(dfc).set_index("model")
    # Add Ensemble metrics if we can evaluate on train
    train_path = os.path.join(DATA_DIR, "训练数据集.xlsx")
    if os.path.exists(train_path):
        # If we have OOFs for all models, approximate ensemble OOF by weighted sum of OOF
        have_all_oof = all(read_oof(n) is not None for n in subs.keys())
        if have_all_oof:
            oofs = [read_oof(n).set_index("id")["oof"] for n in subs.keys()]
            ids_train = oofs[0].index
            oof_mat = np.vstack([s.loc[ids_train].values for s in oofs]).T
            w = np.array([weights[n] for n in subs.keys()])
            oof_ens = (oof_mat @ w)
            y = read_oof(next(iter(subs.keys())))["target"].values
            dfc.loc["ensemble","AUC"] = roc_auc_score(y, oof_ens)
            dfc.loc["ensemble","AP"]  = average_precision_score(y, oof_ens)
            dfc.loc["ensemble","KS"]  = ks_score(y, oof_ens)

    # Plot
    if not dfc.empty:
        cols = ["AUC","AP","KS"]
        fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 4))
        if len(cols)==1: axes=[axes]
        for i,c in enumerate(cols):
            if c in dfc.columns:
                ax=axes[i]
                dfc[c].plot(kind="bar", ax=ax)
                ax.set_title(c); ax.set_xlabel("Model"); ax.set_ylabel(c)
                ax.grid(axis="y", linestyle="--", alpha=0.5)
                for j,v in enumerate(dfc[c].values):
                    if not np.isnan(v):
                        ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        plt.suptitle("各模型性能对比（含融合）")
        plt.tight_layout(rect=[0,0,1,0.93])
        plot_path = os.path.join(OUT, "model_comparison_plot.png")
        plt.savefig(plot_path, dpi=220); plt.close()
        print(f"📈 模型性能对比图已保存：{plot_path}")
    else:
        print("⚠️ 无对比数据，未绘图。")

    print("\n🎯 全流程完成。输出目录：", OUT)

if __name__ == "__main__":
    main()
