# common_utils.py
import os, json, numpy as np, matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    roc_curve, confusion_matrix, classification_report
)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """业务向特征增强：使用率/比率/倒数特征等"""
    df = df.copy()
    # 信用卡使用率
    if "credict_used_amount" in df.columns and "credict_limit" in df.columns:
        denom = (df["credict_limit"].replace(0, np.nan)).astype(float)
        df["credit_utilization_ratio"] = (df["credict_used_amount"].astype(float) / denom).fillna(0.0).clip(0, 10)
    else:
        df["credit_utilization_ratio"] = 0.0

    # 贷款/收入比
    if "amount" in df.columns and "income" in df.columns:
        denom = (df["income"].replace(0, np.nan)).astype(float)
        df["loan_income_ratio"] = (df["amount"].astype(float) / denom).fillna(0.0).clip(0, 100)
    else:
        df["loan_income_ratio"] = 0.0

    # 风险暴露指数（可按经验调权重）
    if "overdue_times" in df.columns and "default_times" in df.columns:
        df["risk_exposure_index"] = df["overdue_times"].fillna(0).astype(float) + 2.0 * df["default_times"].fillna(0).astype(float)
    else:
        df["risk_exposure_index"] = 0.0

    # 信用卡紧张比例
    if "half_used_credict_card" in df.columns and "total_credict_card_number" in df.columns:
        denom = (df["total_credict_card_number"].replace(0, np.nan)).astype(float)
        df["tight_card_ratio"] = (df["half_used_credict_card"].astype(float) / denom).fillna(0.0).clip(0, 1)
    else:
        df["tight_card_ratio"] = 0.0

    # 最近活动的倒数特征（越近越大）
    for col in ["last_overdue_months", "last_credict_card_months", "recent_account_months"]:
        if col in df.columns:
            df[f"inv_{col}"] = (1.0 / (1.0 + df[col].fillna(df[col].median()).astype(float))).clip(0, 1)
        else:
            df[f"inv_{col}"] = 0.0

    return df

def split_feature_columns(df: pd.DataFrame, target_col="target"):
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != target_col]
    num_cols = [c for c in df.columns if c not in cat_cols + [target_col]]
    if "id" in num_cols:
        num_cols.remove("id")
    return cat_cols, num_cols

def ks_score(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))

def best_thr_by_ks(y_true, y_prob):
    fpr, tpr, th = roc_curve(y_true, y_prob)
    ks_vals = np.abs(tpr - fpr)
    return float(th[np.argmax(ks_vals)])

def evaluate_and_plot(model_name, y_true, y_prob, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    auc_v = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ks = ks_score(y_true, y_prob)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=model_name); plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {model_name} (AUC={auc_v:.3f})"); plt.legend()
    roc_path = os.path.join(out_dir, f"{model_name}_roc.png")
    plt.savefig(roc_path, bbox_inches="tight"); plt.close()

    # PR
    plt.figure()
    plt.plot(rec, prec, label=model_name)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR - {model_name} (AP={ap:.3f})"); plt.legend()
    pr_path = os.path.join(out_dir, f"{model_name}_pr.png")
    plt.savefig(pr_path, bbox_inches="tight"); plt.close()

    # KS
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    pos_cum = (np.cumsum(y_sorted) / np.sum(y_sorted)) if np.sum(y_sorted)>0 else np.zeros_like(y_sorted, float)
    neg_cum = (np.cumsum(1 - y_sorted) / np.sum(1 - y_sorted)) if np.sum(1-y_sorted)>0 else np.zeros_like(y_sorted, float)
    plt.figure()
    plt.plot(pos_cum, label="Positive CDF")
    plt.plot(neg_cum, label="Negative CDF")
    plt.xlabel("Samples sorted by score"); plt.ylabel("Cumulative proportion")
    plt.title(f"KS - {model_name} (KS={ks:.3f})"); plt.legend()
    ks_path = os.path.join(out_dir, f"{model_name}_ks.png")
    plt.savefig(ks_path, bbox_inches="tight"); plt.close()

    thr = best_thr_by_ks(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)

    summary = {
        "model": model_name, "AUC": float(auc_v), "AP": float(ap), "KS": float(ks),
        "best_threshold_KS": float(thr), "confusion_matrix": cm.tolist(),
        "classification_report": rep
    }
    with open(os.path.join(out_dir, f"{model_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, {"roc": roc_path, "pr": pr_path, "ks": ks_path}
