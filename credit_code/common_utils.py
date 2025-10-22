# -*- coding: utf-8 -*-
"""
common_utils.py
-------------
通用数据预处理与特征工程（强化强相关字段信号 + 样本加权）
供四个训练脚本共享。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# =========================
# 列名标准化
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

# =========================
# 识别高风险贷款用途（数据驱动）
# =========================
def learn_high_risk_purpose(train_df: pd.DataFrame, target_col: str = "target"):
    if "purpose" not in train_df.columns:
        return set()
    grp = train_df.groupby("purpose")[target_col].mean()
    global_mean = train_df[target_col].mean()
    global_std = train_df[target_col].std(ddof=0)
    if pd.isna(global_std) or global_std < 1e-6:
        high = set(grp.sort_values(ascending=False).head(3).index)
    else:
        high = set(grp[grp >= global_mean + global_std].index)
        if len(high) == 0:
            high = set(grp.sort_values(ascending=False).head(2).index)
    return high

# =========================
# 特征工程（比值/倒数/交互/标识/非线性）
# =========================
def add_business_features(df: pd.DataFrame, high_risk_purposes=set()) -> pd.DataFrame:
    df = df.copy()

    # 统一类别字段为小写字符串
    for c in ["purpose", "housing"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()

    # 比值特征
    if "amount" in df.columns and "income" in df.columns:
        inc = df["income"].replace(0, np.nan).astype(float)
        df["loan_income_ratio"] = (df["amount"].astype(float) / inc).fillna(0).clip(0, 100)

    if "credict_used_amount" in df.columns and "credict_limit" in df.columns:
        lim = df["credict_limit"].replace(0, np.nan).astype(float)
        df["credit_utilization_ratio"] = (df["credict_used_amount"].astype(float) / lim).fillna(0).clip(0, 10)

    if "amount" in df.columns and "credict_limit" in df.columns:
        lim = df["credict_limit"].replace(0, np.nan).astype(float)
        df["amount_to_limit"] = (df["amount"].astype(float) / lim).fillna(0).clip(0, 100)

    # 倒数（时间衰减）
    for c in ["last_overdue_months", "last_credict_card_months"]:
        if c in df.columns:
            s = df[c].astype(float)
            df[f"inv_{c}"] = (1.0 / (1.0 + s.fillna(s.median()))).clip(0, 1)

    # 历史不良综合
    if "overdue_times" in df.columns and "default_times" in df.columns:
        df["risk_exposure_index"] = df["overdue_times"].fillna(0).astype(float) + \
                                    2.0 * df["default_times"].fillna(0).astype(float)

    # housing 指示
    if "housing" in df.columns:
        df["is_rent"] = df["housing"].eq("rent").astype(int)
        df["is_own"] = df["housing"].eq("own").astype(int)

    # 高风险用途
    if "purpose" in df.columns:
        df["purpose_risk_flag"] = df["purpose"].isin(high_risk_purposes).astype(int)

    # 非线性加强
    if "loan_income_ratio" in df.columns:
        df["loan_income_ratio_sq"] = df["loan_income_ratio"] ** 2
    if "credit_utilization_ratio" in df.columns:
        df["log_credit_utilization"] = np.log1p(df["credit_utilization_ratio"])
    if "income" in df.columns:
        df["log_income"] = np.log1p(df["income"].astype(float))

    # 交互项
    if "loan_income_ratio" in df.columns and "is_rent" in df.columns:
        df["int_ratio_rent"] = df["loan_income_ratio"] * df["is_rent"]
    if "credit_utilization_ratio" in df.columns and "inv_last_overdue_months" in df.columns:
        df["int_util_recentoverdue"] = df["credit_utilization_ratio"] * df["inv_last_overdue_months"]

    return df

# =========================
# 样本加权（强化强相关字段区域）
# =========================
def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    w = np.ones(len(df), dtype=float)
    if "loan_income_ratio" in df.columns:
        w *= 1.0 + 0.5 * (df["loan_income_ratio"] > df["loan_income_ratio"].median())
    if "credit_utilization_ratio" in df.columns:
        w *= 1.0 + 0.3 * (df["credit_utilization_ratio"] > df["credit_utilization_ratio"].median())
    if "purpose_risk_flag" in df.columns:
        w *= 1.0 + 0.4 * df["purpose_risk_flag"].astype(float)
    return w

# =========================
# KS 指标
# =========================
def ks_score(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))
