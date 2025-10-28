#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
违约风险预测模型（WOE + PCA + Logistic / ExtraTrees / XGBoost）
-----------------------------------------------------------
输入文件：训练数据集.xlsx、测试集.xlsx、提交样例.csv
输出文件：
  - predict_logistic.csv
  - predict_extratrees.csv
  - predict_xgboost.csv
  - predict_ensemble.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# ===== 1. 文件路径 =====
TRAIN_PATH = "/Users/krt/krt files/Github/new_stat/data/训练数据集.xlsx"
TEST_PATH = "/Users/krt/krt files/Github/new_stat/data/测试集.xlsx"
SAMPLE_PATH = "/Users/krt/krt files/Github/new_stat/data/提交样例.csv"

# ===== 2. 读取数据 =====
train = pd.read_excel(TRAIN_PATH)
test = pd.read_excel(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

id_col = "id"
target_col = "target"

# ===== 3. 定义WOE计算函数 =====
def bin_numeric(x, bins=5):
    try:
        b = pd.qcut(x, bins, duplicates="drop")
        if b.nunique() < 5:
            b = pd.cut(x, 5, duplicates="drop")
    except Exception:
        b = pd.cut(x, 5, duplicates="drop")
    if pd.Series(b).nunique() < 2:
        return pd.Series(["single_bin"] * len(x), index=x.index)
    return b

def woe_iv_for_binned(df_bin, target):
    g = df_bin.groupby("x_bin")[target].agg(["count", "sum"]).rename(columns={"count": "total", "sum": "bad"})
    g["good"] = g["total"] - g["bad"]
    g["bad_prop"] = g["bad"] / g["bad"].sum()
    g["good_prop"] = g["good"] / g["good"].sum()
    g["WOE"] = np.log((g["good_prop"] + 1e-8) / (g["bad_prop"] + 1e-8))
    g["IV"] = (g["good_prop"] - g["bad_prop"]) * g["WOE"]
    return g, g["IV"].sum()

def fit_woe_mapper(df, features, target):
    mapper = {}
    iv_rows = []
    for col in features:
        x = df[col]
        if pd.api.types.is_numeric_dtype(x):
            x_bin = bin_numeric(x, bins=5)
        else:
            x_bin = x.astype(str)
        tmp = pd.DataFrame({"x_bin": x_bin, target: df[target]})
        stats, iv = woe_iv_for_binned(tmp, target)
        mapper[col] = {"map": stats["WOE"].to_dict(), "iv": iv}
        iv_rows.append({"feature": col, "IV": iv})
    iv_df = pd.DataFrame(iv_rows).sort_values("IV", ascending=False).reset_index(drop=True)
    return mapper, iv_df

def apply_woe_mapper(df, mapper):
    res = pd.DataFrame(index=df.index)
    for col, info in mapper.items():
        x = df[col]
        if pd.api.types.is_numeric_dtype(x):
            x_bin = bin_numeric(x, bins=5)
        else:
            x_bin = x.astype(str)
        res[col + "_woe"] = x_bin.astype(str).map(info["map"]).fillna(0)
    return res

# ===== 4. 计算WOE并筛选IV =====
features = [c for c in train.columns if c not in [id_col, target_col]]
mapper, iv_df = fit_woe_mapper(train, features, target_col)

sel = iv_df[(iv_df.IV >= 0.05) & (iv_df.IV <= 0.5)]["feature"].tolist()
if len(sel) < 5:
    sel = iv_df[(iv_df.IV >= 0.02) & (iv_df.IV <= 1.5)]["feature"].tolist()
if len(sel) == 0:
    sel = iv_df.head(10)["feature"].tolist()

X_train_woe = apply_woe_mapper(train[sel], {k: v for k, v in mapper.items() if k in sel})
X_test_woe = apply_woe_mapper(test[sel], {k: v for k, v in mapper.items() if k in sel})
y = train[target_col].astype(int)

# ===== 5. PCA降维 =====
pca = PCA(n_components=min(10, X_train_woe.shape[1]))
X_train_pca = pca.fit_transform(X_train_woe)
X_test_pca = pca.transform(X_test_woe)

# ===== 6. 三种模型训练与预测 =====
models = {
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "extratrees": ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "xgboost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc", random_state=42
    )
}

preds = {}
for name, model in models.items():
    model.fit(X_train_pca, y)
    preds[name] = model.predict_proba(X_test_pca)[:, 1]
    pd.DataFrame({id_col: test[id_col], target_col: preds[name]}).to_csv(f"predict_{name}.csv", index=False)
    print(f"{name} 模型预测文件已生成: predict_{name}.csv")

# ===== 7. 模型融合 =====
preds["ensemble"] = (preds["logistic"] + preds["extratrees"] + preds["xgboost"]) / 3
pd.DataFrame({id_col: test[id_col], target_col: preds["ensemble"]}).to_csv("predict_ensemble.csv", index=False)
print("✅ 融合模型预测文件已生成: predict_ensemble.csv")

# ===== 8. 输出AUC参考（交叉验证） =====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    aucs = []
    for tr, va in cv.split(X_train_pca, y):
        m = model.__class__(**model.get_params())
        m.fit(X_train_pca[tr], y.iloc[tr])
        p = m.predict_proba(X_train_pca[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p))
    print(f"{name} 平均AUC: {np.mean(aucs):.4f}")
