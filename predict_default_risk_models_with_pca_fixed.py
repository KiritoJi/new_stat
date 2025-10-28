#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版：保持 WOE 分箱区间与训练一致，防止测试集重新分箱导致映射为 0
功能：WOE + IV筛选 + PCA + Logistic/ExtraTrees/XGBoost + 融合预测
输入：
  - 训练集：/Users/krt/krt files/Github/new_stat/data/训练数据集.xlsx
  - 测试集：/Users/krt/krt files/Github/new_stat/data/测试集.xlsx
  - 提交样例：/Users/krt/krt files/Github/new_stat/data/提交样例.csv
输出（当前运行目录）：
  - predict_logistic.csv
  - predict_extratrees.csv
  - predict_xgboost.csv
  - predict_ensemble.csv
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
import math

warnings.filterwarnings("ignore")

# ===== 1. 文件路径 =====
TRAIN_PATH = "/Users/krt/krt files/Github/new_stat/data/训练数据集.xlsx"
TEST_PATH = "/Users/krt/krt files/Github/new_stat/data/测试集.xlsx"
SAMPLE_PATH = "/Users/krt/krt files/Github/new_stat/data/提交样例.csv"

ID_COL = "id"
TARGET_COL = "target"

# ===== 2. 读取数据 =====
train = pd.read_excel(TRAIN_PATH)
test = pd.read_excel(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

# ===== 3. 分箱与 WOE 工具函数 =====
def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0

def bin_numeric_train(x: pd.Series, bins: int = 5) -> pd.Series:
    """
    训练阶段：优先等频分箱；若箱数<5则改为等宽分箱；若仍为单箱则返回 single_bin
    """
    try:
        b = pd.qcut(x, bins, duplicates="drop")
        if b.nunique() < 5:
            b = pd.cut(x, 5, duplicates="drop")
    except Exception:
        b = pd.cut(x, 5, duplicates="drop")
    if pd.Series(b).nunique() < 2:
        return pd.Series(["single_bin"] * len(x), index=x.index)
    return b

def place_into_intervals(val: Any, intervals: list) -> str:
    """
    推理阶段：将数值 val 放入训练时的区间 (pd.Interval) 列表中，返回 str(interval) 作为键；
    若 intervals 中是 'single_bin'，直接返回 'single_bin'。
    """
    if len(intervals) == 1 and intervals[0] == "single_bin":
        return "single_bin"
    # 缺失值直接返回 None，由上层处理为 WOE=0
    if pd.isna(val):
        return None
    for itv in intervals:
        if isinstance(itv, pd.Interval):
            # 默认 closed='right'；考虑边界浮点比较
            left_ok = (val > itv.left) or np.isclose(val, itv.left)
            right_ok = (val <= itv.right) or np.isclose(val, itv.right)
            if left_ok and right_ok:
                return str(itv)
    # 落在训练边界之外，采用最近的边界
    # 若小于最小左端点，归到第一个区间；若大于最大右端点，归到最后一个区间
    # 仅当 intervals 为 Interval 列表时生效
    ivs = [itv for itv in intervals if isinstance(itv, pd.Interval)]
    if len(ivs) >= 1:
        if val < ivs[0].left:
            return str(ivs[0])
        if val > ivs[-1].right:
            return str(ivs[-1])
    return None

def woe_iv_for_binned(df_bin: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, float]:
    g = df_bin.groupby("x_bin")[target].agg(["count", "sum"]).rename(columns={"count": "total", "sum": "bad"})
    g["good"] = g["total"] - g["bad"]
    g["bad_prop"] = g["bad"] / g["bad"].sum()
    g["good_prop"] = g["good"] / g["good"].sum()
    # WOE/IV（加入极小平滑避免除零）
    g["WOE"] = np.log((g["good_prop"] + 1e-8) / (g["bad_prop"] + 1e-8))
    g["IV"] = (g["good_prop"] - g["bad_prop"]) * g["WOE"]
    return g, float(g["IV"].sum())

def fit_woe_mapper(df: pd.DataFrame, features: list, target: str) -> Tuple[Dict, pd.DataFrame]:
    """
    训练阶段：
      返回 mapper：
        {
          feature: {
            "type": "numeric"/"categorical",
            "bins": [Interval 或 'single_bin' 或 类别字符串],
            "woe_map": {str(bin): woe},
            "iv": float
          }
        }
      以及 IV 汇总 DataFrame
    """
    mapper = {}
    iv_rows = []
    for col in features:
        x = df[col]
        if pd.api.types.is_numeric_dtype(x):
            x_bin = bin_numeric_train(x, bins=5)
            tmp = pd.DataFrame({"x_bin": x_bin, target: df[target]})
            stats, iv = woe_iv_for_binned(tmp, target)
            mapper[col] = {
                "type": "numeric",
                "bins": stats.index.tolist() if x_bin.dtype != object else ["single_bin"],
                "woe_map": {str(k): float(v) for k, v in stats["WOE"].to_dict().items()},
                "iv": float(iv),
            }
        else:
            cats = x.astype(str)
            tmp = pd.DataFrame({"x_bin": cats, target: df[target]})
            stats, iv = woe_iv_for_binned(tmp, target)
            mapper[col] = {
                "type": "categorical",
                "bins": stats.index.astype(str).tolist(),
                "woe_map": {str(k): float(v) for k, v in stats["WOE"].to_dict().items()},
                "iv": float(iv),
            }
        iv_rows.append({"feature": col, "IV": mapper[col]["iv"]})
    iv_df = pd.DataFrame(iv_rows).sort_values("IV", ascending=False).reset_index(drop=True)
    return mapper, iv_df

def apply_woe_mapper(df: pd.DataFrame, mapper: Dict, is_train: bool) -> pd.DataFrame:
    """
    训练阶段（is_train=True）：对 df 进行分箱并映射 WOE；
    推理阶段（is_train=False）：严格使用训练阶段保存的 'bins' 做落箱，再映射 WOE。
    新类别/越界/缺失 → 映射为 0（中性 WOE）。
    """
    out = pd.DataFrame(index=df.index)
    for col, info in mapper.items():
        if info["type"] == "numeric":
            if is_train:
                # 训练：重新分箱（用于生成训练 WOE 特征）
                x_bin = bin_numeric_train(df[col], bins=5)
                out[col + "_woe"] = x_bin.astype(str).map(info["woe_map"]).fillna(0.0)
            else:
                # 推理：仅用训练 bins 做落箱，不再重新分箱
                bins = info["bins"]
                keys = df[col].apply(lambda v: place_into_intervals(v, bins))
                out[col + "_woe"] = keys.map(lambda k: info["woe_map"].get(str(k), 0.0))
        else:
            if is_train:
                keys = df[col].astype(str)
            else:
                keys = df[col].astype(str)
            out[col + "_woe"] = keys.map(lambda k: info["woe_map"].get(k, 0.0))
    return out

# ===== 4. 计算 WOE 并筛选 IV =====
features = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
woe_mapper, iv_df = fit_woe_mapper(train, features, TARGET_COL)

# 经验筛选：优先 0.05~0.5，不足再放宽到 0.02~1.5；若仍无则取前10
sel = iv_df[(iv_df["IV"] >= 0.05) & (iv_df["IV"] <= 0.5)]["feature"].tolist()
if len(sel) < 5:
    sel = iv_df[(iv_df["IV"] >= 0.02) & (iv_df["IV"] <= 1.5)]["feature"].tolist()
if len(sel) == 0:
    sel = iv_df.head(10)["feature"].tolist()

# 生成 WOE 特征
X_train_woe = apply_woe_mapper(train[sel], {k: v for k, v in woe_mapper.items() if k in sel}, is_train=True)
X_test_woe = apply_woe_mapper(test[sel], {k: v for k, v in woe_mapper.items() if k in sel}, is_train=False)
y = train[TARGET_COL].astype(int)

# ---- 调试：检查方差，避免信息塌缩 ----
print("训练 WOE 方差：")
print(X_train_woe.var())
print("测试 WOE 方差：")
print(X_test_woe.var())

# ===== 5. PCA 降维（保留最多 10 维或特征数上限） =====
n_comp = int(min(10, X_train_woe.shape[1]))
if n_comp < 1:
    raise ValueError("WOE 特征为空，请检查 IV 筛选阈值或数据质量。")

pca = PCA(n_components=n_comp, random_state=42)
X_train_pca = pca.fit_transform(X_train_woe)
X_test_pca = pca.transform(X_test_woe)

print("PCA 各主成分方差：", np.var(X_train_pca, axis=0))

# ===== 6. 三种模型训练与预测 =====
models = {
    "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "extratrees": ExtraTreesClassifier(n_estimators=400, random_state=42, class_weight="balanced"),
    "xgboost": XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="auc", random_state=42, n_jobs=-1
    ),
}

preds = {}
for name, model in models.items():
    model.fit(X_train_pca, y)
    preds[name] = model.predict_proba(X_test_pca)[:, 1]
    pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: preds[name]}).to_csv(f"predict_{name}.csv", index=False)
    print(f"✅ {name} 预测已保存：predict_{name}.csv")

# ===== 7. 融合预测（简单平均） =====
ensemble = (preds["logistic"] + preds["extratrees"] + preds["xgboost"]) / 3.0
pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: ensemble}).to_csv("predict_ensemble.csv", index=False)
print("✅ 融合预测已保存：predict_ensemble.csv")

# ===== 8. 5折交叉验证 AUC（仅供参考，使用训练集内部CV） =====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    aucs = []
    for tr, va in cv.split(X_train_pca, y):
        m = model.__class__(**model.get_params())
        m.fit(X_train_pca[tr], y.iloc[tr])
        p = m.predict_proba(X_train_pca[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p))
    print(f"{name} 5折AUC：{np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# ===== 9. 融合模型的 AUC（使用训练集CV） =====
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_ensemble = []
for tr, va in cv.split(X_train_pca, y):
    # 三模型在每个折叠内训练并预测
    preds_tr = []
    preds_va = []
    for name, model in models.items():
        m = model.__class__(**model.get_params())
        m.fit(X_train_pca[tr], y.iloc[tr])
        preds_va.append(m.predict_proba(X_train_pca[va])[:, 1])
    # 融合
    ensemble_va = np.mean(preds_va, axis=0)
    auc_ensemble.append(roc_auc_score(y.iloc[va], ensemble_va))

print(f"融合模型(ensemble) 5折AUC：{np.mean(auc_ensemble):.4f} ± {np.std(auc_ensemble):.4f}")

print("完成。")
