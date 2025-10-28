# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

from common_utils import (
    normalize_columns, select_core_features,
    add_business_features, compute_sample_weights, ks_score
)

BASE_DIR = "/Users/krt/krt files/Github/new_stat/credit_code"
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

TRAIN = os.path.join(DATA_DIR, "训练数据集.xlsx")
TEST = os.path.join(DATA_DIR, "测试集.xlsx")
SAMPLE = os.path.join(DATA_DIR, "提交样例.csv")
target_col = "target"

# ===============================
# 数据读取与清洗
# ===============================
train_raw = normalize_columns(pd.read_excel(TRAIN))
test_raw = normalize_columns(pd.read_excel(TEST))
train_raw = select_core_features(train_raw)
test_raw = select_core_features(test_raw)

train = add_business_features(train_raw)
test = add_business_features(test_raw)
sample = pd.read_csv(SAMPLE)

num_cols = [c for c in train.columns if c not in ["id", target_col]]
X, y = train[num_cols], train[target_col].astype(int)
w = compute_sample_weights(train)

X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
    X, y, w, test_size=0.25, random_state=42, stratify=y
)

# ===============================
# 建模与评估
# ===============================
pre = Pipeline([("imp", SimpleImputer(strategy="median"))])
clf = Pipeline([
    ("pre", pre),
    ("clf", ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])
clf.fit(X_tr, y_tr, clf__sample_weight=w_tr)

y_prob = clf.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, y_prob)
ap = average_precision_score(y_va, y_prob)
ks = ks_score(y_va, y_prob)
print(f"[ExtraTrees] AUC={auc:.4f}  AP={ap:.4f}  KS={ks:.4f}")

# ===============================
# 结果输出
# ===============================
prob_test = clf.predict_proba(test[num_cols])[:, 1]
pd.DataFrame({"id": sample["id"], "target": prob_test}).to_csv(
    os.path.join(OUT, "submission_extra_trees.csv"), index=False
)

imp = pd.DataFrame({"feature": num_cols, "importance": clf.named_steps["clf"].feature_importances_})
imp.to_csv(os.path.join(OUT, "feature_importance_extratrees.csv"), index=False)
