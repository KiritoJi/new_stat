# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from xgboost import XGBClassifier

from common_utils import (
    normalize_columns, learn_high_risk_purpose, add_business_features,
    compute_sample_weights, ks_score
)

# 路径
BASE_DIR = "/Users/krt/krt files/Github/new_stat/credit_code"
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

TRAIN = os.path.join(DATA_DIR, "训练数据集.xlsx")
TEST = os.path.join(DATA_DIR, "测试集.xlsx")
SAMPLE = os.path.join(DATA_DIR, "提交样例.csv")
target_col = "target"

# 读数 & 特征工程
train_raw = normalize_columns(pd.read_excel(TRAIN, sheet_name=0))
test_raw = normalize_columns(pd.read_excel(TEST, sheet_name=0))
high_risk = learn_high_risk_purpose(train_raw, target_col)
train = add_business_features(train_raw, high_risk)
test = add_business_features(test_raw, high_risk)
sample = pd.read_csv(SAMPLE)

# 列拆分
cat_cols = [c for c in train.columns if train[c].dtype == "object" and c != target_col]
num_cols = [c for c in train.columns if c not in cat_cols + [target_col]]
if "id" in num_cols: num_cols.remove("id")

X, y = train[cat_cols + num_cols], train[target_col].astype(int)
w = compute_sample_weights(train)

X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
    X, y, w, test_size=0.25, random_state=42, stratify=y
)

# 预处理 & 模型
num_tf = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_tf = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])

pos = int(np.sum(y_tr == 1)); neg = int(np.sum(y_tr == 0))
spw = float(neg / max(pos, 1))

clf = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    ))
])

# 训练
clf.fit(X_tr, y_tr, clf__sample_weight=w_tr)

# 评估
y_prob = clf.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, y_prob)
ap  = average_precision_score(y_va, y_prob)
fpr, tpr, _ = roc_curve(y_va, y_prob)
ks  = np.max(np.abs(tpr - fpr))
print(f"[XGBoost] AUC={auc:.4f}  AP={ap:.4f}  KS={ks:.4f}")

# 预测导出
prob_test = clf.predict_proba(test[cat_cols + num_cols])[:, 1]
out_df = pd.DataFrame({"id": sample["id"], "target": prob_test})
out_path = os.path.join(OUT, "submission_xgboost.csv")
out_df.to_csv(out_path, index=False)
print("✅ 概率预测文件：", out_path)

# 特征重要性导出
ohe = clf.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, cat_feature_names])
imp = clf.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
imp_df.sort_values("importance", ascending=False).to_csv(os.path.join(OUT, "feature_importance_xgboost.csv"), index=False)

plt.figure(figsize=(9, 7))
imp_top = imp_df.sort_values("importance", ascending=False).head(30)
plt.barh(imp_top["feature"], imp_top["importance"])
plt.gca().invert_yaxis()
plt.title("Top 30 Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "xgboost_feature_importance.png"), dpi=220)
plt.close()
