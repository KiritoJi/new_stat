# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

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

# 预处理 & 模型（线性回归输出概率需裁剪到[0,1]）
num_tf = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_tf = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])

clf = Pipeline([("pre", pre), ("clf", LinearRegression())])

# 训练
clf.fit(X_tr, y_tr, clf__sample_weight=w_tr)

# 评估
y_pred = np.clip(clf.predict(X_va), 0, 1)
auc = roc_auc_score(y_va, y_pred)
ap  = average_precision_score(y_va, y_pred)
fpr, tpr, _ = roc_curve(y_va, y_pred)
ks  = np.max(np.abs(tpr - fpr))
print(f"[Linear Regression] AUC={auc:.4f}  AP={ap:.4f}  KS={ks:.4f}")

# 预测导出
prob_test = np.clip(clf.predict(test[cat_cols + num_cols]), 0, 1)
out_df = pd.DataFrame({"id": sample["id"], "target": prob_test})
out_path = os.path.join(OUT, "submission_linear_regression.csv")
out_df.to_csv(out_path, index=False)
print("✅ 概率预测文件：", out_path)

# 特征重要性导出（系数绝对值）
ohe = clf.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, cat_feature_names])
coef = clf.named_steps["clf"].coef_
imp_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)})
imp_df.sort_values("importance", ascending=False).to_csv(os.path.join(OUT, "feature_importance_linear.csv"), index=False)

plt.figure(figsize=(9, 7))
imp_top = imp_df.sort_values("importance", ascending=False).head(30)
plt.barh(imp_top["feature"], imp_top["importance"])
plt.gca().invert_yaxis()
plt.title("Top 30 Feature Importances (Linear)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "linear_feature_importance.png"), dpi=220)
plt.close()
