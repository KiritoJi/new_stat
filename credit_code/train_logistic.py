# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.base import clone

# ===============================
# 路径设置
# ===============================
BASE_DIR = "/Users/krt/krt files/Github/new_stat/credit_code"
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUT, exist_ok=True)

TRAIN = os.path.join(DATA_DIR, "训练数据集.xlsx")
TEST = os.path.join(DATA_DIR, "测试集.xlsx")
SAMPLE = os.path.join(DATA_DIR, "提交样例.csv")

# ===============================
# 工具函数：列名统一 & 高风险purpose学习
# ===============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def learn_high_risk_purpose(train_df: pd.DataFrame, target_col: str):
    if "purpose" not in train_df.columns:
        return set()
    grp = train_df.groupby("purpose")[target_col].mean()
    global_mean = train_df[target_col].mean()
    global_std = train_df[target_col].std(ddof=0)
    # 阈值 = 全局均值 + 1σ；若σ极小，则取 TopN(3)
    if pd.isna(global_std) or global_std < 1e-6:
        high = set(grp.sort_values(ascending=False).head(3).index)
    else:
        high = set(grp[grp >= global_mean + global_std].index)
        if len(high) == 0:  # 保底至少2个
            high = set(grp.sort_values(ascending=False).head(2).index)
    return high

# ===============================
# 特征工程（新方法：比值/倒数/交互/标识）
# ===============================
def add_business_features(df: pd.DataFrame, high_risk_purposes=set()) -> pd.DataFrame:
    df = df.copy()
    # 统一类型
    for c in ["purpose", "housing"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()

    # 比值类
    if "amount" in df.columns and "income" in df.columns:
        inc = df["income"].replace(0, np.nan).astype(float)
        df["loan_income_ratio"] = (df["amount"].astype(float) / inc).fillna(0).clip(0, 100)

    if "credict_used_amount" in df.columns and "credict_limit" in df.columns:
        lim = df["credict_limit"].replace(0, np.nan).astype(float)
        df["credit_utilization_ratio"] = (df["credict_used_amount"].astype(float) / lim).fillna(0).clip(0, 10)

    if "amount" in df.columns and "credict_limit" in df.columns:
        lim = df["credict_limit"].replace(0, np.nan).astype(float)
        df["amount_to_limit"] = (df["amount"].astype(float) / lim).fillna(0).clip(0, 100)

    # 倒数类（近期效应）
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

    # purpose 高风险标识（数据驱动）
    if "purpose" in df.columns:
        df["purpose_risk_flag"] = df["purpose"].isin(high_risk_purposes).astype(int)

    # 交互项（线性模型受益明显）
    if "loan_income_ratio" in df.columns and "is_rent" in df.columns:
        df["int_ratio_rent"] = df["loan_income_ratio"] * df["is_rent"]
    if "credit_utilization_ratio" in df.columns and "inv_last_overdue_months" in df.columns:
        df["int_util_recentoverdue"] = df["credit_utilization_ratio"] * df["inv_last_overdue_months"]

    # log 收入（控制偏度）
    if "income" in df.columns:
        df["log_income"] = np.log1p(df["income"].astype(float))

    return df

def ks_score(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))

# ===============================
# 数据读取 & 特征学习
# ===============================
train_raw = normalize_columns(pd.read_excel(TRAIN, sheet_name=0))
test_raw = normalize_columns(pd.read_excel(TEST, sheet_name=0))
target_col = "target"

high_risk_purposes = learn_high_risk_purpose(train_raw, target_col)
train = add_business_features(train_raw, high_risk_purposes)
test = add_business_features(test_raw, high_risk_purposes)
sample = pd.read_csv(SAMPLE)

# 拆列
cat_cols = [c for c in train.columns if train[c].dtype == "object" and c != target_col]
num_cols = [c for c in train.columns if c not in cat_cols + [target_col]]
if "id" in num_cols: num_cols.remove("id")

X = train[cat_cols + num_cols].reset_index(drop=True)
y = train[target_col].astype(int).reset_index(drop=True)

# ===============================
# 预处理 & 模型
# ===============================
num_tf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_tf = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", num_tf, num_cols),
    ("cat", cat_tf, cat_cols)
])

pipeline = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=600,
        C=1.0
    ))
])

# ===============================
# 10 次随机划分 + 参数搜索
# ===============================
cv_inner = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
param_grid = {
    "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "clf__penalty": ["l1", "l2"]
}

print("🔁 Logistic 回归：开始 10 次随机划分 + 网格搜索调参（指标：AUC）...")
grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv_inner,
    n_jobs=-1,
    refit=True,
    verbose=0
)
grid.fit(X, y)
best_model = grid.best_estimator_
print(f"✅ 最优参数：{grid.best_params_}，平均AUC={grid.best_score_:.4f}")

# 使用新的随机划分评估最佳模型的稳定性（输出 AUC/AP/KS 均值）
cv_eval = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=123)
auc_scores, ap_scores, ks_scores = [], [], []
for fold_idx, (tr_idx, val_idx) in enumerate(cv_eval.split(X, y), 1):
    model = clone(best_model)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    y_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
    y_val = y.iloc[val_idx]
    auc_scores.append(roc_auc_score(y_val, y_prob))
    ap_scores.append(average_precision_score(y_val, y_prob))
    ks_scores.append(ks_score(y_val, y_prob))

auc = float(np.mean(auc_scores))
ap = float(np.mean(ap_scores))
ks = float(np.mean(ks_scores))
print(f"[Logistic Regression] CV AUC={auc:.4f}  AP={ap:.4f}  KS={ks:.4f}")

# 最终模型：在全部训练数据上重拟合
best_model.fit(X, y)

# 预测 & 导出（基于重训练后的模型）
prob_test = best_model.predict_proba(test[cat_cols + num_cols])[:, 1]
prob_test = np.where(prob_test < 0.1, prob_test / 10.0, prob_test)
sample["target"] = prob_test
out_path = os.path.join(OUT, "submission_logistic_regression.csv")
sample.to_csv(out_path, index=False)
print("✅ 概率预测文件：", out_path)

# 特征重要性（系数绝对值 Top30），基于重训练后的模型
ohe = best_model.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, cat_feature_names])
coef = best_model.named_steps["clf"].coef_[0]
imp_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)}).sort_values("importance", ascending=False).head(30)

plt.figure(figsize=(9, 7))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.title("Top 30 Feature Importances (Logistic)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "logistic_feature_importance.png"), dpi=220)
plt.close()
print("📊 特征重要性图已保存。")
