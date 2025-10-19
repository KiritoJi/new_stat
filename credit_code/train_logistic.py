import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ==== 路径 ====
TRAIN = "/Users/krt/krt files/Github/new_stat/data/训练数据集.xlsx"
TEST  = "/Users/krt/krt files/Github/new_stat/data/测试集.xlsx"
SAMPLE= "/Users/krt/krt files/Github/new_stat/data/提交样例.csv"
OUT   = "/Users/krt/krt files/Github/new_stat/outputs"
os.makedirs(OUT, exist_ok=True)

# ==== 工具函数 ====
def normalize_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def add_business_features(df):
    df = df.copy()
    if "credict_used_amount" in df.columns and "credict_limit" in df.columns:
        denom = (df["credict_limit"].replace(0, np.nan)).astype(float)
        df["credit_utilization_ratio"] = (df["credict_used_amount"].astype(float) / denom).fillna(0).clip(0, 10)
    if "amount" in df.columns and "income" in df.columns:
        denom = (df["income"].replace(0, np.nan)).astype(float)
        df["loan_income_ratio"] = (df["amount"].astype(float) / denom).fillna(0).clip(0, 100)
    if "overdue_times" in df.columns and "default_times" in df.columns:
        df["risk_exposure_index"] = df["overdue_times"].fillna(0) + 2*df["default_times"].fillna(0)
    for c in ["last_overdue_months","last_credict_card_months","recent_account_months"]:
        if c in df.columns:
            df[f"inv_{c}"] = (1/(1+df[c].fillna(df[c].median()))).clip(0,1)
    return df

def ks_score(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(np.abs(tpr - fpr))

# ==== 读取数据 ====
train = add_business_features(normalize_columns(pd.read_excel(TRAIN)))
test  = add_business_features(normalize_columns(pd.read_excel(TEST)))
sample = pd.read_csv(SAMPLE)

#plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 显示中文标签，苹果电脑
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

target_col = "target"
cat_cols = [c for c in train.columns if train[c].dtype == "object" and c != target_col]
num_cols = [c for c in train.columns if c not in cat_cols + [target_col]]
if "id" in num_cols: num_cols.remove("id")

X = train[cat_cols + num_cols]
y = train[target_col].astype(int)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ==== 管道 ====
num_tf = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_tf = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])

clf = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=300))
])

clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_valid)[:,1]
auc = roc_auc_score(y_valid, y_prob)
ap = average_precision_score(y_valid, y_prob)
ks = ks_score(y_valid, y_prob)

print(f"[Logistic Regression] AUC={auc:.4f}  AP={ap:.4f}  KS={ks:.4f}")

# ==== 概率输出 ====
prob_test = clf.predict_proba(test[cat_cols + num_cols])[:,1]
sample["target"] = prob_test
out_path = os.path.join(OUT, "submission_logistic_regression.csv")
sample.to_csv(out_path, index=False)
print("✅ 概率预测文件已生成：", out_path)

# ==== 特征重要性 ====
# 获取特征名（OneHot 展开后）
ohe = clf.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, cat_feature_names])

coef = clf.named_steps["clf"].coef_[0]
importance_df = pd.DataFrame({"feature": feature_names, "importance": np.abs(coef)})
importance_df = importance_df.sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(8,6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (Logistic Regression)")
plt.xlabel("Absolute Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "logistic_feature_importance.png"), dpi=200)
plt.close()

print("📊 特征重要性图已保存：logistic_feature_importance.png")
