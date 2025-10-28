# -*- coding: utf-8 -*-
"""
run_all.py
-----------
执行全流程建模：运行四个模型、比较性能、融合最优模型、
生成特征重要性综合表和Top10可视化。
"""

import os, sys, subprocess, re, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ===============================
# 路径配置
# ===============================
BASE_DIR = "/Users/krt/krt files/Github/new_stat/credit_code"
ROOT_DIR = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

scripts = [
    "train_logistic.py",
    "train_extratrees.py",
    "train_xgboost.py",
    "train_linear_regression.py"
]

# ===============================
# 工具函数
# ===============================
def run_script(name):
    print(f"\n=== 🚀 正在运行模型: {name} ===")
    res = subprocess.run([sys.executable, os.path.join(BASE_DIR, name)],
                         capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("⚠️ Error:", res.stderr)
    return res.stdout


def parse_metrics(txt):
    """解析AUC/AP/KS"""
    m = re.search(r"AUC=([\d\.]+).*AP=([\d\.]+).*KS=([\d\.]+)", txt)
    if m:
        return {"AUC": float(m.group(1)), "AP": float(m.group(2)), "KS": float(m.group(3))}
    return {"AUC": np.nan, "AP": np.nan, "KS": np.nan}


# ===============================
# 运行四个模型
# ===============================
results = {}
for s in scripts:
    out = run_script(s)
    results[s.replace(".py", "")] = parse_metrics(out)

# ===============================
# 选出表现最好的两个模型
# ===============================
best = sorted(results.items(), key=lambda kv: kv[1]["AUC"], reverse=True)[:2]
weights = {best[0][0]: 0.6, best[1][0]: 0.4}
print(f"\n🧩 融合模型：{weights}")

dfs = {}
for name, _ in best:
    file_name = f"submission_{name.split('_')[-1]}.csv"
    path = os.path.join(OUT_DIR, file_name)
    if os.path.exists(path):
        dfs[name] = pd.read_csv(path)

ids = dfs[list(dfs.keys())[0]]["id"]
prob = sum(dfs[k]["target"] * w for k, w in weights.items())
pd.DataFrame({"id": ids, "target": prob}).to_csv(os.path.join(OUT_DIR, "submission_ensemble.csv"), index=False)
print(f"✅ 已保存融合结果: submission_ensemble.csv")

# ===============================
# 计算融合模型指标
# ===============================
train = pd.read_excel(os.path.join(DATA_DIR, "训练数据集.xlsx"))
y_true = train["target"].astype(int)
auc = roc_auc_score(y_true[:len(prob)], prob[:len(y_true)])
ap = average_precision_score(y_true[:len(prob)], prob[:len(y_true)])
fpr, tpr, _ = roc_curve(y_true[:len(prob)], prob[:len(y_true)])
ks = np.max(np.abs(tpr - fpr))
print(f"\n✅ Ensemble: AUC={auc:.4f}  AP={ap:.4f}  KS={ks:.4f}")

# ===============================
# 汇总特征重要性
# ===============================
imp_files = [f for f in os.listdir(OUT_DIR) if f.startswith("feature_importance_")]
if imp_files:
    dfs_imp = []
    for f in imp_files:
        df = pd.read_csv(os.path.join(OUT_DIR, f))
        model = f.replace("feature_importance_", "").replace(".csv", "")
        df["model"] = model
        dfs_imp.append(df)
    imp_all = pd.concat(dfs_imp)
    mean_imp = imp_all.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()
    mean_imp.to_csv(os.path.join(OUT_DIR, "feature_importance_overall.csv"), index=False)
    print("\n📊 前10个平均特征重要性：")
    print(mean_imp.head(10).to_string(index=False))

    # 绘制前10特征条形图
    top10 = mean_imp.head(10)
    plt.figure(figsize=(8, 5))
    plt.barh(top10["feature"][::-1], top10["importance"][::-1], color="steelblue")
    plt.xlabel("平均重要性")
    plt.ylabel("特征名")
    plt.title("特征重要性 Top 10（全模型平均）")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance_top10.png"), dpi=200)
    plt.close()
    print("📈 已生成图像: feature_importance_top10.png")

# ===============================
# 汇总模型性能表
# ===============================
df = pd.DataFrame(results).T
df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
print("\n✅ 模型性能对比表已保存: model_comparison.csv")
print(df.round(4))
