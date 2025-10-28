# -*- coding: utf-8 -*-
"""
run_all.py
-----------
执行全流程建模：
1️⃣ 顺序运行四个模型脚本
2️⃣ 自动提取AUC/AP/KS
3️⃣ 融合前2个最优模型
4️⃣ 汇总特征重要性并绘制Top10图
5️⃣ 输出中文性能总结报告
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
    """运行子模型脚本"""
    print(f"\n=== 🚀 正在运行模型: {name} ===")
    res = subprocess.run([sys.executable, os.path.join(BASE_DIR, name)],
                         capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("⚠️ 错误信息:\n", res.stderr)
    return res.stdout


def parse_metrics(txt):
    """解析 AUC/AP/KS 指标"""
    m = re.search(r"AUC=([\d\.]+).*AP=([\d\.]+).*KS=([\d\.]+)", txt)
    if m:
        return {"AUC": float(m.group(1)), "AP": float(m.group(2)), "KS": float(m.group(3))}
    return {"AUC": np.nan, "AP": np.nan, "KS": np.nan}


# ===============================
# Step 1️⃣ 运行四个模型
# ===============================
results = {}
for s in scripts:
    out = run_script(s)
    results[s.replace(".py", "")] = parse_metrics(out)

# ===============================
# Step 2️⃣ 选出表现最好的两个模型
# ===============================
best = sorted(results.items(), key=lambda kv: kv[1]["AUC"], reverse=True)[:2]
weights = {best[0][0]: 0.6, best[1][0]: 0.4}
print(f"\n🧩 融合模型（Top2）：{weights}")

# === 修正文件名与Key映射 ===
dfs = {}
for name, _ in best:
    if "logistic" in name:
        file_name, key = "submission_logistic_regression.csv", "logistic"
    elif "extratrees" in name:
        file_name, key = "submission_extra_trees.csv", "extratrees"
    elif "xgboost" in name:
        file_name, key = "submission_xgboost.csv", "xgboost"
    elif "linear" in name:
        file_name, key = "submission_linear_regression.csv", "linear"
    else:
        continue
    path = os.path.join(OUT_DIR, file_name)
    if os.path.exists(path):
        dfs[key] = pd.read_csv(path)

# === 构造权重映射 ===
mapped_weights = {}
for k, w in weights.items():
    if "logistic" in k:
        mapped_weights["logistic"] = w
    elif "extratrees" in k:
        mapped_weights["extratrees"] = w
    elif "xgboost" in k:
        mapped_weights["xgboost"] = w
    elif "linear" in k:
        mapped_weights["linear"] = w

# === 加权融合 ===
ids = dfs[list(dfs.keys())[0]]["id"]
# ✅ 使用映射后的 key，而不是原始 weights
prob = sum(dfs[k]["target"] * w for k, w in mapped_weights.items())
ensemble_path = os.path.join(OUT_DIR, "submission_ensemble.csv")
pd.DataFrame({"id": ids, "target": prob}).to_csv(ensemble_path, index=False)
print(f"✅ 已保存融合结果: {ensemble_path}")

# ===============================
# Step 3️⃣ 计算融合模型指标
# ===============================
train = pd.read_excel(os.path.join(DATA_DIR, "训练数据集.xlsx"))
y_true = train["target"].astype(int)
auc = roc_auc_score(y_true[:len(prob)], prob[:len(y_true)])
ap = average_precision_score(y_true[:len(prob)], prob[:len(y_true)])
fpr, tpr, _ = roc_curve(y_true[:len(prob)], prob[:len(y_true)])
ks = np.max(np.abs(tpr - fpr))
print(f"\n✅ Ensemble: AUC={auc:.4f}  AP={ap:.4f}  KS={ks:.4f}")

# ===============================
# Step 4️⃣ 汇总特征重要性
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

    # === 绘制Top10特征条形图 ===
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
# Step 5️⃣ 汇总性能对比表
# ===============================
df = pd.DataFrame(results).T
df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"))
print("\n✅ 模型性能对比表已保存: model_comparison.csv")
print(df.round(4))

# ===============================
# Step 6️⃣ 中文性能总结报告
# ===============================
best_model_name = best[0][0].replace("train_", "")
second_model_name = best[1][0].replace("train_", "")
best_auc = results[best[0][0]]["AUC"]
second_auc = results[best[1][0]]["AUC"]
improve = (auc - best_auc) / best_auc * 100 if best_auc > 0 else 0

print("\n===============================")
print("📋 中文性能总结报告")
print("===============================")
print(f"最优单模型：{best_model_name}  (AUC={best_auc:.4f})")
print(f"次优模型：{second_model_name}  (AUC={second_auc:.4f})")
print(f"融合模型AUC：{auc:.4f}，较最优单模型提升约 {improve:.2f}%")
print(f"平均精确率(AP)：{ap:.4f}，KS值：{ks:.4f}")
print("特征重要性文件：feature_importance_overall.csv")
print("Top10特征图：feature_importance_top10.png")
print("全部结果已输出至 outputs/ 文件夹中 ✅")
