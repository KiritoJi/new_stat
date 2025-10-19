import os
import sys
import subprocess
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ===============================
# 路径设置
# ===============================
BASE_DIR = "/Users/krt/krt files/Github/new_stat/credit_code"
ROOT_DIR = os.path.dirname(BASE_DIR)  # /Users/krt/krt files/Github/new_stat
OUT = os.path.join(ROOT_DIR, "outputs")  # ✅ 输出文件夹与 credit_code 同级
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(OUT, exist_ok=True)

# ===============================
# 模型脚本列表
# ===============================
scripts = [
    "train_logistic.py",
    "train_extratrees.py",
    "train_xgboost.py",
    "train_linear_regression.py"
]

# ===============================
# 执行模型脚本
# ===============================
def run_script(script):
    print(f"\n🚀 正在运行模型脚本：{script}")
    result = subprocess.run([sys.executable, os.path.join(BASE_DIR, script)],
                            capture_output=True, text=True)
    print(result.stdout)
    if result.stderr.strip():
        print("⚠️ 错误信息：\n", result.stderr)
    return result.stdout

# ===============================
# 从日志中提取指标
# ===============================
def parse_metrics(log_text):
    metrics = {"AUC": np.nan, "AP": np.nan, "KS": np.nan}
    match = re.search(r"AUC=([\d\.]+).*AP=([\d\.]+).*KS=([\d\.]+)", log_text)
    if match:
        metrics = {
            "AUC": float(match.group(1)),
            "AP": float(match.group(2)),
            "KS": float(match.group(3))
        }
    return metrics

# ===============================
# 图表显示中文
# ===============================
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 运行各模型并收集指标
# ===============================
results = {}
for script in scripts:
    log_output = run_script(script)
    metrics = parse_metrics(log_output)
    results[script.replace(".py", "")] = metrics

# ===============================
# 读取预测文件
# ===============================
print("\n⚙️ 正在加载模型预测文件...")
files = {
    "logistic": os.path.join(OUT, "submission_logistic_regression.csv"),
    "extratrees": os.path.join(OUT, "submission_extra_trees.csv"),
    "xgboost": os.path.join(OUT, "submission_xgboost.csv"),
    "linear": os.path.join(OUT, "submission_linear_regression.csv")
}

dfs = {}
for name, path in files.items():
    if os.path.exists(path):
        try:
            dfs[name] = pd.read_csv(path)
            print(f"✅ 已加载：{name} ({dfs[name].shape[0]} 行)")
        except Exception as e:
            print(f"⚠️ 读取 {name} 文件失败：{e}")
    else:
        print(f"⚠️ 未找到 {name} 文件：{path}")

if len(dfs) == 0:
    raise RuntimeError("❌ 没有找到任何模型预测结果，无法执行融合！")

# ===============================
# 自动选择 AUC 最高的两个模型
# ===============================
df_compare = pd.DataFrame(results).T
best_models = df_compare["AUC"].sort_values(ascending=False).head(2).index.tolist()
best_names = [b.replace("train_", "") for b in best_models]
print(f"\n🏆 自动选择表现最好的两个模型用于融合：{best_names}")

# 提取对应的预测结果
available_models = {k:v for k,v in dfs.items() if any(k in b for b in best_names)}
if len(available_models) < 2:
    print("⚠️ 找不到足够的模型用于融合，默认融合全部可用模型。")
    available_models = dfs

# ===============================
# 自动确定 ID 对齐模板
# ===============================
main_key = list(available_models.keys())[0]
ids = available_models[main_key]["id"]
for k in available_models:
    available_models[k] = available_models[k].set_index("id").reindex(ids).reset_index()

# ===============================
# 动态计算权重（按AUC占比）
# ===============================
auc_dict = {k: df_compare.loc[f"train_{k}", "AUC"] if f"train_{k}" in df_compare.index else 0 for k in available_models}
sum_auc = sum(auc_dict.values())
weights = {k: (v / sum_auc if sum_auc > 0 else 1/len(available_models)) for k, v in auc_dict.items()}

print("\n📦 模型融合权重（按AUC自适应）：")
for k, v in weights.items():
    print(f"  {k:<12} → {v:.2f}")

# ===============================
# 计算融合概率
# ===============================
ensemble_prob = np.zeros(len(ids), dtype=float)
for name, df in available_models.items():
    ensemble_prob += df["target"].values * weights[name]

sub_ensemble = pd.DataFrame({"id": ids, "target": ensemble_prob})
ensemble_path = os.path.join(OUT, "submission_ensemble.csv")
sub_ensemble.to_csv(ensemble_path, index=False)
print(f"\n✅ 已生成融合预测文件：{ensemble_path}")

# ===============================
# 融合模型评估
# ===============================
train_path = os.path.join(DATA_DIR, "训练数据集.xlsx")
if os.path.exists(train_path):
    train = pd.read_excel(train_path, sheet_name=0)
    if "target" in train.columns:
        y_true = train["target"].astype(int).values
        ensemble_valid = ensemble_prob[:len(y_true)]
        auc = roc_auc_score(y_true, ensemble_valid)
        ap = average_precision_score(y_true, ensemble_valid)
        fpr, tpr, _ = roc_curve(y_true, ensemble_valid)
        ks = np.max(np.abs(tpr - fpr))

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}, KS={ks:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("假阳性率 (FPR)")
        plt.ylabel("真阳性率 (TPR)")
        plt.title("融合模型 ROC 曲线（Top-2）")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "ensemble_roc_curve.png"), dpi=200)
        plt.close()

        results["ensemble_submission_top2"] = {"AUC": auc, "AP": ap, "KS": ks}
        print(f"📊 融合模型（Top-2）表现：AUC={auc:.4f}, AP={ap:.4f}, KS={ks:.4f}")
    else:
        print("⚠️ 训练数据缺少 target 列，无法计算指标。")
else:
    print("⚠️ 未找到训练数据集文件，跳过评估。")

# ===============================
# 汇总指标与绘图
# ===============================
df_compare = pd.DataFrame(results).T
df_compare.to_csv(os.path.join(OUT, "model_comparison.csv"), index_label="模型名称")
print("\n✅ 模型性能对比表已保存：model_comparison.csv")
print(df_compare.round(4))

# ===============================
# 绘制模型指标对比图
# ===============================
if not df_compare.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["AUC", "AP", "KS"]
    colors = ["#409EFF", "#67C23A", "#E6A23C"]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        df_compare[metric].plot(kind="bar", ax=ax, color=colors[i])
        ax.set_title(f"{metric} 指标对比", fontsize=13)
        ax.set_xlabel("模型")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        for idx, val in enumerate(df_compare[metric]):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("各模型性能对比图（含 Top-2 融合）", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(OUT, "model_comparison_plot.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"📈 模型性能对比图已保存：{plot_path}")
else:
    print("⚠️ 无模型指标数据，无法绘制对比图。")

print("\n🎯 全部流程执行完毕！输出文件均位于：", OUT)
