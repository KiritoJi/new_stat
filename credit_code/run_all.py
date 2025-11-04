# -*- coding: utf-8 -*-
import os, sys, subprocess, pandas as pd, re, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# 强制标准输出/错误使用 UTF-8，避免终端编码不一致
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ===============================
# 路径
# ===============================
BASE_DIR = "/Users/krt/krt files/Github/new_stat/credit_code"
ROOT_DIR = os.path.dirname(BASE_DIR)   # /Users/krt/krt files/Github/new_stat
OUT = os.path.join(ROOT_DIR, "outputs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(OUT, exist_ok=True)

# ===============================
# 模型脚本
# ===============================
scripts = [
    "train_logistic.py",
    "train_extratrees.py"
]

# ===============================
# 运行&解析
# ===============================
def run_script(script):
    print(f"\n🚀 正在运行模型脚本：{script}")
    result = subprocess.run([sys.executable, os.path.join(BASE_DIR, script)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stdout)
    if result.stderr.strip():
        print("⚠️ 错误信息：\n", result.stderr)
    return result.stdout

def parse_metrics(log_text):
    metrics = {"AUC": np.nan, "AP": np.nan, "KS": np.nan}
    m = re.search(r"AUC=([\d\.]+).*AP=([\d\.]+).*KS=([\d\.]+)", log_text)
    if m:
        metrics = {"AUC": float(m.group(1)), "AP": float(m.group(2)), "KS": float(m.group(3))}
    return metrics

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 训练各模型
# ===============================
results = {}
for s in scripts:
    out_text = run_script(s)
    results[s.replace(".py","")] = parse_metrics(out_text)

# ===============================
# 加载预测文件
# ===============================
files = {
    "logistic": os.path.join(OUT, "submission_logistic_regression.csv"),
    "extratrees": os.path.join(OUT, "submission_extra_trees.csv")
}
dfs = {k: pd.read_csv(p) for k,p in files.items() if os.path.exists(p)}
if not dfs:
    raise RuntimeError("❌ 没有找到任何模型输出，无法融合。")

# 固定使用 Logistic 与 ExtraTrees
df_compare = pd.DataFrame(results).T
required_models = ["logistic", "extratrees"]
missing_models = [m for m in required_models if m not in dfs]
if missing_models:
    raise RuntimeError(f"❌ 以下模型预测文件缺失，无法融合：{missing_models}")
best_keys = required_models
print(f"\n🏆 参与融合的模型：{best_keys}")

# ID 对齐
main_key = best_keys[0]
ids = dfs[main_key]["id"]
for k in best_keys:
    dfs[k] = dfs[k].set_index("id").reindex(ids).reset_index()

# 权重（固定 0.9 / 0.1）
weights = {"logistic": 0.9, "extratrees": 0.1}
print("\n📦 融合权重（固定）：", weights)

# 生成融合结果
ensemble_prob = np.zeros(len(ids), dtype=float)
for k in best_keys:
    ensemble_prob += dfs[k]["target"].values * weights[k]
ensemble_prob = np.where(ensemble_prob < 0.1, ensemble_prob / 10.0, ensemble_prob)
sub_ens = pd.DataFrame({"id": ids, "target": ensemble_prob})
ens_path = os.path.join(OUT, "submission_ensemble.csv")
sub_ens.to_csv(ens_path, index=False, encoding="utf-8")
print(f"✅ 已生成融合预测文件：{ens_path}")

# 计算融合指标（在训练集上）
train_path = os.path.join(DATA_DIR, "训练数据集.xlsx")
if os.path.exists(train_path):
    train = pd.read_excel(train_path, sheet_name=0)
    if "target" in train.columns:
        y_true = train["target"].astype(int).values
        y_pred = ensemble_prob[:len(y_true)]
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks = np.max(np.abs(tpr - fpr))
        print(f"📊 融合（Logistic+ExtraTrees）评估：AUC={auc:.4f}  AP={ap:.4f}  KS={ks:.4f}")

        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}, KS={ks:.4f}")
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.xlabel("假阳性率 (FPR)"); plt.ylabel("真阳性率 (TPR)")
        plt.title("融合模型 ROC 曲线（Logistic+ExtraTrees）"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(OUT,"ensemble_roc_curve.png"), dpi=220); plt.close()

        results["ensemble_logistic_extratrees"] = {"AUC": auc, "AP": ap, "KS": ks}

# 汇总表 + 中文图
cmp_path = os.path.join(OUT, "model_comparison.csv")
pd.DataFrame(results).T.to_csv(cmp_path, index_label="模型名称", encoding="utf-8")
print(f"\n✅ 模型性能对比表已保存：{cmp_path}")

dfc = pd.read_csv(cmp_path, index_col=0, encoding="utf-8")
if not dfc.empty:
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    metrics = ["AUC","AP","KS"]; colors=["#409EFF","#67C23A","#E6A23C"]
    for i,m in enumerate(metrics):
        ax = axes[i]
        if m in dfc.columns:
            dfc[m].plot(kind="bar", ax=ax, color=colors[i])
            ax.set_title(f"{m} 指标对比"); ax.set_xlabel("模型"); ax.set_ylabel(m)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            for idx, val in enumerate(dfc[m].values):
                if pd.notna(val): ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("各模型性能对比图（含 Logistic+ExtraTrees 融合）")
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(os.path.join(OUT,"model_comparison_plot.png"), dpi=220); plt.close()
    print(f"📈 模型性能对比图已保存：{os.path.join(OUT,'model_comparison_plot.png')}")
else:
    print("⚠️ 无对比数据，未绘图。")

print("\n🎯 全流程完成。输出目录：", OUT)
