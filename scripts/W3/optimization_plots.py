"""
optimization_plots.py - W3 交付配图（阶段 5 报告/周报用）
项目：信贷违约预测模型
输入：outputs/W3/random_search_results.csv、outputs/W3/model_comparison_optimized.csv
输出：outputs/W3/random_search_auc.png、outputs/W3/model_comparison_optimized.png

说明：只做"把已有调优/评估结果画成图"，不重新训练模型，几秒跑完。
      图内文字用英文，与 W1/W2 交付图风格一致。
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
OUT = BASE_DIR / "outputs" / "W3"

# ---------- 图 1：随机搜索 16 组参数的 5 折平均 AUC ----------
rs = pd.read_csv(OUT / "random_search_results.csv")
rs = rs.sort_values(["model", "mean_cv_auc"], ascending=[True, False]).reset_index(drop=True)
rs["label"] = rs.groupby("model").cumcount().map(lambda i: f"#{i+1}")

colors = rs["model"].map({"XGBoost": "#4C72B0", "LightGBM": "#55A868"})
fig, ax = plt.subplots(figsize=(11, 4.8), dpi=130)
bars = ax.bar(range(len(rs)), rs["mean_cv_auc"], color=colors, width=0.72,
              edgecolor="black", linewidth=0.5)
# 每组最优两根柱子加粗边框高亮
for i, m in enumerate(rs["model"]):
    if rs.loc[i, "mean_cv_auc"] == rs.groupby("model")["mean_cv_auc"].transform("max")[i]:
        bars[i].set_edgecolor("#C44E52"); bars[i].set_linewidth(1.8)

for i, v in enumerate(rs["mean_cv_auc"]):
    ax.text(i, v + 0.0004, f"{v:.4f}", ha="center", fontsize=7.5)

# 用两根柱子的 std 画误差棒
ax.errorbar(range(len(rs)), rs["mean_cv_auc"], yerr=rs["std_cv_auc"],
            fmt="none", ecolor="black", elinewidth=0.7, capsize=2)

ax.set_xticks(range(len(rs)))
ax.set_xticklabels(rs["label"], fontsize=8)
ax.set_ylim(0.712, 0.7275)
ax.set_xlabel("Parameter combination (ranked within each model, #1 = best)", fontsize=9)
ax.set_ylabel("Mean 5-fold CV AUC", fontsize=9)
ax.set_title("Random Search: 5-fold CV AUC of 16 Parameter Combinations "
             "(blue = XGBoost, green = LightGBM, red border = best per model)",
             fontsize=10)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(OUT / "random_search_auc.png", bbox_inches="tight")
plt.close()
print("已保存:", OUT / "random_search_auc.png")

# ---------- 图 2：优化前后 + 集成的验证集对比（AUC / F1） ----------
comp = pd.read_csv(OUT / "model_comparison_optimized.csv")


def grp(name):
    if "基线" in name:
        return "Baseline (W2)"
    if "集成" in name:
        return "Ensemble"
    return "Tuned"


palette = {"Baseline (W2)": "#999999", "Tuned": "#4C72B0", "Ensemble": "#DD8452"}
comp["group"] = comp["model"].map(grp)
comp = comp.sort_values("auc").reset_index(drop=True)
# 图内用英文标签，避免中文字形缺失（与 W1/W2 交付图风格一致）
name_en = {
    "LR(基线)": "LR (baseline)", "RF(基线)": "RF (baseline)",
    "XGB(基线)": "XGB (baseline)", "XGB(优化)": "XGB (tuned)",
    "LGB(优化)": "LGB (tuned)", "Voting(集成)": "Voting (ensemble)",
    "Stacking(集成)": "Stacking (ensemble)",
}
comp["label_en"] = comp["model"].map(name_en)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130)
for ax, col, title in [(axes[0], "auc", "AUC (ranking ability, higher better)"),
                       (axes[1], "f1", "F1 score at threshold 0.5")]:
    cols = comp["group"].map(palette)
    b = ax.barh(comp["label_en"], comp[col], color=cols, height=0.62,
                edgecolor="black", linewidth=0.5)
    for i, v in enumerate(comp[col]):
        ax.text(v + (0.0006 if col == "auc" else 0.002), i, f"{v:.4f}",
                va="center", fontsize=7.5)
    ax.set_title(title, fontsize=9.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(0, comp[col].max() * 1.18)
    if col == "auc":
        base = comp.loc[comp["model"] == "XGB(基线)", "auc"].values[0]
        ax.axvline(base, color="#C44E52", linestyle="--", linewidth=1)
        ax.text(base, len(comp) - 0.35, f" XGB baseline {base:.4f}",
                color="#C44E52", fontsize=7.5, va="top")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.5)
           for c in palette.values()]
fig.legend(handles, palette.keys(), loc="lower center", ncol=3,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Validation-set performance: baselines vs tuned models vs ensembles",
             fontsize=10, y=0.99)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig(OUT / "model_comparison_optimized.png", bbox_inches="tight")
plt.close()
print("已保存:", OUT / "model_comparison_optimized.png")
