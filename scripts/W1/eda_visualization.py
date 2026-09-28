# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
eda_visualization.py - 探索性数据分析与可视化
项目：信贷违约预测模型（W1 交付物）
输入：data/train_clean.csv（清洗后数据）
输出：outputs/W1/eda_*.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
import seaborn as sns

# 向上查找含 data/ 的目录作为项目根
BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W1"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "train_clean.csv")
print(f"数据: {len(df):,} 行 x {df.shape[1]} 列")
print(f"违约率: {df['isDefault'].mean()*100:.2f}%")

KEY_NUM = ["loanAmnt", "interestRate", "installment", "annualIncome",
           "dti", "revolUtil", "ficoRangeLow"]

# 1. 目标变量分布
fig, ax = plt.subplots(figsize=(5, 4))
df["isDefault"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_xticklabels(["0 = Normal", "1 = Default"], rotation=0)
ax.set_title("Target: isDefault")
for i, v in enumerate(df["isDefault"].value_counts().sort_index()):
    ax.text(i, v + 3000, f"{v:,}", ha="center")
fig.tight_layout()
fig.savefig(OUT / "eda_target_distribution.png", dpi=120)
plt.close(fig)
print("1/5 目标分布  ->", OUT / "eda_target_distribution.png")

# 2. 关键连续变量直方图
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, KEY_NUM[:6]):
    df[col].hist(bins=50, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title(col)
fig.suptitle("Numeric distributions (train_clean)")
fig.tight_layout()
fig.savefig(OUT / "eda_numeric_distributions.png", dpi=120)
plt.close(fig)
print("2/5 数值分布  ->", OUT / "eda_numeric_distributions.png")

# 3. 违约 vs 不违约 箱线图对比
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, KEY_NUM[:6]):
    sns.boxplot(data=df, x="isDefault", y=col, ax=ax,
                hue="isDefault", palette=["#4C72B0", "#C44E52"],
                legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 Normal", "1 Default"])
    ax.set_title(col)
    if col == "annualIncome":
        ax.set_yscale("log")
fig.suptitle("Default vs Non-default comparison")
fig.tight_layout()
fig.savefig(OUT / "eda_compare_default.png", dpi=120)
plt.close(fig)
print("3/5 违约对比  ->", OUT / "eda_compare_default.png")

# 4. 信用等级 grade 的违约率
g = df.groupby("grade")["isDefault"].mean().sort_index() * 100
fig, ax = plt.subplots(figsize=(7, 4))
g.plot(kind="bar", ax=ax, color="#C44E52")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by grade")
for i, v in enumerate(g):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
fig.tight_layout()
fig.savefig(OUT / "eda_grade_default_rate.png", dpi=120)
plt.close(fig)
print("4/5 grade违约率 ->", OUT / "eda_grade_default_rate.png")

# 5. 相关性热力图（关键数值列）
corr = df[KEY_NUM + ["isDefault"]].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation heatmap (key numeric features)")
fig.tight_layout()
fig.savefig(OUT / "eda_corr_heatmap.png", dpi=120)
plt.close(fig)
print("5/5 相关性热力图 ->", OUT / "eda_corr_heatmap.png")

print("\n完成！5 张图已保存到 outputs/W1/")
