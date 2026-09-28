# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
correlation_analysis.py - 特征相关性分析（W2 补充交付物，2026/9/14）
项目：信贷违约预测模型
输入：data/train_final.csv（38 个最终建模特征 + isDefault）
输出：outputs/W2/correlation_heatmap.png（全量 38×38）、correlation_top15.png（Top 15 放大）

说明：按带教老师反馈补充——① 特征与目标变量的相关性排序；
      ② 特征之间的相关性/共线性结构（热力图）。
      相关系数基于最终建模特征（金额列已 log 变换、连续列已标准化），
      度量的是线性相关；0/1 独热列与目标的相关 = 点二列相关，量级可比。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 向上查找含 data/ 的目录作为项目根
BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W2"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "train_final.csv")
print(f"加载: train_final.csv {df.shape}")

corr = df.corr(numeric_only=True)
target_corr = corr["isDefault"].drop("isDefault")
target_rank = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)

print("\n[1] 与 isDefault 的相关系数 Top 10（按绝对值排序）:")
for k, v in target_rank.head(10).items():
    print(f"  {k:<28s} {v:+.3f}")

# 特征间高相关对（|r| >= 0.6，排除目标列）
pairs = []
feat = [c for c in corr.columns if c != "isDefault"]
for i in range(len(feat)):
    for j in range(i + 1, len(feat)):
        r = corr.iloc[corr.columns.get_loc(feat[i]), corr.columns.get_loc(feat[j])]
        if abs(r) >= 0.6:
            pairs.append((feat[i], feat[j], r))
pairs.sort(key=lambda x: -abs(x[2]))
print(f"\n[2] 特征间 |r| >= 0.6 的高相关对（共 {len(pairs)} 对），前 12:")
for a, b, r in pairs[:12]:
    print(f"  {a:<26s} ~ {b:<26s} {r:+.3f}")

# ---------- 图 1/2：全量 38×38 相关矩阵（不标数字，看整体结构，报告"图 2"用） ----------
cmap = "RdBu_r"

fig, ax = plt.subplots(figsize=(7.0, 6.5))
im0 = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns, fontsize=7)
ax.set_title("(a) All 38 features + isDefault (39 x 39)\n(train_final.csv, n = 798,498, no annotations)",
             fontsize=9)
fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.02, label="Pearson correlation")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=300)
print(f"\n已保存: {OUT / 'correlation_heatmap.png'}")
plt.close(fig)

# ---------- 图 2/2：与目标相关性 Top 15 单幅热力图（带数值标注，报告"图 3"用） ----------
top15 = list(target_rank.head(15).index) + ["isDefault"]   # Top 15 特征 + 目标列

fig2, ax2 = plt.subplots(figsize=(8.6, 7.6))
sub2 = corr.loc[top15, top15]
im2 = ax2.imshow(sub2.values, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
ax2.set_xticks(range(len(top15)))
ax2.set_xticklabels(top15, rotation=45, ha="right", fontsize=10)
ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels(top15, fontsize=10)
for i in range(len(top15)):
    for j in range(len(top15)):
        v = sub2.values[i, j]
        ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if abs(v) > 0.55 else "black")
ax2.set_title("Correlation: Top 15 features by |corr| with isDefault\n(train_final.csv, n = 798,498)",
              fontsize=12)
fig2.colorbar(im2, ax=ax2, fraction=0.045, pad=0.02, label="Pearson correlation")
fig2.tight_layout()
fig2.savefig(OUT / "correlation_top15.png", dpi=170)
print(f"已保存: {OUT / 'correlation_top15.png'}")

