# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
eda_income_log_compare.py - annualIncome 分布：原始刻度 vs log10 变换对比
项目：信贷违约预测模型（W1 交付物补充）
输入：data/train_clean.csv
输出：outputs/W1/eda_income_log_compare.png
说明：原始刻度下右尾不可见；log10 变换后近似钟形（对数正态）。
      仅展示用，不改动数据；W2 特征工程再做正式变换。
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd

# 向上查找含 data/ 的目录作为项目根
BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
OUT = BASE_DIR / "outputs" / "W1"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASE_DIR / "data" / "train_clean.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
s = df["annualIncome"]

# 左：原始刻度
axes[0].hist(s, bins=50, color="#4C72B0", edgecolor="white")
axes[0].set_title("annualIncome (raw scale)")
axes[0].set_xlabel("annualIncome")

# 右：log10 变换后的分布（仅展示，横轴为 log10 值）
log_inc = np.log10(s)
axes[1].hist(log_inc, bins=50, color="#4C72B0", edgecolor="white")
axes[1].set_title("annualIncome (log10 transformed)")
axes[1].set_xlabel("log10(annualIncome)")

fig.suptitle("annualIncome distribution: raw vs log10 view")
fig.tight_layout()
fig.savefig(OUT / "eda_income_log_compare.png", dpi=120)
plt.close(fig)
print("已保存:", OUT / "eda_income_log_compare.png")
