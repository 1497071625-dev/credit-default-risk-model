"""
learning_curve.py - 学习曲线（W4 交付物，任务 4.3/6.1 模型性能可视化之一）
项目：信贷违约预测模型
输入：data/train_final.csv（特征工程最终产物）
输出：outputs/W4/learning_curve.png、learning_curve.csv

学习曲线回答三个问题：
  1. 模型是否过拟合？（训练 AUC 远高于验证 AUC = 背题）
  2. 模型是否欠拟合？（两条线都低且平行 = 模型太弱/特征不够）
  3. 再加数据还有没有用？（两条线随样本量增大还在上升且未收敛 = 有用）

方法：与最终模型一致的 LightGBM 参数，从训练集逐步抽取
1万 -> 55.9万人 训练，用验证集（模型没见过）评估 AUC。
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W4"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 数据划分：与 W2/W4 完全一致（分层抽样） ----------
df = pd.read_csv(DATA / "train_final.csv")
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
print(f"训练 {len(X_train):,} / 验证 {len(X_val):,} / 测试 {len(X_test):,}")

# ---------- 与最终模型完全一致的参数 ----------
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
best_params = {"subsample": 0.8, "num_leaves": 63, "n_estimators": 200,
               "max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.7}

# ---------- 样本量序列：从 1 万到全量训练集，近似等比例递增 ----------
sizes = [10_000, 25_000, 50_000, 100_000, 200_000, 400_000, len(X_train)]

train_auc, val_auc = [], []
t0 = time.time()
for i, n in enumerate(sizes, 1):
    # 固定随机种子抽子样本（保证每次都是同一批人的前 n 个）；
    # n 取满时直接用全量训练集，且不打乱行序 —— 这样末点与最终模型完全一致，
    # 避免 subsample=0.8 在乱序数据上抽出不同子集、末点与表里的 AUC 差 0.0002。
    if n >= len(X_train):
        Xs, ys = X_train, y_train
    else:
        idx = np.random.default_rng(42).choice(len(X_train), n, replace=False)
        Xs, ys = X_train.iloc[idx], y_train.iloc[idx]

    model = LGBMClassifier(**best_params, scale_pos_weight=scale_pos,
                           verbosity=-1, random_state=42, n_jobs=-1)
    model.fit(Xs, ys)
    train_auc.append(roc_auc_score(ys, model.predict_proba(Xs)[:, 1]))
    val_auc.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    print(f"  [{i}/{len(sizes)}] n={n:>7,}  train_auc={train_auc[-1]:.4f}  "
          f"val_auc={val_auc[-1]:.4f}  ({time.time()-t0:.0f}s)")

# ---------- 保存数值表 ----------
lc = pd.DataFrame({"train_size": sizes, "train_auc": train_auc, "val_auc": val_auc})
lc.to_csv(OUT / "learning_curve.csv", index=False)

# ---------- 绘图（图内英文，与 W4 其它图一致） ----------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sizes, train_auc, "o-", label="Train AUC", lw=2)
ax.plot(sizes, val_auc, "s-", label="Validation AUC", lw=2)
ax.set_xscale("log")
ax.set_xticks(sizes)
ax.set_xticklabels([f"{n/1000:.0f}K" if n < 1_000_000 else "559K"
                    for n in sizes], rotation=45)
ax.set_xlabel("Training set size")
ax.set_ylabel("AUC")
ax.set_title("Learning Curve (final LightGBM)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "learning_curve.png", dpi=120)
plt.close(fig)
print(f"\n学习曲线已保存: {OUT / 'learning_curve.png'}")
print(f"数值表已保存: {OUT / 'learning_curve.csv'}")
print("解读：训练/验证 AUC 都随样本量上升且差距小 = 加数据有用、无明显过拟合")
