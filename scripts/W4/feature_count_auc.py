# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
feature_count_auc.py - 特征数量 vs 模型性能（W4 Track B 第③项）
项目：信贷违约预测模型

为什么做这个：
  最终特征表保留了 38 个特征。业务方一定会问：38 个都要吗？少放几个行不行？
  少放特征的好处：数据采集和维护成本低、模型轻、上线快、给业务解释也更简单；
  代价：信息变少，模型可能变差。所以做一次"逐步加特征 + 重训模型"的实验，
  看验证集 AUC 加到第几个特征后涨不动了 —— 那个位置就是"够用"的特征数量。

做法（保持测试集纪律）：
  1) 特征排序用 W2 随机森林特征重要性（只反映训练信息，不看测试集）；
  2) 依次取 Top-N（N = 3,5,8,10,15,20,25,30,38），用同一个 LGB 参数重训；
  3) 只比较训练集与验证集 AUC，测试集全程不参与本实验。

输入：data/train_final.csv、outputs/W2/feature_importance.csv（特征排序）
产出：outputs/W4/feature_count_auc.csv   每个 N 的训练/验证 AUC 与训练用时
      outputs/W4/feature_count_auc.png   曲线图（英文标注，与 W1-W3 图风格一致）
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

BEST_PARAMS = {"subsample": 0.8, "num_leaves": 63, "n_estimators": 200,
               "max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.7}
N_LIST = [3, 5, 8, 10, 15, 20, 25, 30, 38]

df = pd.read_csv(DATA / "train_final.csv")
all_feats = [c for c in df.columns if c != "isDefault"]
y = df["isDefault"]
print(f"加载: train_final.csv {df.shape}，候选特征 {len(all_feats)} 个")

# ---------- 特征排序：W2 随机森林重要性（降序），没进过榜单的排到最后 ----------
imp = pd.read_csv(BASE_DIR / "outputs" / "W2" / "feature_importance.csv", index_col=0)["importance"]
ranked = [f for f in imp.sort_values(ascending=False).index if f in all_feats]
ranked += [f for f in all_feats if f not in ranked]
assert len(ranked) == len(all_feats), "特征排序与 train_final 列不匹配"
print(f"排序依据: W2 随机森林重要性；Top1 = {ranked[0]}，Top10 = {ranked[:10]}")

# ---------- 数据划分：与 W2/W4 最终评估完全一致 ----------
X = df[all_feats]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
del X_temp, y_temp, X_test, y_test, X
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"训练 {len(X_train):,} / 验证 {len(X_val):,}（测试集不参与本实验）\n")

# ---------- 逐档训练 ----------
rows = []
prev = 0
for n in N_LIST:
    feats = ranked[:n]
    added = [f for f in feats if f not in ranked[:prev]]
    model = LGBMClassifier(**BEST_PARAMS, scale_pos_weight=scale_pos,
                           verbosity=-1, random_state=42, n_jobs=-1)
    t0 = time.time()
    model.fit(X_train[feats], y_train)
    dur = time.time() - t0
    auc_tr = roc_auc_score(y_train, model.predict_proba(X_train[feats])[:, 1])
    auc_va = roc_auc_score(y_val, model.predict_proba(X_val[feats])[:, 1])
    rows.append({"n_features": n, "train_auc": round(auc_tr, 4), "val_auc": round(auc_va, 4),
                 "seconds": round(dur, 1), "added": ", ".join(added)})
    print(f"Top-{n:>2} 特征: train AUC {auc_tr:.4f} | val AUC {auc_va:.4f} | {dur:.0f}s | 新增: {', '.join(added)}")
    prev = n

res = pd.DataFrame(rows)
res.to_csv(OUT / "feature_count_auc.csv", index=False, encoding="utf-8-sig")

# ---------- 收敛点：验证 AUC 达到最大值 - 0.001 的最小特征数 ----------
best_n = int(res.loc[res["val_auc"].idxmax(), "n_features"])
best_auc = res["val_auc"].max()
ok = res[res["val_auc"] >= best_auc - 0.001]
plateau_n = int(ok["n_features"].min())
full = res[res["n_features"] == res["n_features"].max()].iloc[0]
print(f"\n验证 AUC 最高 = {best_auc:.4f}（Top-{best_n}）")
print(f"参考点: Top-{plateau_n} 起验证 AUC 与峰值差距 < 0.001")
print(f"全量 Top-{int(full['n_features'])}: val AUC {full['val_auc']:.4f}"
      f"（比 Top-{plateau_n} 高 {full['val_auc'] - float(ok['val_auc'].min()):.4f}）")

# ---------- 作图（英文标注） ----------
fig, ax = plt.subplots(figsize=(8.2, 5))
ax.plot(res["n_features"], res["train_auc"], marker="o", ms=5, lw=1.8, color="#8c8c8c",
        ls="--", label="Train AUC")
ax.plot(res["n_features"], res["val_auc"], marker="o", ms=5, lw=2.2, color="#1f5fa9",
        label="Validation AUC")
for _, r in res.iterrows():
    ax.annotate(f"{r['val_auc']:.4f}", (r["n_features"], r["val_auc"]),
                textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7.5, color="#1f5fa9")
ax.axvline(plateau_n, color="#c0392b", lw=1.2, ls=":")
ax.text(plateau_n, ax.get_ylim()[0] + 0.002, f" plateau: Top-{plateau_n}", color="#c0392b", fontsize=8.5)
ax.set_xlabel("Number of features used (Top-N by W2 random-forest importance)")
ax.set_ylabel("AUC")
ax.set_title("Validation AUC vs. Number of Features (LightGBM, tuned params)")
ax.set_xticks(res["n_features"])
ax.grid(alpha=0.3, linestyle=":")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "feature_count_auc.png", dpi=120)
print(f"\n已保存: {OUT / 'feature_count_auc.csv'}、{OUT / 'feature_count_auc.png'}")
