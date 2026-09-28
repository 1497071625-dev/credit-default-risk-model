# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
probability_calibration.py - 概率校准（W4 Track B 第①项）
项目：信贷违约预测模型
输入：data/train_final.csv（38 特征 + isDefault）
输出：outputs/W4/calibration_compare.csv            方法对比与测试集指标（校准前后）
      outputs/W4/calibration_bins.csv               可靠性曲线分箱明细（10 箱）
      outputs/W4/calibration_curve.png              概率校准图（左：可靠性曲线；右：概率分布对比）
      outputs/W4/test_predictions_calibrated.csv    测试集：原始概率 + 校准概率（供第②项期望损失使用）

为什么做校准：
  模型 AUC（排序能力）没问题，但输出概率的"绝对值"不可信——阈值 0.5 时被判"会违约"
  的客户占 41.4%，而实际违约率只有约 20%，概率整体偏高。概率不可信就无法直接算钱
  （期望损失 = 概率 × 损失）。校准就是把"语气过重的概率"压回真实水平。

方法（两种主流做法，用数据选，不拍脑袋）：
  · isotonic（等渗回归）：非参数，形状自由，样本量大时首选；
  · Platt（Sigmoid 校准）：用一个 logistic 把 log-odds 线性缩放，样本少时更稳。

数据纪律（与 W2/W3/W4 一致）：
  - 切分与 model_final_eval.py 完全相同（同一随机种子），保证是同一份测试集；
  - 方法选择在验证集内部完成（验证集对半：一半拟合、一半比 Brier 分数）；
  - 选定方法后在**全量验证集**上重新拟合，再应用到测试集做最终评估；
  - 测试集全程不参与任何拟合与选择。
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, precision_score, recall_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W4"
OUT.mkdir(parents=True, exist_ok=True)

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_isotonic(p, y):
    """等渗回归校准器：拟合 p → 实际违约率 的单调映射。"""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p, y)
    return lambda q: ir.predict(q)


def fit_platt(p, y):
    """Platt 校准器：对 log-odds 做一次 logistic 回归（a·z + b）。"""
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]


# ---------- 数据与切分（与 model_final_eval.py 完全一致） ----------
df = pd.read_csv(DATA / "train_final.csv")
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
print(f"划分: 训练 {len(X_train):,} / 验证 {len(X_val):,} / 测试 {len(X_test):,}")

# ---------- 最终模型（W3 随机搜索最优参数，与 W4 其他脚本一致） ----------
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
best_params = {"subsample": 0.8, "num_leaves": 63, "n_estimators": 200,
               "max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.7}
model = LGBMClassifier(**best_params, scale_pos_weight=scale_pos,
                       verbosity=-1, random_state=42, n_jobs=-1)
t0 = time.time()
model.fit(X_train, y_train)
print(f"最终模型训练完成，用时 {time.time()-t0:.0f}s")

p_val = model.predict_proba(X_val)[:, 1]
p_test = model.predict_proba(X_test)[:, 1]
y_val_np = y_val.to_numpy()
y_test_np = y_test.to_numpy()

print(f"\n[校准前] 验证集：概率均值 {p_val.mean():.4f} | 实际违约率 {y_val_np.mean():.4f}")
print(f"[校准前] 测试集：概率均值 {p_test.mean():.4f} | 概率>=0.5 占比 {np.mean(p_test>=0.5)*100:.1f}%"
      f" | 实际违约率 {y_test_np.mean()*100:.1f}%")

# ---------- 方法选择：验证集对半（一半拟合、一半评估），测试集不参与 ----------
pos = np.arange(len(y_val_np))
idx_a, idx_b = train_test_split(pos, test_size=0.5, stratify=y_val_np, random_state=42)
p_a, y_a = p_val[idx_a], y_val_np[idx_a]
p_b, y_b = p_val[idx_b], y_val_np[idx_b]

brier_iso_hold = brier_score_loss(y_b, np.clip(fit_isotonic(p_a, y_a)(p_b), 0, 1))
brier_platt_hold = brier_score_loss(y_b, np.clip(fit_platt(p_a, y_a)(p_b), 0, 1))
brier_raw_hold = brier_score_loss(y_b, p_b)
chosen = "isotonic" if brier_iso_hold <= brier_platt_hold else "platt"
print("\n[方法选择·验证集内部对半] Brier 分数（越低越好）")
print(f"  未校准 {brier_raw_hold:.5f} | isotonic {brier_iso_hold:.5f} | Platt {brier_platt_hold:.5f}"
      f"  -> 采用：{chosen}")

# ---------- 在全量验证集上重新拟合选定方法，应用到测试集 ----------
cal_fn = (fit_isotonic if chosen == "isotonic" else fit_platt)(p_val, y_val_np)
p_cal = np.clip(cal_fn(p_test), 0, 1)
# 两种方法的测试集表现都算出来，写进报告作为对照
p_cal_iso = np.clip(fit_isotonic(p_val, y_val_np)(p_test), 0, 1)
p_cal_platt = np.clip(fit_platt(p_val, y_val_np)(p_test), 0, 1)


def at_threshold(p, y, t=0.5):
    pred = (p >= t).astype(int)
    return {
        "reject_rate": float(pred.mean()),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
    }


rows = [
    {"stage": "raw(test)", "brier": brier_score_loss(y_test_np, p_test),
     "auc": roc_auc_score(y_test_np, p_test), "mean_pred": float(p_test.mean()),
     "share_ge_0.5": float(np.mean(p_test >= 0.5)), **at_threshold(p_test, y_test_np)},
    {"stage": "isotonic(test)", "brier": brier_score_loss(y_test_np, p_cal_iso),
     "auc": roc_auc_score(y_test_np, p_cal_iso), "mean_pred": float(p_cal_iso.mean()),
     "share_ge_0.5": float(np.mean(p_cal_iso >= 0.5)), **at_threshold(p_cal_iso, y_test_np)},
    {"stage": "platt(test)", "brier": brier_score_loss(y_test_np, p_cal_platt),
     "auc": roc_auc_score(y_test_np, p_cal_platt), "mean_pred": float(p_cal_platt.mean()),
     "share_ge_0.5": float(np.mean(p_cal_platt >= 0.5)), **at_threshold(p_cal_platt, y_test_np)},
]
compare = pd.DataFrame(rows)
compare.insert(1, "note", ["未校准", f"采用（验证集内部选出）" if chosen == "isotonic" else "对照",
                           f"采用（验证集内部选出）" if chosen == "platt" else "对照"])
compare.loc[len(compare)] = {"stage": "actual(test)", "note": "实际违约率",
                             "brier": np.nan, "auc": np.nan, "mean_pred": float(y_test_np.mean()),
                             "share_ge_0.5": np.nan, "reject_rate": np.nan,
                             "precision": np.nan, "recall": np.nan}
compare.to_csv(OUT / "calibration_compare.csv", index=False)
print("\n[测试集指标]")
print(compare.round(4).to_string(index=False))

# ---------- 可靠性曲线分箱（按原始概率十分位分箱） ----------
q = pd.qcut(p_test, 10, labels=False, duplicates="drop")
bins = pd.DataFrame({"bin": q, "p_raw": p_test, "p_cal": p_cal, "y": y_test_np}) \
    .groupby("bin").agg(count=("y", "size"), mean_pred_raw=("p_raw", "mean"),
                        mean_pred_cal=("p_cal", "mean"), actual_rate=("y", "mean")).reset_index()
bins.to_csv(OUT / "calibration_bins.csv", index=False)
print("\n[可靠性曲线分箱]（10 箱：预测 vs 实际）")
print(bins.round(4).to_string(index=False))

# ---------- 图：左可靠性曲线，右概率分布前后对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
ax = axes[0]
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
ax.plot(bins["mean_pred_raw"], bins["actual_rate"], "o-", color="#8899aa", label="Uncalibrated")
ax.plot(bins["mean_pred_cal"], bins["actual_rate"], "o-", color="#1f6fb2",
        label=f"Calibrated ({chosen})")
ax.set_xlabel("Mean predicted probability (by decile)")
ax.set_ylabel("Actual default rate")
ax.set_title("Reliability curve (test set, 79,850 loans)")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

ax = axes[1]
ax.hist(p_test, bins=40, alpha=0.55, color="#8899aa", label="Uncalibrated")
ax.hist(p_cal, bins=40, alpha=0.65, color="#1f6fb2", label=f"Calibrated ({chosen})")
ax.axvline(0.5, color="#cc4444", ls="--", lw=1)
ax.text(0.505, ax.get_ylim()[1]*0.93, "0.5", color="#cc4444", fontsize=9)
ax.set_xlabel("Predicted default probability")
ax.set_ylabel("Number of loans")
ax.set_title("Predicted probability distribution (test set)")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(OUT / "calibration_curve.png", dpi=150)
plt.close()
print(f"\n图已保存: {OUT / 'calibration_curve.png'}")

# ---------- 保存校准后的测试集概率（供第②项期望损失框架使用） ----------
# 金额/期限/利率取原始尺度（train_clean，与 train_final 行序一致），期望损失框架可直接算金额账
clean_info = pd.read_csv(DATA / "train_clean.csv", usecols=["loanAmnt", "term", "interestRate"])
pred = pd.DataFrame({
    "y_true": y_test_np,
    "prob_raw": p_test,
    "prob_cal": p_cal,
    "loanAmnt": clean_info.loc[X_test.index, "loanAmnt"].values,
    "term": clean_info.loc[X_test.index, "term"].values,
    "interestRate": clean_info.loc[X_test.index, "interestRate"].values,
})
pred.to_csv(OUT / "test_predictions_calibrated.csv", index=False)
print(f"已保存: {OUT / 'test_predictions_calibrated.csv'}（{len(pred):,} 行）")

print("\n" + "=" * 60)
print("一句话结论（写报告用）")
print("=" * 60)
print(f"未校准：概率均值 {p_test.mean():.3f}，均值比实际违约率（{y_test_np.mean():.3f}）高 "
      f"{(p_test.mean()/y_test_np.mean()-1)*100:.0f}%；校准后：概率均值 {p_cal.mean():.3f}，"
      f"Brier {brier_score_loss(y_test_np, p_test):.4f} -> {brier_score_loss(y_test_np, p_cal):.4f}，"
      f"AUC {roc_auc_score(y_test_np, p_test):.4f} -> {roc_auc_score(y_test_np, p_cal):.4f}（排序能力不变）。")
