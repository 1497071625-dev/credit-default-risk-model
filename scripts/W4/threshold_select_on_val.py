# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
threshold_select_on_val.py - 把"最优审批阈值"的选择挪到验证集上做（补 B1）

要解决的毛病
    threshold_sweep.py 里的最优阈值 t*=0.553 是在**测试集**上扫出来的。
    测试集本该只用来验证一次，一旦拿它挑参数，报出来的收益就会偏乐观。
    这个脚本改成：阈值在验证集上选，测试集只报一次结果。

做法
    1. 复现与 model_final_eval.py / probability_calibration.py 完全相同的切分
       （70/20/10，stratify，random_state=42）与同一个 LightGBM 参数；
    2. 在验证集上扫阈值：拒绝 = 原始概率 ≥ t，金额用校准后概率算（中性口径 6% / 30%）；
    3. 取验证集上期望利润最高的阈值 t_val*，再把它原样拿到测试集上跑一次；
    4. 与"测试集上直接挑出的 0.553"和"默认 0.5"对比，看结论差多少。

输入：data/train_final.csv、outputs/W4/test_predictions_calibrated.csv
输出：outputs/W4/threshold_select_on_val.csv（验证集逐阈值曲线 + 三个口径的测试集结果）
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W4"

V_NEUTRAL, REC_BASE = 0.06, 0.30
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]


def profit_wan(prob, loan, term, reject):
    """中性口径下的总期望利润（万元）：只给被批准的人算账。
    reject 为"被拒"的布尔数组。"""
    approve = ~reject
    unit = (1 - prob) * V_NEUTRAL * term - prob * (1 - REC_BASE)
    return float((unit[approve] * loan[approve]).sum() / 1e4)


def gain_wan(y, prob, loan, term, reject):
    """实现值净增益（万元）：避免的坏账 − 放弃的收入，相对全批放款。"""
    return float(((1 - REC_BASE) * loan[reject & (y == 1)].sum()
                  - V_NEUTRAL * (loan * term)[reject & (y == 0)].sum()) / 1e4)


# ---------- 1) 同一份切分、同一个模型 ----------
df = pd.read_csv(DATA / "train_final.csv")
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
# train_final 里的 loanAmnt 已经 log + 标准化，金额要从清洗后的原表按索引取
# （与 prepare_test_predictions.py 同一做法）
clean = pd.read_csv(DATA / "train_clean.csv", usecols=["loanAmnt", "term"])
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1 / 3, stratify=y_temp, random_state=42)
print(f"划分：训练 {len(X_train):,} / 验证 {len(X_val):,} / 测试 {len(X_test):,}")

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
best_params = {"subsample": 0.8, "num_leaves": 63, "n_estimators": 200,
               "max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.7}
model = LGBMClassifier(**best_params, scale_pos_weight=scale_pos,
                       verbosity=-1, random_state=42, n_jobs=-1)
t0 = time.time()
model.fit(X_train, y_train)
print(f"模型训练完成，用时 {time.time() - t0:.0f}s")

# ---------- 2) 验证集：原始概率 → 校准概率 ----------
p_val = model.predict_proba(X_val)[:, 1]
y_val_np = y_val.to_numpy()
loan_val = clean.loc[X_val.index, "loanAmnt"].to_numpy(float)
term_val = clean.loc[X_val.index, "term"].to_numpy(float)
p_val_cal = np.clip(fit_platt(p_val, y_val_np)(p_val), 0, 1)
print(f"验证集：原始概率均值 {p_val.mean():.4f} → 校准后 {p_val_cal.mean():.4f}"
      f"（实际违约率 {y_val_np.mean():.4f}）")

# ---------- 3) 验证集上扫阈值 ----------
grid = np.round(np.arange(0.10, 0.95, 0.005), 3)
rows = []
for t in grid:
    reject = p_val >= t
    rows.append({
        "阈值": float(t),
        "被拒占比": float(reject.mean()),
        "拦截率": float(y_val_np[reject].sum() / y_val_np.sum()),
        "被拒者精确率": float(y_val_np[reject].mean()) if reject.any() else np.nan,
        "总期望利润(万元)": profit_wan(p_val_cal, loan_val, term_val, reject),
        "净增益(万元)": gain_wan(y_val_np, p_val_cal, loan_val, term_val, reject),
    })
val_curve = pd.DataFrame(rows)
best_row = val_curve.loc[val_curve["总期望利润(万元)"].idxmax()]
t_val = float(best_row["阈值"])
# 平台区：与最优值相差不到 0.5% 的阈值范围
plateau = val_curve[val_curve["总期望利润(万元)"]
                    >= best_row["总期望利润(万元)"] * 0.995]
print(f"\n验证集最优阈值 t_val* = {t_val:.3f}"
      f"（拒绝 {best_row['被拒占比'] * 100:.1f}%，期望利润 {best_row['总期望利润(万元)']:,.0f} 万元）")
print(f"验证集上的平台区（利润在最优 99.5% 以内）："
      f"{plateau['阈值'].min():.3f} ~ {plateau['阈值'].max():.3f}")

# ---------- 4) 把验证集选出的阈值搬到测试集，只跑一次 ----------
test = pd.read_csv(OUT / "test_predictions_calibrated.csv")
p_raw_t = test["prob_raw"].to_numpy(float)
p_cal_t = test["prob_cal"].to_numpy(float)
loan_t = test["loanAmnt"].to_numpy(float)
term_t = test["term"].to_numpy(float)
y_t = test["y_true"].to_numpy(int)

auc_re = float(pd.Series(p_raw_t).corr(pd.Series(y_t), method="spearman"))
print(f"\n复现自检：测试集原始概率均值 {p_raw_t.mean():.4f}"
      f"｜校准后 {p_cal_t.mean():.4f}｜阈值 0.5 拒绝率 {np.mean(p_raw_t >= 0.5) * 100:.1f}%"
      "（应与 test_predictions_calibrated.csv 一致）")

# 测试集上的最优（仅供对照，不用于决策）
grid_t = np.round(np.arange(0.10, 0.95, 0.005), 3)
prof_t = np.array([profit_wan(p_cal_t, loan_t, term_t, p_raw_t >= t) for t in grid_t])
t_test_best = float(grid_t[int(np.argmax(prof_t))])

cases = [
    ("默认 0.5", 0.5),
    ("测试集最优（对照，不可用于决策）", t_test_best),
    ("验证集选出的 t_val*", t_val),
]
summary = []
for name, t in cases:
    rj = p_raw_t >= t
    summary.append({
        "口径": name, "阈值": round(float(t), 3),
        "测试集拒绝率": float(rj.mean()),
        "测试集期望利润(万元)": profit_wan(p_cal_t, loan_t, term_t, rj),
        "测试集净增益(万元)": gain_wan(y_t, p_cal_t, loan_t, term_t, rj),
    })
summary = pd.DataFrame(summary)
print("\n测试集结果（每个口径都只在这个表里出现一次）")
print(summary.round(4).to_string(index=False))

val_curve.to_csv(OUT / "threshold_select_on_val.csv", index=False)
print(f"\n已写出 {OUT / 'threshold_select_on_val.csv'}")
