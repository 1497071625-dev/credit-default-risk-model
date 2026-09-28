# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
tuning_method_compare.py - 调参方法对比实验（阶段 5.1 补充）
项目：信贷违约预测模型
输入：data/train_final.csv（特征工程最终产物，38 特征 + isDefault）
输出：outputs/W3/tuning_method_compare.csv    全部试验记录（逐次）
      outputs/W3/tuning_method_summary.csv    方法级汇总（预算/最优AUC/用时/最优参数）
      outputs/W3/tuning_method_compare.png    收敛曲线（1×2）

目的：任务文档 5.1 列了三种超参数搜索方式（网格 / 随机 / 贝叶斯）。
      W3 正式调参只用了随机搜索，这里把三种方法放在**完全相同的预算**下跑一遍，
      回答"换成网格或贝叶斯，能不能找到更好的参数？"，让 W3 报告的选择有数据支撑，
      而不是一句"随机搜索够用"。

  实验 1（可网格化的粗空间）：离散 24 点（4 个旋钮）= 网格搜索 24 次
      vs 随机搜索 24 次 vs 贝叶斯 TPE 24 次。等预算，看谁找到的最优高、谁收敛快。
  实验 2（网格爆炸的大空间）：6 个旋钮各取 5 档 = 15,625 组组合，
      网格搜索在算力上不可行；只比 随机搜索 30 次 vs 贝叶斯 TPE 30 次。

公平性保证：
  - 同一份子样本（分层抽 20 万行）、同一套交叉验证折（StratifiedKFold(3, seed=42)）；
  - 同一评价指标（3 折平均 AUC）；
  - 随机搜索与 TPE 共用**同一份参数空间定义**（都用 optuna sampler 采样）——
    区别只在"怎么选下一组"，不在空间大小，排除空间定义带来的偏差。

效率说明：全量 56 万行 × 5 折调一次要几十分钟，方法对比只需相对高低，
      故用 20 万行子样本 + 3 折（单次评估约 6 秒），总耗时约 15 分钟。
      全量数据上的最终调参结果仍以 model_optimization.py 的随机搜索为准。
"""
from pathlib import Path
import itertools
import sys
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, RandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 加 --from-cache 可跳过训练，直接用手上已有的试验记录重画图/重算汇总（改图改表不必再等 22 分钟）
FROM_CACHE = "--from-cache" in sys.argv

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W3"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 数据：与 W2/W3 完全一致的划分，再从训练集分层抽子样本 ----------
if FROM_CACHE:
    print("--from-cache：跳过模型训练，读取已有试验记录后重算汇总与图")

df = pd.read_csv(DATA / "train_final.csv")
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

N_SUB = 200_000
sub_idx = (y_train.groupby(y_train, group_keys=False)
           .apply(lambda s: s.sample(int(round(N_SUB * len(s) / len(y_train))),
                                     random_state=42)).index)
X_sub, y_sub = X_train.loc[sub_idx], y_train.loc[sub_idx]
print(f"调参实验数据：训练集 {len(X_train):,} 行中分层抽 {len(X_sub):,} 行"
      f"（违约率 {y_sub.mean()*100:.1f}%），3 折交叉验证")

CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
evaluate_calls = 0

def evaluate(params):
    """3 折交叉验证的平均 AUC（三种方法共用同一套折，保证可比）。"""
    global evaluate_calls
    evaluate_calls += 1
    t0 = time.time()
    model = LGBMClassifier(**params, verbosity=-1, random_state=42, n_jobs=-1)
    auc = cross_val_score(model, X_sub, y_sub, cv=CV, scoring="roc_auc", n_jobs=1).mean()
    return float(auc), time.time() - t0

# ---------- 参数空间定义（两套实验共用同一份 spec） ----------
# 实验 1：离散粗空间，4 个真正生效的旋钮（LightGBM 的 subsample 必须配 subsample_freq
#         才生效，否则是死旋钮，故不放进搜索空间）
SPACE1 = {
    "num_leaves":       {"type": "cat", "values": [31, 63, 127]},
    "learning_rate":    {"type": "cat", "values": [0.05, 0.1]},
    "n_estimators":     {"type": "cat", "values": [200, 300]},
    "colsample_bytree": {"type": "cat", "values": [0.7, 1.0]},
}
# 实验 2：大空间，6 个旋钮；若每个取 5 档 → 5^6 = 15,625 组组合
SPACE2 = {
    "num_leaves":       {"type": "int",   "low": 20,  "high": 150, "step": 10},
    "max_depth":        {"type": "int",   "low": 4,   "high": 12},
    "learning_rate":    {"type": "float", "low": 0.01, "high": 0.15, "log": True},
    "n_estimators":     {"type": "int",   "low": 100, "high": 500, "step": 50},
    "min_child_samples":{"type": "int",   "low": 5,   "high": 50},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
}

def suggest(spec, trial):
    """按 spec 从 optuna trial 里采一组参数（随机搜索与 TPE 共用）。"""
    out = {}
    for name, s in spec.items():
        if s["type"] == "cat":
            out[name] = trial.suggest_categorical(name, s["values"])
        elif s["type"] == "int":
            out[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1))
        else:
            out[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
    return out

def run_optuna(spec, sampler, n_trials, label):
    """用给定的 optuna 采样器跑 n_trials 次，返回逐次试验记录。"""
    trace, t0 = [], time.time()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    state = {"best": -np.inf}

    def objective(trial):
        params = suggest(spec, trial)
        auc, dt = evaluate(params)
        state["best"] = max(state["best"], auc)
        trace.append({"method": label, "trial": len(trace) + 1, "auc": auc,
                      "best_so_far": state["best"], "cum_seconds": time.time() - t0,
                      "trial_seconds": dt, "params": str(params)})
        return auc

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    print(f"  [{label}] {n_trials} 次试验，用时 {time.time()-t0:5.0f}s，"
          f"最优 3 折平均 AUC = {study.best_value:.4f}")
    return trace, study

def run_grid(spec, label):
    """网格搜索：按固定顺序枚举全部参数组合（实验 1 的 24 个点）。"""
    names = list(spec)
    combos = [dict(zip(names, v)) for v in itertools.product(
        *[spec[n]["values"] for n in names])]
    trace, t0, best = [], time.time(), -np.inf
    for params in combos:
        auc, dt = evaluate(params)
        best = max(best, auc)
        trace.append({"method": label, "trial": len(trace) + 1, "auc": auc,
                      "best_so_far": best, "cum_seconds": time.time() - t0,
                      "trial_seconds": dt, "params": str(params)})
    print(f"  [{label}] {len(combos)} 组全枚举，用时 {time.time()-t0:5.0f}s，"
          f"最优 3 折平均 AUC = {best:.4f}")
    return trace

# ================= 开始实验 =================
if not FROM_CACHE:
    print("\n" + "=" * 66)
    print("实验 1：网格 vs 随机 vs 贝叶斯（等预算，离散 24 点空间）")
    print("=" * 66)
    grid_n = int(np.prod([len(s["values"]) for s in SPACE1.values()]))
    print(f"空间共 {grid_n} 个组合，三种方法各给 {grid_n} 次评估预算\n")
    t_all = time.time()
    trace1 = []
    trace1 += run_grid(SPACE1, "Grid")
    trace1 += run_optuna(SPACE1, RandomSampler(seed=42), grid_n, "Random")[0]
    trace1 += run_optuna(SPACE1, TPESampler(seed=42, n_startup_trials=5), grid_n, "Bayesian(TPE)")[0]

    print("\n" + "=" * 66)
    print("实验 2：随机 vs 贝叶斯（等预算，6 旋钮大空间，网格不可行）")
    print("=" * 66)
    n_trials2 = 30
    print(f"网格需枚举 5^6 = {5**6:,} 组（约 {5**6 * 6 / 3600:.0f} 小时），已放弃；"
          f"两种方法各给 {n_trials2} 次预算\n")
    trace2 = []
    trace2 += run_optuna(SPACE2, RandomSampler(seed=42), n_trials2, "Random")[0]
    trace2 += run_optuna(SPACE2, TPESampler(seed=42, n_startup_trials=5), n_trials2, "Bayesian(TPE)")[0]

    trace_all = pd.concat([pd.DataFrame(trace1).assign(experiment="exp1_grid_vs_random_vs_tpe"),
                           pd.DataFrame(trace2).assign(experiment="exp2_random_vs_tpe")],
                          ignore_index=True)
    trace_all.to_csv(OUT / "tuning_method_compare.csv", index=False)
    print(f"\n实验总用时 {(time.time()-t_all)/60:.1f} 分钟，共 {evaluate_calls} 次模型评估")
else:
    trace_all = pd.read_csv(OUT / "tuning_method_compare.csv")
    print(f"读取试验记录: {len(trace_all)} 条")

df1 = trace_all[trace_all["experiment"] == "exp1_grid_vs_random_vs_tpe"].copy()
df2 = trace_all[trace_all["experiment"] == "exp2_random_vs_tpe"].copy()
GRID_N, TRIALS2 = int(df1["trial"].max() / 3), int(df2["trial"].max() / 2)

# ---------- 方法级汇总 ----------
summary = []
for exp, d in [("exp1", df1), ("exp2", df2)]:
    for m, g in d.groupby("method", sort=False):
        best_row = g.loc[g["auc"].idxmax()]
        summary.append({
            "experiment": exp, "method": m, "budget": len(g),
            "best_cv_auc": round(best_row["auc"], 4),
            "mean_cv_auc": round(g["auc"].mean(), 4),
            "trial_reached_best": int(best_row["trial"]),
            "total_seconds": round(g["cum_seconds"].iloc[-1], 1),
            "best_params": best_row["params"],
        })
summary = pd.DataFrame(summary)
summary["mean_gap_vs_random"] = summary.groupby("experiment")["mean_cv_auc"].transform(
    lambda s: s - s[summary.loc[s.index, "method"] == "Random"].iloc[0])
summary["mean_gap_vs_random"] = summary["mean_gap_vs_random"].round(4)
summary.to_csv(OUT / "tuning_method_summary.csv", index=False)
print("\n" + summary.to_string(index=False))

# ---------- 图：收敛曲线 + 平均试错质量 ----------
fig = plt.figure(figsize=(15.5, 4.8), dpi=130)
axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
colors = {"Grid": "#4C72B0", "Random": "#DD8452", "Bayesian(TPE)": "#55A868"}

for ax, d, title in [
    (axes[0], df1, f"(a) Experiment 1: equal budget ({GRID_N} evals)\ndiscrete 24-point space"),
    (axes[1], df2, f"(b) Experiment 2: equal budget ({TRIALS2} trials)\n6-knob space (grid infeasible)")]:
    for m, g in d.groupby("method", sort=False):
        ax.plot(g["trial"], g["best_so_far"], marker="o", markersize=3.5,
                linewidth=1.8, color=colors[m], label=m)
        ax.annotate(f"{g['best_so_far'].iloc[-1]:.4f}",
                    xy=(g["trial"].iloc[-1], g["best_so_far"].iloc[-1]),
                    xytext=(4, -3), textcoords="offset points",
                    fontsize=8, color=colors[m])
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best 3-fold CV AUC so far")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="lower right")
    lo, hi = d["best_so_far"].min(), d["best_so_far"].max()
    pad = max((hi - lo) * 0.18, 0.0004)
    ax.set_ylim(lo - pad, hi + pad)

# 第三格：平均试错质量（每个方法花掉的每一次评估，平均拿到了多少 AUC）
ax = axes[2]
methods = ["Grid", "Random", "Bayesian(TPE)"]
exps = [("exp1", df1, "Exp1 (24-pt space)"), ("exp2", df2, "Exp2 (large space)")]
width, xs = 0.36, np.arange(len(exps))
for k, m in enumerate(methods):
    vals = [d.loc[d["method"] == m, "auc"].mean() if (d["method"] == m).any() else np.nan
            for _, d, _ in exps]
    bars = ax.bar(xs + (k - 1) * width, vals, width * 0.92, label=m,
                  color=colors[m], edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.00015, f"{v:.4f}",
                    ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels([t for _, _, t in exps], fontsize=9)
ax.set_ylabel("Average AUC over all evaluations")
ax.set_title("(c) Quality of the trials each method spent", fontsize=10)
ax.grid(alpha=0.3, linestyle="--", axis="y")
lo = min(d.loc[d["method"].isin(methods), "auc"].mean() for _, d, _ in exps)
ax.set_ylim(lo - 0.0015, max(d["auc"].mean() for _, d, _ in exps) + 0.0012)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "tuning_method_compare.png", bbox_inches="tight")
plt.close(fig)

print(f"已保存: {OUT / 'tuning_method_compare.csv'}")
print(f"已保存: {OUT / 'tuning_method_summary.csv'}")
print(f"已保存: {OUT / 'tuning_method_compare.png'}")
