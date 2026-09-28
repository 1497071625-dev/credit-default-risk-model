# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
tuning_full_combo.py - 调参方法全组合对照实验（W3 补充，按老师反馈）
项目：信贷违约预测模型
输入：data/train_final.csv（38 特征 + isDefault）
输出：outputs/W3/tuning_full_combo.csv           全部试验记录（逐次）
      outputs/W3/tuning_full_combo_summary.csv   组合级汇总
      outputs/W3/tuning_full_combo_bigspace.csv  大空间（6 旋钮）逐次记录
      outputs/W3/tuning_full_combo.png           全组合收敛曲线（2×2）

目的：W3 正式调参针对 XGBoost 与 LightGBM 两个模型，但原方法对照实验只做了
      LightGBM 一个模型。本脚本把对照补成**全组合**：
      2 个模型 × 3 种方法（Grid / Random / Bayesian TPE）= 6 个组合，
      每个组合给相同的 24 次评估预算，画各自的收敛曲线。
      另外把"大空间"（6 个旋钮、网格不可行）的随机 vs 贝叶斯对照也补成两个模型，
      其中 LightGBM 沿用 tuning_method_compare.py 实验二的记录（同一子样本、同一折、
      同一空间设定与种子，重跑只是浪费时间），XGBoost 为本脚本新跑。

公平性：每个模型定义一套结构对应的 4 旋钮离散空间（各 24 个组合）；
        同一份 20 万行子样本、同一套 3 折交叉验证折；随机与贝叶斯共用
        同一份空间定义，区别只在"怎么选下一组"。

运行：.venv/bin/python scripts/W3/tuning_full_combo.py
     加 --from-cache 跳过小空间训练、加 --big-cache 跳过大空间训练（只重画图时两个都加）
     加 --big-only  只跑大空间（小空间读既有记录）
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
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
SKIP_SMALL = ("--from-cache" in sys.argv) or ("--big-only" in sys.argv)
SKIP_BIG = ("--from-cache" in sys.argv) or ("--big-cache" in sys.argv)

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W3"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 数据：与 W2/W3 完全一致的划分，再从训练集分层抽子样本 ----------
df = pd.read_csv(DATA / "train_final.csv")
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

N_SUB = 200_000
sub_idx = (y_train.groupby(y_train, group_keys=False)
           .apply(lambda s: s.sample(int(round(N_SUB * len(s) / len(y_train))),
                                     random_state=42)).index)
X_sub, y_sub = X_train.loc[sub_idx], y_train.loc[sub_idx]
print(f"数据：训练集 {len(X_train):,} 行中分层抽 {len(X_sub):,} 行"
      f"（违约率 {y_sub.mean()*100:.1f}%），3 折交叉验证")

CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ---------- 两个模型各一套结构对应的离散空间（各 24 个组合） ----------
SPACES = {
    "XGBoost": {
        "max_depth":        {"type": "cat", "values": [4, 6, 8]},
        "learning_rate":    {"type": "cat", "values": [0.05, 0.1]},
        "n_estimators":     {"type": "cat", "values": [200, 300]},
        "colsample_bytree": {"type": "cat", "values": [0.7, 1.0]},
    },
    "LightGBM": {
        "num_leaves":       {"type": "cat", "values": [31, 63, 127]},
        "learning_rate":    {"type": "cat", "values": [0.05, 0.1]},
        "n_estimators":     {"type": "cat", "values": [200, 300]},
        "colsample_bytree": {"type": "cat", "values": [0.7, 1.0]},
    },
}
MODEL_FACTORY = {
    "XGBoost": lambda p: XGBClassifier(**p, eval_metric="logloss", random_state=42,
                                       n_jobs=-1, verbosity=0),
    "LightGBM": lambda p: LGBMClassifier(**p, verbosity=-1, random_state=42, n_jobs=-1),
}

evaluate_calls = 0

def evaluate(model_name, params):
    """3 折交叉验证的平均 AUC（同一模型内三种方法共用同一套折，保证可比）。"""
    global evaluate_calls
    evaluate_calls += 1
    model = MODEL_FACTORY[model_name](params)
    auc = cross_val_score(model, X_sub, y_sub, cv=CV, scoring="roc_auc", n_jobs=1).mean()
    return float(auc)

def suggest(spec, trial):
    out = {}
    for name, s in spec.items():
        if s["type"] == "cat":
            out[name] = trial.suggest_categorical(name, s["values"])
        elif s["type"] == "int":
            out[name] = trial.suggest_int(name, s["low"], s["high"], step=s.get("step", 1))
        else:
            out[name] = trial.suggest_float(name, s["low"], s["high"], log=s.get("log", False))
    return out

def run_optuna(model_name, spec, sampler, n_trials, label):
    trace, t0 = [], time.time()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    state = {"best": -np.inf}

    def objective(trial):
        params = suggest(spec, trial)
        auc = evaluate(model_name, params)
        state["best"] = max(state["best"], auc)
        trace.append({"model": model_name, "method": label, "trial": len(trace) + 1,
                      "auc": auc, "best_so_far": state["best"],
                      "cum_seconds": time.time() - t0, "params": str(params)})
        return auc

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    print(f"  [{model_name} | {label}] {n_trials} 次试验，用时 {time.time()-t0:5.0f}s，"
          f"最优 3 折平均 AUC = {study.best_value:.4f}", flush=True)
    return trace

def run_grid(model_name, spec, label):
    names = list(spec)
    combos = [dict(zip(names, v)) for v in itertools.product(
        *[spec[n]["values"] for n in names])]
    trace, t0, best = [], time.time(), -np.inf
    for params in combos:
        auc = evaluate(model_name, params)
        best = max(best, auc)
        trace.append({"model": model_name, "method": label, "trial": len(trace) + 1,
                      "auc": auc, "best_so_far": best,
                      "cum_seconds": time.time() - t0, "params": str(params)})
    print(f"  [{model_name} | {label}] {len(combos)} 组全枚举，用时 {time.time()-t0:5.0f}s，"
          f"最优 3 折平均 AUC = {best:.4f}", flush=True)
    return trace

# ================= 全组合实验：2 模型 × 3 方法（4 旋钮小空间） =================
if not SKIP_SMALL:
    print("\n" + "=" * 70)
    print("全组合对照：XGBoost / LightGBM × Grid / Random / Bayesian TPE")
    print("=" * 70)
    t_all = time.time()
    trace_all = []
    for model_name, spec in SPACES.items():
        grid_n = int(np.prod([len(s["values"]) for s in spec.values()]))
        print(f"\n--- {model_name}（空间 {grid_n} 个组合，三方法各给 {grid_n} 次预算）---")
        trace_all += run_grid(model_name, spec, "Grid")
        trace_all += run_optuna(model_name, spec, RandomSampler(seed=42), grid_n, "Random")
        trace_all += run_optuna(model_name, spec, TPESampler(seed=42, n_startup_trials=5),
                                grid_n, "Bayesian(TPE)")
    trace_all = pd.DataFrame(trace_all)
    trace_all.to_csv(OUT / "tuning_full_combo.csv", index=False)
    print(f"\n全组合实验总用时 {(time.time()-t_all)/60:.1f} 分钟，共 {evaluate_calls} 次模型评估")
else:
    trace_all = pd.read_csv(OUT / "tuning_full_combo.csv")
    print(f"小空间：读取既有试验记录 {len(trace_all)} 条")

# ================= 大空间补充：6 旋钮，Random vs TPE，两个模型 =================
# 小空间只有 24 个点，可能被质疑"空间太小、方法之间当然没差别"。这里放大到 6 个旋钮
# （每个 5 档 → 5^6 = 15,625 组，网格在算力上不可行），只比随机与贝叶斯，各 30 次预算。
BIG_SPACES = {
    "XGBoost": {
        "max_depth":        {"type": "int",   "low": 4,    "high": 12},
        "learning_rate":    {"type": "float", "low": 0.01, "high": 0.15, "log": True},
        "n_estimators":     {"type": "int",   "low": 100,  "high": 500, "step": 50},
        "min_child_weight": {"type": "int",   "low": 1,    "high": 50},
        "colsample_bytree": {"type": "float", "low": 0.5,  "high": 1.0},
        "subsample":        {"type": "float", "low": 0.5,  "high": 1.0},
    },
    "LightGBM": {
        "num_leaves":        {"type": "int",   "low": 20,   "high": 150, "step": 10},
        "max_depth":         {"type": "int",   "low": 4,    "high": 12},
        "learning_rate":     {"type": "float", "low": 0.01, "high": 0.15, "log": True},
        "n_estimators":      {"type": "int",   "low": 100,  "high": 500, "step": 50},
        "min_child_samples": {"type": "int",   "low": 5,    "high": 50},
        "colsample_bytree":  {"type": "float", "low": 0.5,  "high": 1.0},
    },
}
BIG_TRIALS = 30
BIG_CSV = OUT / "tuning_full_combo_bigspace.csv"
# LightGBM 的大空间记录直接沿用 tuning_method_compare.py 实验二，本脚本只新跑 XGBoost；
# 若那份记录不存在（换环境重跑），才把 LightGBM 一起跑出来。
BIG_RUN = ["XGBoost"] if (OUT / "tuning_method_compare.csv").exists() \
    else ["XGBoost", "LightGBM"]

def attach_lgb_big(df):
    """把 tuning_method_compare.py 实验二（LightGBM 大空间）的记录并进来，避免重复训练。"""
    cmp_path = OUT / "tuning_method_compare.csv"
    if not cmp_path.exists() or "LightGBM" in set(df["model"]):
        return df
    old = pd.read_csv(cmp_path)
    old = old[old["experiment"].astype(str).str.startswith("exp2")
              & old["method"].isin(["Random", "Bayesian(TPE)"])].copy()
    if old.empty:
        return df
    lgb_big = old[["method", "trial", "auc", "best_so_far", "cum_seconds", "params"]]
    lgb_big.insert(0, "model", "LightGBM")
    lgb_big["source"] = "tuning_method_compare.py(exp2)"
    return pd.concat([df, lgb_big], ignore_index=True)

if (not SKIP_BIG) or (not BIG_CSV.exists()):
    print("\n" + "=" * 70)
    print(f"大空间对照（6 旋钮）：{', '.join(BIG_RUN)} × Random / Bayesian TPE，各 30 次")
    print("=" * 70)
    big_trace, t_big = [], time.time()
    for model_name in BIG_RUN:
        spec = BIG_SPACES[model_name]
        print(f"\n--- {model_name} 大空间（6 旋钮）---")
        big_trace += run_optuna(model_name, spec, RandomSampler(seed=42), BIG_TRIALS, "Random")
        big_trace += run_optuna(model_name, spec, TPESampler(seed=42, n_startup_trials=5),
                                BIG_TRIALS, "Bayesian(TPE)")
    big_df = pd.DataFrame(big_trace)
    big_df["source"] = "tuning_full_combo.py"
    big_df = attach_lgb_big(big_df)
    big_df.to_csv(BIG_CSV, index=False)
    print(f"\n大空间实验总用时 {(time.time()-t_big)/60:.1f} 分钟")
else:
    big_df = attach_lgb_big(pd.read_csv(BIG_CSV))
    big_df.to_csv(BIG_CSV, index=False)
    print(f"大空间：读取既有试验记录 {len(big_df)} 条")

# ---------- 组合级汇总 ----------
rows = []
for (model_name, method), g in trace_all.groupby(["model", "method"], sort=False):
    best_row = g.loc[g["auc"].idxmax()]
    rows.append({"model": model_name, "method": method, "budget": len(g),
                 "best_cv_auc": round(best_row["auc"], 4),
                 "mean_cv_auc": round(g["auc"].mean(), 4),
                 "trial_reached_best": int(best_row["trial"]),
                 "total_seconds": round(g["cum_seconds"].iloc[-1], 1),
                 "best_params": best_row["params"]})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "tuning_full_combo_summary.csv", index=False)
print("\n" + summary.to_string(index=False))

# ---------- 图：全组合 2×2（(a)(b) 小空间收敛曲线、(c) 试错质量、(d) 大空间对照） ----------
from matplotlib.lines import Line2D

COLORS = {"Grid": "#4C72B0", "Random": "#DD8452", "Bayesian(TPE)": "#55A868"}
BIG_COLORS = {"XGBoost": "#C44E52", "LightGBM": "#4C72B0"}
SHORT = {"Grid": "Grid", "Random": "Random", "Bayesian(TPE)": "TPE"}

fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4), dpi=130)
ax_xgb, ax_lgb, ax_quality, ax_big = axes.ravel()

def plot_convergence(ax, model_name, label):
    d = trace_all[trace_all["model"] == model_name]
    for method in ["Grid", "Random", "Bayesian(TPE)"]:
        g = d[d["method"] == method].sort_values("trial")
        ax.plot(g["trial"], g["best_so_far"], marker="o", markersize=3.5, linewidth=1.8,
                color=COLORS[method], label=method)
        ax.annotate(f"{g['best_so_far'].iloc[-1]:.4f}",
                    xy=(g["trial"].iloc[-1], g["best_so_far"].iloc[-1]),
                    xytext=(4, -3), textcoords="offset points",
                    fontsize=8.5, color=COLORS[method], fontweight="bold")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best 3-fold CV AUC so far")
    ax.set_title(f"{label}: three methods, equal budget (24 evaluations)", fontsize=10.5)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=9.5, loc="lower right")
    lo, hi = d["best_so_far"].min(), d["best_so_far"].max()
    pad = max((hi - lo) * 0.18, 0.0004)
    ax.set_ylim(lo - pad, hi + pad)

plot_convergence(ax_xgb, "XGBoost", "(a) XGBoost (4 knobs, 24-point space)")
plot_convergence(ax_lgb, "LightGBM", "(b) LightGBM (4 knobs, 24-point space)")

# ---- (c) 六个组合的试错质量：24 次试验 AUC 的分布 + 均值 + 最好一次 ----
ORDER = [(m, s) for m in ["XGBoost", "LightGBM"] for s in ["Grid", "Random", "Bayesian(TPE)"]]
pos = [1, 2, 3, 5.1, 6.1, 7.1]
data = [trace_all[(trace_all["model"] == m) & (trace_all["method"] == s)]["auc"].values
        for m, s in ORDER]
bp = ax_quality.boxplot(data, positions=pos, widths=0.62, patch_artist=True,
                        medianprops=dict(color="#333333", linewidth=1.1),
                        whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"),
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.45,
                                        markerfacecolor="#888888", markeredgecolor="none"))
for patch, (m, s) in zip(bp["boxes"], ORDER):
    patch.set_facecolor(COLORS[s])
    patch.set_alpha(0.55)
    patch.set_edgecolor("#555555")
for x, (m, s) in zip(pos, ORDER):
    vals = trace_all[(trace_all["model"] == m) & (trace_all["method"] == s)]["auc"]
    ax_quality.plot(x, vals.mean(), marker="D", markersize=5.5, color="#B03A2E", zorder=5)
    ax_quality.plot(x, vals.max(), marker="*", markersize=9, color="#1F3B73", zorder=5)
ax_quality.set_xticks(pos)
ax_quality.set_xticklabels([f"{'XGB' if m == 'XGBoost' else 'LGB'}\n{SHORT[s]}" for m, s in ORDER],
                           fontsize=9)
ax_quality.set_xlim(0.2, 7.9)
ax_quality.set_ylabel("3-fold CV AUC over 24 evaluations")
ax_quality.set_title("(c) Search quality: all 6 model x method combinations", fontsize=10.5)
ax_quality.grid(alpha=0.3, linestyle="--", axis="y")
ax_quality.legend(handles=[Line2D([], [], marker="D", linestyle="none", markersize=5.5,
                                  color="#B03A2E", label="mean of 24 trials"),
                           Line2D([], [], marker="*", linestyle="none", markersize=9,
                                  color="#1F3B73", label="best of 24 trials")],
                  fontsize=8.5, loc="lower right")

# ---- (d) 大空间（6 旋钮）：两个模型的 Random vs TPE ----
for model_name in ["XGBoost", "LightGBM"]:
    for method in ["Random", "Bayesian(TPE)"]:
        g = big_df[(big_df["model"] == model_name) & (big_df["method"] == method)]
        g = g.sort_values("trial")
        ax_big.plot(g["trial"], g["best_so_far"], linewidth=1.8,
                    linestyle="-" if model_name == "XGBoost" else "--",
                    color=COLORS[method], alpha=1.0 if model_name == "XGBoost" else 0.55,
                    label=f"{model_name} · {SHORT[method]}")
        ax_big.annotate(f"{g['best_so_far'].iloc[-1]:.4f}",
                        xy=(g["trial"].iloc[-1], g["best_so_far"].iloc[-1]),
                        xytext=(4, -3), textcoords="offset points", fontsize=8,
                        color=COLORS[method])
ax_big.set_xlabel("Number of evaluations")
ax_big.set_ylabel("Best 3-fold CV AUC so far")
ax_big.set_title("(d) Large space (6 knobs, ~15,625 grid points): Random vs Bayesian",
                 fontsize=10.5)
ax_big.grid(alpha=0.3, linestyle="--")
ax_big.legend(fontsize=8.5, loc="lower right", ncol=2)
lo, hi = big_df["best_so_far"].min(), big_df["best_so_far"].max()
pad = max((hi - lo) * 0.18, 0.0004)
ax_big.set_ylim(lo - pad, hi + pad)

fig.suptitle("Hyperparameter search methods, full combination of model x method "
             "(200k-row subsample, 3-fold CV)", fontsize=11.5, y=1.00)
fig.tight_layout()
fig.savefig(OUT / "tuning_full_combo.png", bbox_inches="tight")
plt.close(fig)

print(f"\n已保存: {OUT / 'tuning_full_combo.csv'}")
print(f"已保存: {OUT / 'tuning_full_combo_summary.csv'}")
print(f"已保存: {BIG_CSV}")
print(f"已保存: {OUT / 'tuning_full_combo.png'}")
