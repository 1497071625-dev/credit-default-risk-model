# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
model_significance_test.py - 模型差异的统计显著性检验（阶段 5.4 补充）
项目：信贷违约预测模型
输入：data/train_final.csv
输出：outputs/W3/significance_test.csv    两两模型的 AUC 差异检验结果
      outputs/W3/significance_ci.png      AUC 差异 + 95% 置信区间森林图
缓存：.cache/W3/val_predictions.csv       各模型在验证集上的预测概率（避免重复训练）
      .cache/W3/bootstrap_auc.npy        Bootstrap 每次重抽样的 AUC（避免重复重抽样）
      说明：缓存放 .cache/（不进交付包），图内文字用英文，与 W1/W2/W3 其他交付图一致。

要回答的问题：验证集 AUC 排行榜上 Stacking 0.7298、LGB 0.7294、XGB 0.7286……
     这些小数点后第 3、4 位的差距，**是真的模型更强，还是抽样噪声？**
     如果差异和噪声分不开，那么"选哪个模型"就不该由这 0.0005 决定，
     而应该看别的（可解释性、跑得快不快、上线好不好维护）。

统计方法（两条腿互相印证）：
  1. DeLong 检验（Sun & Xu 2014 快速算法）：AUC 差值的配对检验，
     风控/医学里比较两个打分模型 AUC 的标准做法。
     同一批验证样本上跑两个模型 → 配对，所以用"配对"的协方差而不是独立方差。
  2. Bootstrap 重抽样：验证集有放回抽 1000 次，每次重算全部模型的 AUC，
     看差值的分布。给出 95% 置信区间 + 符号翻转比例（p 值）。
     优点是不依赖正态近似，和 DeLong 结果对比可验证结论稳健。

判读标准：p > 0.05 或置信区间跨 0 → 两组模型分不出高下，差异属噪声。
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W3"
OUT.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / ".cache" / "W3"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE = CACHE_DIR / "val_predictions.csv"       # 验证集预测（避免重复训练）
CACHE_BOOT = CACHE_DIR / "bootstrap_auc.npy"    # Bootstrap 结果（避免重复重抽样）
N_BOOT = 1000

# ==================== 1. 数据与模型（与 W2/W3 完全一致，保证数字可比） ====================
df = pd.read_csv(DATA / "train_final.csv")
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"划分: 训练 {len(X_train):,} / 验证 {len(X_val):,}（检验用）")

# 参与检验的模型：覆盖 W3 报告里所有"值得争一争"的模型
MODELS = {
    "XGB(基线)": lambda: XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                       scale_pos_weight=scale_pos, eval_metric="logloss",
                                       random_state=42, n_jobs=-1, verbosity=0),
    "XGB(优化)": lambda: XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.7,
                                       min_child_weight=3, scale_pos_weight=scale_pos,
                                       eval_metric="logloss", random_state=42,
                                       n_jobs=-1, verbosity=0),
    "LGB(优化)": lambda: LGBMClassifier(n_estimators=200, max_depth=8, num_leaves=63,
                                        learning_rate=0.05, subsample=0.8,
                                        colsample_bytree=0.7, scale_pos_weight=scale_pos,
                                        verbosity=-1, random_state=42, n_jobs=-1),
}

def _base_estimators():
    """Voting / Stacking 的内部基模型（与 W3 保持一致）。"""
    return [
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=12,
                                      class_weight="balanced", random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.7, min_child_weight=3,
                              scale_pos_weight=scale_pos, eval_metric="logloss",
                              random_state=42, n_jobs=-1, verbosity=0)),
        ("lgb", LGBMClassifier(n_estimators=200, max_depth=8, num_leaves=63, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.7,
                               scale_pos_weight=scale_pos, verbosity=-1,
                               random_state=42, n_jobs=-1)),
    ]

MODELS["Voting(集成)"] = lambda: VotingClassifier(estimators=_base_estimators(),
                                                  voting="soft", n_jobs=1)
MODELS["Stacking(集成)"] = lambda: StackingClassifier(
    estimators=_base_estimators(),
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
    cv=5, n_jobs=1)

# ---------- 取验证集预测概率（有缓存则直接用，省去十几分钟重训） ----------
if CACHE.exists():
    cached = pd.read_csv(CACHE)
    y_val = cached["isDefault"].astype(int)
    probs = {c: cached[c].values for c in cached.columns if c != "isDefault"}
    print(f"读取预测缓存: {CACHE.name}（{len(probs)} 个模型 × {len(y_val):,} 行）")
else:
    probs = {}
    for name, build in MODELS.items():
        t0 = time.time()
        m = build().fit(X_train, y_train)
        probs[name] = m.predict_proba(X_val)[:, 1]
        print(f"  训练 {name:14s} 用时 {time.time()-t0:5.0f}s | "
              f"验证集 AUC = {roc_auc_score(y_val, probs[name]):.4f}")
    pd.DataFrame({"isDefault": np.asarray(y_val), **probs}).to_csv(CACHE, index=False)
    print(f"验证集预测已缓存: {CACHE}")

names = list(probs)
P = np.vstack([probs[n] for n in names])            # (k, N)
y_arr = np.asarray(y_val, dtype=int)

# ==================== 2. DeLong 检验（Sun & Xu 2014 快速算法） ====================
def midrank(x):
    """带并列（相同值）处理的秩：并列取平均秩。"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(N)
    out[J] = T
    return out

def delong(y_true, scores):
    """返回 (每个模型的 AUC, AUC 估计量的协方差矩阵)。scores 形状 (k, N)。"""
    order = np.argsort(-y_true, kind="mergesort")   # 正样本（违约）排前面
    preds = scores[:, order]
    y_sorted = y_true[order]
    m = int(y_sorted.sum())                          # 违约数
    n = len(y_sorted) - m                            # 正常数
    k = preds.shape[0]

    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = midrank(preds[r, :m])
        ty[r] = midrank(preds[r, m:])
        tz[r] = midrank(preds[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v_pos = (tz[:, :m] - tx) / n          # 每个违约样本的"贡献"
    v_neg = 1.0 - (tz[:, m:] - ty) / m    # 每个正常样本的"贡献"
    cov = np.atleast_2d(np.cov(v_pos)) / m + np.atleast_2d(np.cov(v_neg)) / n
    return aucs, cov

aucs, cov = delong(y_arr, P)
auc_table = {n: a for n, a in zip(names, aucs)}
print("\nDeLong 估计的 AUC（与 sklearn 对照）:")
for n, a in zip(names, aucs):
    print(f"  {n:14s} DeLong={a:.4f}  sklearn={roc_auc_score(y_arr, probs[n]):.4f}")

# ==================== 3. Bootstrap 重抽样（不依赖正态近似） ====================
if CACHE_BOOT.exists():
    boot = np.load(CACHE_BOOT)
    print(f"读取 Bootstrap 缓存: {CACHE_BOOT.name}"
          f"（{boot.shape[0]} 次 × {boot.shape[1]} 个模型）")
else:
    rng = np.random.default_rng(42)
    N = len(y_arr)
    boot = np.full((N_BOOT, len(names)), np.nan)
    for b in range(N_BOOT):
        idx = rng.integers(0, N, N)
        yb = y_arr[idx]
        if yb.min() == yb.max():
            continue
        for r in range(len(names)):
            boot[b, r] = roc_auc_score(yb, P[r, idx])
    np.save(CACHE_BOOT, boot)
    print(f"Bootstrap {N_BOOT} 次完成，有效重抽样 {np.sum(~np.isnan(boot[:,0]))} 次"
          f"（已缓存: {CACHE_BOOT}）")

# ==================== 4. 两两比较 ====================
rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
        z = (aucs[i] - aucs[j]) / np.sqrt(var) if var > 0 else np.nan
        p_delong = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        d = boot[:, i] - boot[:, j]
        d = d[~np.isnan(d)]
        ci_lo, ci_hi = np.percentile(d, [2.5, 97.5])
        p_boot = min(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0)
        rows.append({
            "model_a": a, "model_b": b,
            "auc_a": round(aucs[i], 4), "auc_b": round(aucs[j], 4),
            "auc_diff": round(aucs[i] - aucs[j], 4),
            "delong_z": round(z, 2), "delong_p": round(p_delong, 4),
            "boot_ci_low": round(ci_lo, 5), "boot_ci_high": round(ci_hi, 5),
            "boot_p": round(p_boot, 4),
            "significant_5pct": "是" if (p_delong < 0.05 and p_boot < 0.05) else "否",
        })

sig = pd.DataFrame(rows).sort_values("auc_diff", ascending=False)
sig.to_csv(OUT / "significance_test.csv", index=False)
print("\n" + "=" * 90)
print("两两比较（diff = 前者减后者；p<0.05 且置信区间不跨 0 才算真的分得出高下）")
print("=" * 90)
print(sig.to_string(index=False))

# ==================== 5. 森林图：AUC 差异 + 95% 置信区间 ====================
# 图内用英文标签，避免中文字形缺失（与 W1/W2/W3 其他交付图风格一致）
NAME_EN = {
    "LR(基线)": "LR (baseline)", "RF(基线)": "RF (baseline)",
    "XGB(基线)": "XGB (baseline)", "XGB(优化)": "XGB (tuned)",
    "LGB(优化)": "LGB (tuned)", "Voting(集成)": "Voting (ensemble)",
    "Stacking(集成)": "Stacking (ensemble)",
}
fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(sig) + 2.2), dpi=130)
ypos = np.arange(len(sig))[::-1]
for yp, (_, r) in zip(ypos, sig.iterrows()):
    color = "#C44E52" if r["significant_5pct"] == "是" else "#8C8C8C"
    ax.plot([r["boot_ci_low"], r["boot_ci_high"]], [yp, yp], color=color, linewidth=2.2)
    ax.plot(r["auc_diff"], yp, "o", color=color, markersize=6)
    ax.text(r["boot_ci_high"] + 0.00008, yp,
            f"{r['auc_diff']:+.4f}  (p={r['delong_p']:.3f})",
            va="center", fontsize=8.5, color=color)
ax.axvline(0, color="black", linewidth=1.1, linestyle="--")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{NAME_EN.get(r['model_a'], r['model_a'])}  -  "
                    f"{NAME_EN.get(r['model_b'], r['model_b'])}" for _, r in sig.iterrows()],
                   fontsize=9)
ax.set_xlabel("AUC difference (95% bootstrap CI)\nCI crossing 0 = the two models cannot be told apart")
ax.set_title("Pairwise AUC differences on the validation set", fontsize=11)
ax.grid(alpha=0.3, linestyle="--", axis="x")
ax.legend(handles=[plt.Line2D([], [], color="#C44E52", marker="o", lw=2.2,
                              label="Significant at 5% (DeLong and bootstrap agree)"),
                   plt.Line2D([], [], color="#8C8C8C", marker="o", lw=2.2,
                              label="Not significant: models cannot be told apart")],
          loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "significance_ci.png", bbox_inches="tight")
plt.close(fig)

n_sig = (sig["significant_5pct"] == "是").sum()
print(f"\n{len(sig)} 组两两比较中，只有 {n_sig} 组在 5% 水平上显著。")
print(f"已保存: {OUT / 'significance_test.csv'}")
print(f"已保存: {OUT / 'significance_ci.png'}")
