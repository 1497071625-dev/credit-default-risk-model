# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
threshold_sweep.py - 阈值扫描：把「设一条线 → 看亏多少钱」补成完整曲线
（W4 自用讲解材料，不进交付包；依赖 W4 已交付的产物，不改动它们）

要回答的三个问题：
  Q1「校准后 0.5 这条线直接失效了」——这个说法成立吗？
     Platt 校准是【严格单调】变换，所以：
       · 全局阈值策略族在两条轴上一一对应，可实现的决策集合【完全相同】；
       · p_raw < 0.5 和 p_cal < q 给出的是同一份拒绝名单（本脚本逐行验证）；
     => 「0.5 失效」不是校准造成的后果，只是同一个策略换了个刻度叫法。
        真正的事实是：0.5 从来就不是业务最优线（报告早就写明是"数学默认值"）。

  Q2 那校准到底有什么用？
     关键区分：
       · 全局阈值（一条线管所有人）只依赖【排序】→ 校准对它零影响（可任意单调换轴）；
       · 逐笔期望损失规则 EP(p) 是 p 的【线性函数】，依赖概率数值本身 →
         必须校准，否则每笔账都算错（不校准直接用框架 → 拒绝率虚高到 87.7%）。
     校准的意义在于让"概率 × 金额"这个乘法成立，不在于挪动那条线。

  Q3「设阈值看亏多少钱」为什么舍弃了？
     没舍弃，但被拆碎了：报告 §10.1 只剩 precision/recall（没有钱），
     §10.2 只保留了固定阈值 0.5 一行金额。本脚本把它还原成完整形态：
     横轴阈值、纵轴钱，并且把"最佳单一线"和"逐笔算账"画在一起对比。

输出：outputs/W4/threshold_sweep.csv    阈值扫描明细（原始轴 + 校准轴 + 金额）
      outputs/W4/threshold_equiv.csv    等效阈值对照表（同一拒绝率在两条轴上的刻度）
      outputs/W4/threshold_sweep.png    净增益 vs 阈值（两条轴叠画 + 逐笔算账上限线）
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

# 中文字体（macOS 系统字体；仓库其他图沿用英文标签，本图是自用讲解件故用中文）
for _f in ["/System/Library/Fonts/Hiragino Sans GB.ttc",
           "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
OUT = BASE_DIR / "outputs" / "W4"

df = pd.read_csv(OUT / "test_predictions_calibrated.csv")
y = df["y_true"].to_numpy(int)
L = df["loanAmnt"].to_numpy(float)
T = df["term"].to_numpy(float)
rate = df["interestRate"].to_numpy(float) / 100.0
p_raw = df["prob_raw"].to_numpy(float)
p_cal = df["prob_cal"].to_numpy(float)

V_NEUTRAL, REC_BASE = 0.06, 0.3
V_OPT, V_CONS = rate, 0.03          # 乐观：利息全额；保守：净息差 3%


def unit_ep(p, v, rec=REC_BASE):
    return (1 - p) * v * T - p * (1 - rec)


# 统一评价尺子：中性口径 + 校准概率（与 expected_loss_framework.py 完全一致）
ep_scale = unit_ep(p_cal, V_NEUTRAL)
ep_opt = unit_ep(p_cal, V_OPT)
ep_cons = unit_ep(p_cal, V_CONS)


def score(approve):
    """给一个批准集合打分。金额单位：万元。"""
    reject = ~approve
    bad = y == 1
    return {
        "被拒占比": reject.mean(),
        "拦截率": bad[reject].sum() / bad.sum() if bad.sum() else np.nan,
        "被拒者精确率": bad[reject].mean() if reject.any() else np.nan,
        "误伤率": (~bad)[reject].sum() / (~bad).sum(),
        "总期望利润(万元)": np.where(approve, ep_scale * L, 0.0).sum() / 1e4,
        "利润_乐观(万元)": np.where(approve, ep_opt * L, 0.0).sum() / 1e4,
        "利润_保守(万元)": np.where(approve, ep_cons * L, 0.0).sum() / 1e4,
        # 仪表板页签③的口径：相对"全批放款"的实现值净增益 = 避免坏账 − 放弃收入
        "净增益_实现值(万元)": (
            (L[reject & bad] * (1 - REC_BASE)).sum()
            - (L[reject & ~bad] * V_NEUTRAL * T[reject & ~bad]).sum()
        ) / 1e4,
    }


print("=" * 74)
print("Q1  校准是不是严格单调？——决定「0.5 失效」这个说法的性质")
print("=" * 74)
order_raw = np.argsort(p_raw)
cal_sorted = p_cal[order_raw]
viol = int((np.diff(cal_sorted) < 0).sum())
strict = bool((np.diff(cal_sorted) > 0).all())
rho = float(np.corrcoef(pd.Series(p_raw).rank(), pd.Series(p_cal).rank())[0, 1])
print(f"  按 p_raw 排序后 p_cal 的违反次数（要求 0）：{viol}")
print(f"  严格递增（无并列）：{strict}")
print(f"  Spearman(p_raw, p_cal) = {rho:.10f}")
print(f"  Pearson(p_raw, p_cal)  = {float(np.corrcoef(p_raw, p_cal)[0,1]):.6f}  "
      f"（不是 1 —— 非线性，但单调）")

# 决策集合等价性：p_raw < 0.5 与 p_cal < q 是否给出同一份拒绝名单
rej_raw_05 = p_raw >= 0.5
q_star = float(p_cal[rej_raw_05].min()) if rej_raw_05.any() else np.nan
same = bool(np.array_equal(rej_raw_05, p_cal >= q_star))
print()
print(f"  原口径 p_raw ≥ 0.5 的拒绝人数：{int(rej_raw_05.sum()):,}（{rej_raw_05.mean()*100:.1f}%）")
print(f"  等效校准轴刻度 q* = {q_star:.6f}")
print(f"  校准轴 p_cal ≥ q* 的拒绝人数：{int((p_cal >= q_star).sum()):,}")
print(f"  两份名单是否逐行完全一致：{same}")
print("  => 同一个策略，两套刻度。'0.5 失效'只是换刻度后的叫法变化，不是策略被破坏。")

# ---------- 等效阈值对照表 ----------
qs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
eq_rows = []
for q in qs:
    t_raw = float(np.quantile(p_raw, 1 - q))
    t_cal = float(np.quantile(p_cal, 1 - q))
    eq_rows.append({
        "目标拒绝率": q,
        "原始轴阈值 p_raw ≥": t_raw,
        "校准轴阈值 p_cal ≥": t_cal,
        "实际拒绝率": float((p_raw >= t_raw).mean()),
    })
equiv = pd.DataFrame(eq_rows)
equiv.to_csv(OUT / "threshold_equiv.csv", index=False)
print()
print("=" * 74)
print("等效阈值对照：想在两条轴上切出同一个拒绝率，刻度分别是多少")
print("=" * 74)
print(equiv.round(4).to_string(index=False))

# ---------- 阈值扫描 ----------
grid_raw = np.arange(0.001, 1.0, 0.001)
grid_cal = np.arange(0.001, 1.0, 0.001)

rows = []
for t in grid_raw:
    rows.append({"轴": "原始(未校准)", "阈值": t, **score(p_raw < t)})
for t in grid_cal:
    rows.append({"轴": "校准后", "阈值": t, **score(p_cal < t)})
sweep = pd.DataFrame(rows)
sweep.to_csv(OUT / "threshold_sweep.csv", index=False)

best = {}
for ax in ["原始(未校准)", "校准后"]:
    sub = sweep[sweep["轴"] == ax]
    r = sub.loc[sub["总期望利润(万元)"].idxmax()]
    best[ax] = (float(r["阈值"]), float(sub["总期望利润(万元)"].max()),
                float(r["被拒占比"]))

print()
print("=" * 74)
print("阈值扫描结果：最佳「单一线」在两条轴上的位置（统一中性尺子）")
print("=" * 74)
for ax, (t, prof, rej) in best.items():
    print(f"  {ax}：最优阈值 t* = {t:.3f} → 拒绝率 {rej*100:.1f}%，"
          f"总期望利润 {prof:,.0f} 万元")

# 一致性检验：最优阈值在两条轴上是否对应同一决策集合
t_best_raw = best["原始(未校准)"][0]
rej_best_raw = p_raw < t_best_raw
q_equiv = float(p_cal[~rej_best_raw].min())
t_best_cal = best["校准后"][0]
print()
print(f"  原始轴最优 t*={t_best_raw:.3f} 的【精确】等效校准刻度 = {q_equiv:.6f}；"
      f"校准轴独立扫出的最优 = {t_best_cal:.3f}（网格步长 0.001）")
same_exact = bool(np.array_equal(rej_best_raw, p_cal < q_equiv))
mism = int((rej_best_raw != (p_cal < t_best_cal)).sum())
print(f"  按精确刻度比 → 决策集合完全一致：{same_exact}")
print(f"  按网格刻度比 → 不一致 {mism} 人（占 {mism/len(df)*100:.4f}%，"
      f"纯粹是 0.001 网格取整，不是方法差异）")

# 两条金额口径的最优点必须落在同一档，仪表板③才敢只用净增益一套数
sub_raw_grid = sweep[sweep["轴"] == "原始(未校准)"]
t_best_gain = float(sub_raw_grid.loc[sub_raw_grid["净增益_实现值(万元)"].idxmax(), "阈值"])
assert abs(t_best_gain - t_best_raw) < 1e-9, \
    f"净增益口径最优 {t_best_gain:.3f} 与期望利润口径最优 {t_best_raw:.3f} 不在同一档"

# ---------- 三条基准线 ----------
approve_all = np.ones(len(df), bool)
row_all = score(approve_all)
row_thr05 = score(p_raw < 0.5)
row_el = score(unit_ep(p_cal, V_NEUTRAL) > 0)
row_el_raw = score(unit_ep(p_raw, V_NEUTRAL) > 0)

print()
print("=" * 74)
print("四条基准线的对照（金额单位：万元）")
print("=" * 74)
bench = pd.DataFrame([
    {"策略": "全放（不筛选）", **row_all},
    {"策略": "单一线 · 原口径 0.5（=校准轴 "
             f"{q_star:.3f}）", **row_thr05},
    {"策略": f"单一线 · 最优 t*={t_best_raw:.3f}（原始轴）",
     **score(p_raw < t_best_raw)},
    {"策略": "逐笔期望损失框架（中性 6%/30%）", **row_el},
    {"策略": "逐笔框架 · 未校准概率（对照）", **row_el_raw},
])
print(bench[["策略", "被拒占比", "拦截率", "被拒者精确率", "误伤率",
             "总期望利润(万元)", "净增益_实现值(万元)"]].round(4).to_string(index=False))

gap = row_el["总期望利润(万元)"] - score(p_raw < t_best_raw)["总期望利润(万元)"]
print()
print(f"  → 单一线最优 {score(p_raw < t_best_raw)['总期望利润(万元)']:,.0f} 万元 "
      f"vs 逐笔算账 {row_el['总期望利润(万元)']:,.0f} 万元，"
      f"差 {gap:,.0f} 万元（{gap/abs(row_el['总期望利润(万元)'])*100:.1f}%）")
print(f"  → 固定 0.5（{row_thr05['总期望利润(万元)']:,.0f} 万元）距离单一线最优还差 "
      f"{score(p_raw < t_best_raw)['总期望利润(万元)'] - row_thr05['总期望利润(万元)']:,.0f} 万元"
      f" —— 说明 0.5 这个数本身就是拍偏的，跟校准无关。")

# ---------- 图 ----------
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))

ax = axes[0]
for ax_name, grid, p, color in [
    ("原始轴 p_raw", grid_raw, p_raw, "#b0483f"),
    ("校准轴 p_cal", grid_cal, p_cal, "#1f6fb2"),
]:
    sub = sweep[sweep["轴"] == ("原始(未校准)" if ax_name.startswith("原始") else "校准后")]
    ax.plot(sub["阈值"], sub["净增益_实现值(万元)"], color=color, lw=1.8, label=ax_name)
ax.axhline(row_el["净增益_实现值(万元)"], color="#3f9a6f", ls="--", lw=1.4,
           label=f"逐笔算账 = {row_el['净增益_实现值(万元)']:,.0f} 万（上限）")
ax.axvline(0.5, color="#b0483f", ls=":", lw=1.2, alpha=.8)
ax.annotate(f"旧口径 0.5（拒 {row_thr05['被拒占比'] * 100:.1f}%）",
            xy=(0.5, row_thr05["净增益_实现值(万元)"]),
            xytext=(0.06, 0.12), textcoords="axes fraction", fontsize=8.5, color="#b0483f",
            arrowprops=dict(arrowstyle="->", color="#b0483f", lw=.9))

# 曲线最高点：单一线最优（净增益口径与期望利润口径落在同一档，脚本末尾有互校）
row_best_raw = score(p_raw < t_best_raw)
ax.plot([t_best_raw], [row_best_raw["净增益_实现值(万元)"]], "o", color="#b0483f", ms=4)
ax.annotate(f"单一线最优 t*={t_best_raw:.3f}", xy=(t_best_raw, row_best_raw["净增益_实现值(万元)"]),
            xytext=(0.58, 0.82), textcoords="axes fraction", fontsize=8.5, color="#b0483f",
            arrowprops=dict(arrowstyle="->", color="#b0483f", lw=.9))
ax.axvline(q_star, color="#1f6fb2", ls=":", lw=1.2, alpha=.8)
ax.set_xlabel("阈值（两条轴刻度不同，曲线形状相同）")
ax.set_ylabel("净增益（万元，相对全批放款）")
ax.set_title("设一条线，看钱随阈值怎么变")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.grid(alpha=.25)

ax = axes[1]
sub_raw = sweep[sweep["轴"] == "原始(未校准)"].sort_values("被拒占比")
# 拒绝率低于 1% 时被拒样本只剩几十人，被拒者精确率会在 0~1 之间乱跳，
# 连起来就是左端那条竖线（与 P9 的 PR 图同源）。只画拒绝率 ≥1%（约 800 人）的一段。
sub_raw = sub_raw[sub_raw["被拒占比"] >= 0.01]
ax.plot(sub_raw["被拒占比"] * 100, sub_raw["拦截率"] * 100, color="#1f6fb2", lw=1.8,
        label="拦截率（真违约被拒）")
ax.plot(sub_raw["被拒占比"] * 100, sub_raw["被拒者精确率"] * 100, color="#d18b2a", lw=1.8,
        label="被拒者中真违约占比")
ax.plot(sub_raw["被拒占比"] * 100, sub_raw["误伤率"] * 100, color="#8a6fb0", lw=1.8,
        label="误伤率（正常人被拒）")
ax.axvline(row_thr05["被拒占比"] * 100, color="#b0483f", ls=":", lw=1.2)
ax.annotate("旧口径 0.5", xy=(row_thr05["被拒占比"] * 100, 20),
            xytext=(row_thr05["被拒占比"] * 100 + 3, 8), fontsize=8.5, color="#b0483f")
ax.set_xlim(1, 100)
ax.set_xticks([1, 20, 40, 60, 80, 100])
ax.set_xlabel("拒绝率（%）\n拒绝率低于 1%（不足 800 人）的一段只剩几十个被拒样本，读数不稳，未画")
ax.set_ylabel("比例（%）")
ax.set_title("同一批决策的三条读数（与阈值一一对应，与轴的刻度无关）")
ax.legend(frameon=False, fontsize=8.5)
ax.grid(alpha=.25)

plt.tight_layout()
plt.savefig(OUT / "threshold_sweep.png", dpi=150)
plt.close()

print()
print(f"已写出：{OUT/'threshold_sweep.csv'}")
print(f"        {OUT/'threshold_equiv.csv'}")
print(f"        {OUT/'threshold_sweep.png'}")
