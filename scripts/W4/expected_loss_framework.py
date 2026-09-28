"""
expected_loss_framework.py - 期望损失决策框架与敏感性分析（W4 Track B 第②项）
项目：信贷违约预测模型

为什么做这个：
  固定阈值（如 0.5）把所有客户按同一把尺子切，但一笔贷款"值不值得放"不只看违约概率：
    · 金额：赚和亏都按金额放大；
    · 利率 / 期限：正常还款能赚多少利息；
    · 回收率：真违约了能追回多少，决定实际亏多少。
  把这几件事合成一笔账（概率用第①项校准后的概率，否则概率虚高、账会算错）：

    每 1 元贷款的期望利润 EP = (1 - p) × 净收益率 × 期限 - p × (1 - 回收率)

  决策规则：EP > 0 批准；EP <= 0 拒绝。等价于逐笔的隐含概率阈值：
    p* = 净收益率 × 期限 / (净收益率 × 期限 + 1 - 回收率)
  期限长、利率高的贷款"赚得多"，能容忍更高的违约概率；利差薄的贷款则更严格。

输入：outputs/W4/test_predictions_calibrated.csv（79,850 人：原始/校准概率 + 金额/期限/利率）
产出：outputs/W4/expected_loss_summary.csv       策略对比（全放 / 固定阈值 / 期望损失框架）
      outputs/W4/expected_loss_sensitivity.csv   回收率 × 收益口径 的敏感性表（16 组合）
      outputs/W4/expected_loss_sensitivity.png   敏感性可视化（拒绝率、拦截率 vs 回收率）

口径说明（演示性测算：参数为假设值，需机构财务口径确认，不构成利润结论）：
  - 基准：净收益率 = 每笔实际利率（利息收入全算、未扣资金成本）；回收率 0.3；
  - 敏感性：净收益率 ∈ {实际利率, 9%, 6%, 3%} × 回收率 ∈ {20%, 30%, 40%, 50%}；
  - 违约损失按"未收回本金"简化，不计催收成本、提前还款与资金时间价值。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
p_cal = df["prob_cal"].to_numpy(float)
p_raw = df["prob_raw"].to_numpy(float)

REC_BASE = 0.3


def unit_ep(p, v, rec):
    """每 1 元贷款的期望利润。v：净收益率（逐笔数组或标量）；rec：回收率。"""
    return (1 - p) * v * T - p * (1 - rec)


def evaluate(name, approve, ep_unit, note=""):
    """给一个"批准集合"打分：拒绝规模、拦截效果、经济后果（统一用基准口径的 EP 评价）。"""
    reject = ~approve
    bad = y == 1
    good = ~bad
    profit = np.where(approve, ep_unit * L, 0.0).sum() / 1e4  # 万元
    return {
        "策略": name,
        "批准占比": approve.mean(),
        "被拒占比": reject.mean(),
        "拦截率(真违约被拒占比)": bad[reject].sum() / bad.sum(),
        "被拒者精确率(里面真违约占比)": bad[reject].mean() if reject.any() else np.nan,
        "误伤率(正常人被拒占比)": good[reject].sum() / good.sum(),
        "总期望利润(万元)": profit,
        "说明": note,
    }


# ---------- 策略对比 ----------
# 收益口径按情景假设（利息全额 = 乐观上界；净息差 6% = 中性；3% = 保守），回收率统一 30%。
# 所有策略的经济后果统一用"中性口径"的期望利润评价，保证横向可比。
V_NEUTRAL = 0.06
ep_scale = unit_ep(p_cal, V_NEUTRAL, REC_BASE)     # 经济评价的统一尺子（校准概率）
approve_all = np.ones(len(df), dtype=bool)
approve_thr = p_raw < 0.5                          # 固定阈值 0.5（未校准概率，W4 第 10 节口径）
approve_el_raw = unit_ep(p_raw, V_NEUTRAL, REC_BASE) > 0

scenarios = [("乐观（利息全额）", rate), ("中性（净息差 6%）", V_NEUTRAL), ("保守（净息差 3%）", 0.03)]
rows = [
    evaluate("全放（不筛选）", approve_all, ep_scale, "基准"),
    evaluate("固定阈值 0.5（未校准概率）", approve_thr, ep_scale, "第 10 节原口径"),
]
for name, v in scenarios:
    rows.append(evaluate(f"期望损失框架 · {name}", unit_ep(p_cal, v, REC_BASE) > 0,
                         ep_scale, "校准概率（回收率 30%）"))
rows.append(evaluate("期望损失框架（未校准概率）", approve_el_raw, ep_scale, "对照：不校准的后果"))

summary = pd.DataFrame(rows)[["策略", "批准占比", "被拒占比", "拦截率(真违约被拒占比)",
                              "被拒者精确率(里面真违约占比)", "误伤率(正常人被拒占比)",
                              "总期望利润(万元)", "说明"]]
summary.to_csv(OUT / "expected_loss_summary.csv", index=False)
print("=" * 62)
print("策略对比（测试集 79,850 人；经济评价统一用中性口径：净息差 6%、回收率 30%）")
print("=" * 62)
print(summary.round(4).to_string(index=False))
print()
print("解读：")
for _, r in summary.iterrows():
    print(f"  · {r['策略']}：拒绝 {r['被拒占比']*100:.1f}% 的申请，"
          f"拦截 {r['拦截率(真违约被拒占比)']*100:.1f}% 的违约者，"
          f"总期望利润 {r['总期望利润(万元)']:,.0f} 万元")

# ---------- 敏感性：回收率 × 收益口径 ----------
calibers = [
    ("Interest income (full rate)", rate),
    ("Net margin 9%", 0.09),
    ("Net margin 6%", 0.06),
    ("Net margin 3%", 0.03),
]
recs = [0.2, 0.3, 0.4, 0.5]

rows = []
for label, v in calibers:
    vt = np.asarray(v) * T
    p_star = vt / (vt + 1 - np.array(recs)[:, None])          # 每笔隐含阈值（4×n）
    for j, rec in enumerate(recs):
        ep = unit_ep(p_cal, v, rec)
        approve = ep > 0
        reject = ~approve
        rows.append({
            "收益口径": label,
            "回收率": rec,
            # 这批数据只有 3 年期、5 年期两种贷款，隐含阈值按期限各列一列（中位数在该期限内取）
            "隐含阈值 3 年": float(np.median(p_star[j][T == 3])),
            "隐含阈值 5 年": float(np.median(p_star[j][T == 5])),
            "拒绝率": float(reject.mean()),
            "拦截率": float(y[reject].sum() / y.sum()),
            "被拒者精确率": float(y[reject].mean()) if reject.any() else np.nan,
        })
sens = pd.DataFrame(rows)
sens.to_csv(OUT / "expected_loss_sensitivity.csv", index=False)
print()
print("=" * 62)
print("敏感性：回收率 × 收益口径（概率均为校准后）")
print("=" * 62)
print(sens.round(4).to_string(index=False))

# ---------- 图：拒绝率 / 拦截率 随回收率假设的变化 ----------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
colors = ["#1f6fb2", "#3f9a6f", "#d18b2a", "#b0483f"]
for (label, v), c in zip(calibers, colors):
    rej, cap = [], []
    for rec in recs:
        approve = unit_ep(p_cal, v, rec) > 0
        rej.append((~approve).mean() * 100)
        cap.append(y[~approve].sum() / y.sum() * 100)
    axes[0].plot([r * 100 for r in recs], rej, "o-", color=c, label=label)
    axes[1].plot([r * 100 for r in recs], cap, "o-", color=c, label=label)

for ax, title, ylab in [
    (axes[0], "Rejection rate vs recovery assumption", "Rejection rate (%)"),
    (axes[1], "Default capture rate vs recovery assumption", "Defaults captured (%)"),
]:
    ax.set_xlabel("Recovery rate assumption (%)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(OUT / "expected_loss_sensitivity.png", dpi=150)
plt.close()
print()
print(f"图已保存: {OUT / 'expected_loss_sensitivity.png'}")

# ---------- 一句话结论（写报告用） ----------
base = summary[summary["策略"] == "期望损失框架 · 中性（净息差 6%）"].iloc[0]
opt = summary[summary["策略"] == "期望损失框架 · 乐观（利息全额）"].iloc[0]
con = summary[summary["策略"] == "期望损失框架 · 保守（净息差 3%）"].iloc[0]
thr = summary[summary["策略"] == "固定阈值 0.5（未校准概率）"].iloc[0]
allf = summary[summary["策略"] == "全放（不筛选）"].iloc[0]
rawf = summary[summary["策略"] == "期望损失框架（未校准概率）"].iloc[0]
print()
print("=" * 62)
print("一句话结论（写报告用）")
print("=" * 62)
print(
    f"中性情景（净息差 6%、回收率 30%）下：期望损失框架拒绝 {base['被拒占比']*100:.1f}% 的申请、"
    f"拦截 {base['拦截率(真违约被拒占比)']*100:.1f}% 的违约者"
    f"（被拒者中 {base['被拒者精确率(里面真违约占比)']*100:.1f}% 是真违约），"
    f"总期望利润 {base['总期望利润(万元)']:,.0f} 万元；"
    f"收益口径是最大敏感源：同样的框架，乐观情景（利息全额）只拒 {opt['被拒占比']*100:.1f}%、"
    f"保守情景（净息差 3%）要拒 {con['被拒占比']*100:.1f}%；"
    f"固定阈值 0.5 拒绝 {thr['被拒占比']*100:.1f}%、拦截 {thr['拦截率(真违约被拒占比)']*100:.1f}%、"
    f"利润 {thr['总期望利润(万元)']:,.0f} 万元（全放为 {allf['总期望利润(万元)']:,.0f} 万元）；"
    f"若不校准概率直接套框架，拒绝率会虚高到 {rawf['被拒占比']*100:.1f}%。"
)
