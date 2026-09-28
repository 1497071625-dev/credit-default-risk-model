# -*- coding: utf-8 -*-
"""
review_band.py —— 人工复审带（W4 补充分析）

为什么要有这个脚本
    拆线方案把每个人判成"放"或"拒"，现实中还有一种中间处理：人工复审。
    但人工是稀缺资源，不能所有人都送审。最该送审的是**卡在审批线附近**的人：
    模型对他们期望利润算出来接近 0，放与拒的代价都小，判错的损失也小；
    而离审批线很远的人（要么明显该拒、要么明显该放），送审是浪费人力。
    本脚本以拆线为基线，把靠线的被拒者切出来做为"人工复审带"，给出：
      三段分工（直接放行 / 人工复审 / 直接拒绝）的人数、占比、实际违约率；
      复审带的钱账：全放会亏多少（模型口径）、里面正常人全放能赚多少息差（事后口径）。

复审带定义
    拆线的两条线（校准概率轴：3 年期 0.2045、5 年期 0.3000）。
    线上方 20% 以内（p < 1.2 × p*）进复审带；更上面的直接拒。
    带宽 20% 是可调的政策参数，不是模型算出来的。

输入
    outputs/W4/test_predictions_calibrated.csv（prob_cal、y_true、loanAmnt、term）
输出
    outputs/W4/review_band.csv      三段分工表
    outputs/W4/review_band_key.csv  复审带关键数字
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "outputs").is_dir()),
    Path(__file__).resolve().parents[2],
)
W4 = BASE_DIR / "outputs" / "W4"

V_NEUTRAL, REC_BASE, BAND_W = 0.06, 0.3, 0.20
cal = pd.read_csv(W4 / "test_predictions_calibrated.csv")
p = cal["prob_cal"].to_numpy(float)
y = cal["y_true"].to_numpy(int)
loan = cal["loanAmnt"].to_numpy(float)
term = cal["term"].to_numpy(float)
n = len(y)

# 每期限的审批线（与 ④ 页同一条：期望利润 = 0 解出来的 p*）
band = np.zeros(n, bool)
far = np.zeros(n, bool)
for t in sorted(set(term)):
    ps = V_NEUTRAL * t / (V_NEUTRAL * t + 1 - REC_BASE)
    m = term == t
    band |= m & (p >= ps) & (p < ps * (1 + BAND_W))
    far |= m & (p >= ps * (1 + BAND_W))
approve = ~(band | far)
assert int(approve.sum() + band.sum() + far.sum()) == n

ep = (1 - p) * V_NEUTRAL * term * loan - p * (1 - REC_BASE) * loan   # 单位期望利润（元）

rows = []
for name, m, note in [
    ("直接放行", approve, "线以下，模型认为期望利润为正"),
    ("人工复审", band, f"线上方 {BAND_W:.0%} 以内，模型拿不准"),
    ("直接拒绝", far, f"线上方 {BAND_W:.0%} 以外，模型认为明显该拒"),
]:
    rows.append({"分工": name, "人数": int(m.sum()), "占测试集": float(m.mean()),
                 "实际违约率": float(y[m].mean()), "说明": note})
band_tbl = pd.DataFrame(rows)
band_tbl.to_csv(W4 / "review_band.csv", index=False)

good = band & (y == 0)
bad = band & (y == 1)
key = pd.DataFrame([
    {"项目": "复审带人数", "值": int(band.sum())},
    {"项目": "复审带占被拒比例", "值": float(band.sum() / (band | far).sum())},
    {"项目": "复审带内正常人", "值": int(good.sum())},
    {"项目": "复审带内违约者", "值": int(bad.sum())},
    {"项目": "复审带全放期望利润(万元)", "值": float(ep[band].sum() / 1e4)},
    {"项目": "复审带正常人全放息差(万元)", "值": float((V_NEUTRAL * term * loan)[good].sum() / 1e4)},
    {"项目": "复审带人均息差(元)", "值": float((V_NEUTRAL * term * loan)[good].mean())},
])
key.to_csv(W4 / "review_band_key.csv", index=False)

# 带宽敏感性：说明 20% 是政策选择
for w in [0.10, 0.15, 0.30]:
    bb = np.zeros(n, bool)
    for t in sorted(set(term)):
        ps = V_NEUTRAL * t / (V_NEUTRAL * t + 1 - REC_BASE)
        m = term == t
        bb |= m & (p >= ps) & (p < ps * (1 + w))
    print(f"  带宽 {w:.0%}: 复审 {int(bb.sum()):,} 人，违约率 {y[bb].mean()*100:.1f}%")

print(band_tbl.to_string(index=False))
print(key.to_string(index=False))
print("已保存 review_band.csv、review_band_key.csv")
