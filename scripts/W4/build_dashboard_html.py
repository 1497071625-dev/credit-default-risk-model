# -*- coding: utf-8 -*-
# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
build_dashboard_html.py —— 生成"自包含"业务仪表板 HTML（W4 交付物 · 可视化成果）

为什么要有这个脚本
    Streamlit 版仪表板（app_dashboard.py）需要对方装 Python、跑命令才能看，不适合交付。
    本脚本把同样的内容导出成**一个 HTML 文件**：双击就用浏览器打开，不装环境、不联网也能看；
    阈值滑块可实时联动（数据在生成时已预计算好），Ctrl/⌘+P 还能直接存成 PDF。

输入（全部是已有产物，不重新训练）
    outputs/W4/test_predictions.csv            逐人预测概率（只用于预计算阈值网格，不写进 HTML）
    outputs/W4/{model_final_metrics,generalization_compare,threshold_table,
                calibration_compare,calibration_bins,expected_loss_summary,
                expected_loss_sensitivity,feature_count_auc}.csv
    outputs/W3/model_comparison_optimized.csv
    outputs/{W1,W2,W3,W4}/*.png               各阶段配图（base64 内嵌，保证单文件）

输出
    outputs/W4/dashboard.html                  约 2–3 MB，单个文件即可交付

运行
    .venv/bin/python scripts/W4/build_dashboard_html.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import svg_charts as SC  # noqa: E402  内联 SVG 图渲染（服务端生成，前端只做悬浮读数）

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "outputs").is_dir()),
    Path(__file__).resolve().parents[2],
)
OUT = BASE_DIR / "outputs"
W4 = OUT / "W4"
DST = W4 / "dashboard.html"

# ----------------------------------------------------------------------------
# 1) 阈值网格：把逐人概率算成 19 个阈值的汇总指标（只把汇总写进 HTML）
# ----------------------------------------------------------------------------
pred = pd.read_csv(W4 / "test_predictions.csv")
y = pred["y_true"].to_numpy()
prob = pred["prob_default"].to_numpy()
N = int(len(y))
N_POS = int(y.sum())
N_NEG = N - N_POS

# 金额口径（与 expected_loss_framework.py / threshold_sweep.py 完全一致）：
#   逐笔单位期望利润 = (1 - p_cal) * 净息差 * 期限 - p_cal * (1 - 回收率)
#   概率必须用【校准后】的值，否则金额算出来是错的
cal = pd.read_csv(W4 / "test_predictions_calibrated.csv")
assert len(cal) == N and bool((cal["prob_raw"].to_numpy() == prob).all()), \
    "test_predictions_calibrated.csv 与 test_predictions.csv 的行序/概率不一致"
p_cal = cal["prob_cal"].to_numpy(float)
loan = cal["loanAmnt"].to_numpy(float)
term = cal["term"].to_numpy(float)
V_NEUTRAL, REC_BASE = 0.06, 0.3
unit_ep = (1 - p_cal) * V_NEUTRAL * term - p_cal * (1 - REC_BASE)

grid = []
for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
    reject = prob >= t
    tp = int((reject & (y == 1)).sum())
    fp = int((reject & (y == 0)).sum())
    fn = int((~reject & (y == 1)).sum())
    tn = int((~reject & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    approve = ~reject
    # 金额明细：把"回收率/净息差"这两个业务参数从金额里提出来，
    # 这样 HTML 里拖动这两个滑块时不必重算原始数据，只需一次乘加。
    #   少放贷款   = Σ L            （被拒者的贷款金额合计）
    #   避免坏账   = Σ L_bad   × (1 − 回收率)
    #   放弃收入   = Σ L×T_good × 净息差
    #   期望利润   = 净息差 × Σ(1−p_cal)·T·L − (1−回收率) × Σ p_cal·L   （只对批准的人求和）
    agg = dict(
        loans_rej=float(loan[reject].sum() / 1e4),
        avoided_base=float(loan[reject & (y == 1)].sum() / 1e4),
        foregone_base=float((loan * term)[reject & (y == 0)].sum() / 1e4),
        ep_pos=float(((1 - p_cal) * term * loan)[approve].sum() / 1e4),
        ep_neg=float((p_cal * loan)[approve].sum() / 1e4),
    )
    profit = float(V_NEUTRAL * agg["ep_pos"] - (1 - REC_BASE) * agg["ep_neg"])
    gain = float((1 - REC_BASE) * agg["avoided_base"] - V_NEUTRAL * agg["foregone_base"])
    assert abs(profit - float((unit_ep[approve] * loan[approve]).sum() / 1e4)) < 1e-6, "期望利润分解式与直算式不一致"
    assert abs(gain - float(((loan[reject & (y == 1)] * (1 - REC_BASE)).sum()
                             - (loan[reject & (y == 0)] * V_NEUTRAL * term[reject & (y == 0)]).sum()) / 1e4)) < 1e-6, \
        "净增益分解式与直算式不一致"
    grid.append(dict(t=float(t), tp=tp, fp=fp, fn=fn, tn=tn,
                     reject_rate=(tp + fp) / N, precision=prec, recall=rec, f1=f1,
                     profit=profit, gain=gain, **agg))

_fm = pd.read_csv(W4 / "model_final_metrics.csv").iloc[0]
_g5 = next(g for g in grid if abs(g["t"] - 0.5) < 1e-9)
assert abs(_g5["precision"] - _fm["precision"]) < 1e-6, "阈值 0.5 的精确率与 model_final_metrics.csv 不一致"
assert abs(_g5["recall"] - _fm["recall"]) < 1e-6, "阈值 0.5 的召回率与 model_final_metrics.csv 不一致"

# 金额一致性：0.5 档的期望利润必须与已交付的 expected_loss_summary.csv 对得上
_el = pd.read_csv(W4 / "expected_loss_summary.csv").set_index("策略")["总期望利润(万元)"]
_el05 = float(_el["固定阈值 0.5（未校准概率）"])
assert abs(_g5["profit"] - _el05) < 0.5, \
    f"阈值 0.5 的期望利润 {_g5['profit']:.1f} 万元与 expected_loss_summary.csv 的 {_el05:.1f} 不一致"

# ----------------------------------------------------------------------------
# 1b) 默认档（阈值 0.50）预渲染：图与数字直接写进 HTML，
#     即使浏览器禁用了 JavaScript 也能看到完整默认状态（滑块除外）
# ----------------------------------------------------------------------------
VB = dict(w=540, h=225, l=46, r=12, t=14, b=40)
_ix = lambda t: VB["l"] + (t - 0.05) / 0.90 * (VB["w"] - VB["l"] - VB["r"])
_iy = lambda v: VB["t"] + (1 - v) * (VB["h"] - VB["t"] - VB["b"])
def _pts(key: str) -> str:
    return " ".join(f"{_ix(g['t']):.1f},{_iy(g[key]):.1f}" for g in grid)

_wan = lambda v: f"{v:,.0f}"
DEF = {
    "PT_PREC": _pts("precision"),
    "PT_REC": _pts("recall"),
    "PT_F1": _pts("f1"),
    "CIRCLES": "".join(f'<circle cx="{_ix(g["t"]):.1f}" cy="{_iy(g["precision"]):.1f}" '
                       f'r="2.8" fill="#1f5fbf"/>' for g in grid),
    "MX": f'{_ix(_g5["t"]):.1f}',
    "TIP": (f'阈值 {_g5["t"]:.2f}｜精确率 {_g5["precision"] * 100:.1f}%｜'
            f'召回率 {_g5["recall"] * 100:.1f}%｜F1 {_g5["f1"]:.3f}'),
    "K_REJECT": f'{_g5["reject_rate"] * 100:.1f}%',
    "K_REJECT_N": f'{_g5["tp"] + _g5["fp"]:,} 人被拒',
    "K_PREC": f'{_g5["precision"] * 100:.1f}%',
    "K_REC": f'{_g5["recall"] * 100:.1f}%',
    "K_REC_N": f'抓到 {_g5["tp"]:,} / {N_POS:,} 名违约者',
    "K_F1": f'{_g5["f1"]:.3f}',
    "K_AUC": f'{float(_fm["auc"]):.4f}',
    "K_GAIN": f'{_wan(_g5["gain"])} 万',
    "CM_TN": f'{_g5["tn"]:,}', "CM_FP": f'{_g5["fp"]:,}',
    "CM_FN": f'{_g5["fn"]:,}', "CM_TP": f'{_g5["tp"]:,}',
    "LEDGER": (f'按阈值 <b>{_g5["t"]:.2f}</b> 审批：拒绝 <b>{_g5["reject_rate"] * 100:.1f}%</b> '
               f'的申请人（{_g5["tp"] + _g5["fp"]:,} 人），其中真违约 <b>{_g5["tp"]:,}</b> 人、'
               f'误伤正常客户 <b>{_g5["fp"]:,}</b> 人；全部 {N_POS:,} 名违约者中拦下 '
               f'<b>{_g5["recall"] * 100:.1f}%</b>，漏放 {_g5["fn"]:,} 人；'
               f'这批决策的净增益 <b>{_wan(_g5["gain"])} 万元</b>'
               f'（相对"全批放款"的增量，挑线就看它）。'),
}

# ---- 金额（"设阀门 → 看赚多少/亏多少"）的默认档预渲染 ----
DEF["K_LOANS"] = f'{_wan(_g5["loans_rej"])} 万'
DEF["K_AVOIDED"] = f'{_wan((1 - REC_BASE) * _g5["avoided_base"])} 万'
DEF["K_FOREGONE"] = f'{_wan(V_NEUTRAL * _g5["foregone_base"])} 万'
DEF["REC_PCT"] = f'{REC_BASE * 100:.0f}%'
DEF["MAR_PCT"] = f'{V_NEUTRAL * 100:.0f}%'

# ---- "各阈值下的净增益曲线"：Python 侧与 JS 侧用同一套公式，先把默认状态画好 ----
MG = dict(w=540, h=196, l=50, r=12, t=14, b=38)
_mgains = [g["gain"] for g in grid]
_mhi = max(_mgains)
_mlo = min(_mgains)
_mpad = (_mhi - _mlo) * 0.12 or 1.0
_mhi += _mpad
_mlo = min(_mlo - _mpad, 0.0)          # 保证 0 线在画面内
_gx = lambda t: MG["l"] + (t - 0.05) / 0.90 * (MG["w"] - MG["l"] - MG["r"])
_gy = lambda v: MG["t"] + (1 - (v - _mlo) / (_mhi - _mlo)) * (MG["h"] - MG["t"] - MG["b"])
_best_t = grid[int(np.argmax(_mgains))]["t"]

def _gain_group(ptrs: dict) -> str:
    """把金额曲线需要的东西一次性按当前参数画出来（Python 预渲染用）。"""
    out = []
    for k in range(5):
        v = _mlo + (_mhi - _mlo) * k / 4
        y = _gy(v)
        out.append(f'<line x1="{MG["l"]}" y1="{y:.1f}" x2="{MG["w"] - MG["r"]}" y2="{y:.1f}" stroke="#e8ecf1"/>')
        out.append(f'<text x="{MG["l"] - 8}" y="{y + 4:.1f}" font-size="11" fill="#5b6470" '
                   f'text-anchor="end">{v:,.0f}</text>')
    out.append(f'<line x1="{MG["l"]}" y1="{_gy(0):.1f}" x2="{MG["w"] - MG["r"]}" y2="{_gy(0):.1f}" '
               f'stroke="#9aa6b4" stroke-dasharray="4 4"/>')
    out.append(f'<text x="{MG["w"] - MG["r"]}" y="{_gy(0) - 6:.1f}" font-size="11" fill="#5b6470" '
               f'text-anchor="end">0（不赚不亏）</text>')
    for i, g in enumerate(grid):
        if i % 2:
            continue
        x = _gx(g["t"])
        out.append(f'<line x1="{x:.1f}" y1="{MG["t"]}" x2="{x:.1f}" y2="{MG["h"] - MG["b"]}" stroke="#f1f4f7"/>')
        out.append(f'<text x="{x:.1f}" y="{MG["h"] - MG["b"] + 16}" font-size="11" fill="#5b6470" '
                   f'text-anchor="middle">{g["t"]:.2f}</text>')
    out.append(f'<text x="{(MG["l"] + MG["w"] - MG["r"]) / 2:.0f}" y="{MG["h"] - 6}" font-size="12" '
               f'fill="#5b6470" text-anchor="middle">审批阈值（模型报的概率 ≥ 阈值即拒）</text>')
    return "".join(out)

DEF["G_GRID"] = _gain_group({})
DEF["G_POINTS"] = " ".join(f"{_gx(g['t']):.1f},{_gy(g['gain']):.1f}" for g in grid)
DEF["G_PTS"] = "".join(f'<circle cx="{_gx(g["t"]):.1f}" cy="{_gy(g["gain"]):.1f}" r="2.8" fill="#1e8449"/>'
                       for g in grid)
DEF["G_MX"] = f'{_gx(_g5["t"]):.1f}'
DEF["G_BEST_X"] = f'{_gx(_best_t):.1f}'
DEF["G_BEST_Y"] = f'{_gy(dict(zip([g["t"] for g in grid], _mgains))[_best_t]):.1f}'

# 建议框的默认文案（与 JS 里的 adviceText 同结构；禁 JS 时也能看到结论）
_best_g = grid[int(np.argmax(_mgains))]
_tone5 = ("宽进档：抓得全，但误伤多" if _g5["t"] <= 0.35 else
          "严审档：批得准，但放走的多" if _g5["t"] > 0.65 else
          "平衡档：日常风控常规档")

# ----------------------------------------------------------------------------
# 1c) 页签⑤ 的二维网格：回收率 × 净息差 → 逐笔框架 vs 最优单一线
#     这里两条线用完全相同的口径（既定放谁、又用来算钱），所以滑块拖到哪都对得上。
#     有了网格，HTML 侧只需查表，不必重算 8 万行。
# ----------------------------------------------------------------------------
_srt = np.argsort(prob, kind="stable")
_pra_s, _pc_s = prob[_srt], p_cal[_srt]
_L_s, _T_s = loan[_srt] / 1e4, term[_srt]
_CS = np.concatenate([[0.0], np.cumsum((1 - _pc_s) * _T_s * _L_s)])   # 按原始概率排序的 Σ(1−p)·期限·金额
_DS = np.concatenate([[0.0], np.cumsum(_pc_s * _L_s)])               # 同上的 Σ p·金额
_TGRID = np.round(np.arange(0.001, 1.0, 0.001), 4)
_J = np.searchsorted(_pra_s, _TGRID)

ERECS = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]
EMARS = [round(x, 4) for x in np.arange(0.0, 0.2001, 0.01)]
_ep_m, _th_m, _t_m = [], [], []
for _rec in ERECS:
    _er, _tr, _tt = [], [], []
    for _v in EMARS:
        _epv = (1 - p_cal) * _v * term - p_cal * (1 - _rec)
        _er.append(float((_epv[_epv > 0] * loan[_epv > 0]).sum() / 1e4))
        _prof = _v * _CS[_J] - (1 - _rec) * _DS[_J]
        _bi = int(np.argmax(_prof))
        _tr.append(float(_prof[_bi]))
        _tt.append(float(_TGRID[_bi]))
    _ep_m.append(_er), _th_m.append(_tr), _t_m.append(_tt)

EPV = {"recs": ERECS, "mars": EMARS, "ep": _ep_m, "th": _th_m, "t": _t_m,
       "ri0": ERECS.index(0.30), "mi0": EMARS.index(0.06)}
_EPV_EP, _EPV_TH = _ep_m[EPV["ri0"]][EPV["mi0"]], _th_m[EPV["ri0"]][EPV["mi0"]]
assert abs(_EPV_EP - float(_el["期望损失框架 · 中性（净息差 6%）"])) < 1.0, \
    f"页签⑤ 中性档 {_EPV_EP:.1f} 与 expected_loss_summary.csv 不一致"
# 单一线最优要对齐 0.001 步长的细扫描（threshold_sweep.csv）；
# 页签③ 那张表只有 0.05 步长的 19 档，最优点会略低（6,340.5 vs 6,344.0），属网格粗化所致。
_sweep = pd.read_csv(W4 / "threshold_sweep.csv")
_sweep_best = float(_sweep["总期望利润(万元)"].max())
assert abs(_EPV_TH - _sweep_best) < 1.0, \
    f"页签⑤ 单一线最优 {_EPV_TH:.1f} 与 threshold_sweep.csv 的 {_sweep_best:.1f} 不一致"
assert abs(_best_g["profit"] - _EPV_TH) < 10, \
    f"页签③ 粗网格最优 {_best_g['profit']:.1f} 与细扫描 {_EPV_TH:.1f} 差得过多"

# ---- 页签⑤ 的核心业务结论：⑤ 的隐含阈值按期限各自一条，③ 只有一个数 ----
# ⑤ 每笔贷款的审批线（校准概率轴）：p* = V·T / (V·T + 1 − rec)
_T55 = float(_best_g["t"])                       # 页签③ 选出的原始概率轴阈值
_sweep_raw = _sweep[_sweep["轴"].astype(str).str.startswith("原始")]
_bi_f = int(_sweep_raw["总期望利润(万元)"].idxmax())
_FINE_T = float(_sweep_raw.loc[_bi_f, "阈值"])
_FINE_M = float(_sweep_raw.loc[_bi_f, "总期望利润(万元)"])
_T55_FINE = float(_sweep_raw.loc[np.isclose(_sweep_raw["阈值"], _T55), "总期望利润(万元)"].iloc[0])
_FINE_DIFF = _FINE_M - _T55_FINE
assert 0 < _FINE_DIFF < 50, f"细扫与 0.55 档差值异常：{_FINE_DIFF:.1f} 万元"

# ③ 页只讲净增益一种口径（与 grid 里的 gain 同公式），细扫最优点两套口径落在同一个 t 上
def _gain_at(t: float) -> float:
    r = prob >= t
    return float(((1 - REC_BASE) * loan[r & (y == 1)]).sum()
                 - (V_NEUTRAL * term[r & (y == 0)] * loan[r & (y == 0)]).sum()) / 1e4

_FINE_GAIN = _gain_at(_FINE_T)
_FINE_GAIN_DIFF = _FINE_GAIN - _gain_at(_T55)
assert abs(_gain_at(_T55) - _best_g["gain"]) < 1e-6, "细扫净增益与网格最优档的净增益不一致"
assert 0 < _FINE_GAIN_DIFF < 50, f"细扫净增益与最优档差值异常：{_FINE_GAIN_DIFF:.1f} 万元"

# 两把刻度的换算：校准是单调变换，不改变排序，最优线是同一批人，只是数字不同
_C55_FINE = float(p_cal[prob >= _FINE_T].min())          # 原始轴 _FINE_T 对应到校准轴上的那根线
_N_FINE_REJ = int((prob >= _FINE_T).sum())
_REJ_XAXIS = float((p_cal >= _FINE_T).mean() * 100)      # 把原始轴的数直接搬到校准轴上用的后果
assert abs(_C55_FINE - 0.2406) < 0.003, f"原始轴最优换算到校准轴异常：{_C55_FINE:.4f}"
assert abs(_REJ_XAXIS - 1.7) < 0.3, f"跨轴误用后的拒绝率异常：{_REJ_XAXIS:.1f}%"

# 换一批数据还站得住吗：阈值改到验证集上挑（threshold_select_on_val.py），再搬回测试集算净增益
_val_curve = pd.read_csv(W4 / "threshold_select_on_val.csv")
_VAL_T = float(_val_curve.loc[_val_curve["总期望利润(万元)"].idxmax(), "阈值"])
_VAL_GAIN = _gain_at(_VAL_T)
_VAL_GAP = _FINE_GAIN - _VAL_GAIN
_VAL_GAP_PCT = _VAL_GAP / _FINE_GAIN * 100
_PLAT_LOW_PCT = min(_gain_at(float(_q)) for _q in np.round(np.arange(0.54, 0.5801, 0.001), 3)) / _FINE_GAIN * 100
assert 0.54 < _VAL_T < 0.58, f"验证集重挑的阈值跑出平台区：{_VAL_T}"
assert 0 < _VAL_GAP < 40, f"验证集选线搬到测试集的净增益差异常：{_VAL_GAP:.1f} 万元"
assert _PLAT_LOW_PCT > 95, f"平台区下限异常：{_PLAT_LOW_PCT:.1f}%"
_n55 = int((prob >= _T55).sum())
_pc_desc = np.sort(p_cal)[::-1]
_C55 = float(_pc_desc[_n55 - 1])                 # 换算到校准轴上的等价线（一个数，不分期限）
TERM_ROWS = []
for _t in sorted(set(term)):
    _m = term == _t
    _ps = V_NEUTRAL * _t / (V_NEUTRAL * _t + 1 - REC_BASE)
    _r3, _r5 = prob[_m] >= _T55, p_cal[_m] >= _ps
    TERM_ROWS.append({
        "期限": f"{_t:.0f} 年",
        "人数": f"{int(_m.sum()):,}",
        "实际违约率": f"{y[_m].mean() * 100:.1f}%",
        "模型报高倍数": f"{prob[_m].mean() / y[_m].mean():.2f} 倍",
        "逐笔想要的线": f"{_ps:.4f}",
        "③ 给的线": f"{_C55:.4f}",
        "③ 拒多少": f"{_r3.mean() * 100:.1f}%",
        "逐笔该拒多少": f"{_r5.mean() * 100:.1f}%",
    })
_ps_arr = np.where(term == 5.0, V_NEUTRAL * 5 / (V_NEUTRAL * 5 + 1 - REC_BASE),
                   V_NEUTRAL * 3 / (V_NEUTRAL * 3 + 1 - REC_BASE))
_r3a, _r5a = prob >= _T55, p_cal >= _ps_arr
DIFF_N = int((_r3a != _r5a).sum())
DIFF_PCT = float((_r3a != _r5a).mean() * 100)
_g3, _g5r = TERM_ROWS[0], TERM_ROWS[-1]        # 3 年 / 5 年两行
TERM_NOTE = (
    f'3 年期该拒 <b>{_g3["逐笔该拒多少"]}</b>，③ 只拒了 <b>{_g3["③ 拒多少"]}</b>，放进来太多；'
    f'5 年期拒 <b>{_g5r["逐笔该拒多少"]}</b> 就够，③ 却拒了 <b>{_g5r["③ 拒多少"]}</b>。'
    f'两套规则对 <b>{DIFF_N:,} 人（{DIFF_PCT:.1f}%）</b>的判定不一致。')
DEF["TERM_CONC"] = TERM_NOTE

# ---- ⑤ 能落地的形式：把 ③ 的一条线拆成"按期限两条线"，看这笔钱到底多大 ----
_ps_map = {_t: V_NEUTRAL * _t / (V_NEUTRAL * _t + 1 - REC_BASE) for _t in sorted(set(term))}
_rej_two = np.zeros(N, bool)
RAW_LINES = {}
for _t, _ps in _ps_map.items():
    _m = term == _t
    _r = _m & (p_cal >= _ps)
    _rej_two |= _r
    RAW_LINES[_t] = float(prob[_r].min())          # 同一批人在原始轴上的等价线（Platt 单调，名单逐行相同）
TWO_M = float((((1 - p_cal) * V_NEUTRAL * term - p_cal * (1 - REC_BASE)) * loan)[~_rej_two].sum() / 1e4)
TWO_REJ_N = int(_rej_two.sum())
TWO_GAIN = TWO_M - _EPV_TH
assert abs(TWO_M - _EPV_EP) < 1.0, \
    f"按期限两条线 {TWO_M:.1f} 应等于逐笔上限 {_EPV_EP:.1f}（中性口径下逐笔规则只按期限分叉）"
assert abs(TWO_GAIN - (_EPV_EP - _EPV_TH)) < 1.0, "两条线相对单一线的差额与 ⑤ 相对 ③ 的差额不一致"
RAW3, RAW5 = RAW_LINES[sorted(RAW_LINES)[0]], RAW_LINES[sorted(RAW_LINES)[-1]]

# ---- 页头建议段用的两个参照：全放不筛人、③ 页签默认的 0.50 ----
_BASE_M = float((unit_ep * loan).sum() / 1e4)
assert abs(_BASE_M - float(_el["全放（不筛选）"])) < 0.5, "全放基准与 expected_loss_summary.csv 不一致"
_G50_N = int(_g5["tp"] + _g5["fp"])
_G50_M = float(_g5["profit"])
_D1 = TWO_M - _G50_M                      # 拆线比 0.50 单线多赚
_D2 = _EPV_TH - _G50_M                    # 最优单一线比 0.50 单线多赚
assert 400 < _D1 < 500 and 150 < _D2 < 250, f"参照差额异常：{_D1:.0f} / {_D2:.0f}"
_RT3 = float(cal.loc[cal["term"] == 3, "interestRate"].mean())
_RT5 = float(cal.loc[cal["term"] == 5, "interestRate"].mean())

# ② 相对 ③ 多出来的那笔钱是谁贡献的：两边各换了一批人（按期限各自解线 vs 一条线管所有人）
_psta = np.where(term == 5.0, V_NEUTRAL * 5 / (V_NEUTRAL * 5 + 1 - REC_BASE),
                 V_NEUTRAL * 3 / (V_NEUTRAL * 3 + 1 - REC_BASE))
_acc3, _acc4 = prob < _FINE_T, p_cal < _psta
_sw_in, _sw_out = _acc4 & ~_acc3, _acc3 & ~_acc4
_SW_IN_N, _SW_IN_M = int(_sw_in.sum()), float((unit_ep * loan)[_sw_in].sum() / 1e4)
_SW_OUT_N, _SW_OUT_M = int(_sw_out.sum()), float(-(unit_ep * loan)[_sw_out].sum() / 1e4)
_SW_IN_RT = float(cal.loc[_sw_in, "interestRate"].mean())
_SW_OUT_RT = float(cal.loc[_sw_out, "interestRate"].mean())
assert abs((_SW_IN_M + _SW_OUT_M) - TWO_GAIN) < 1.0, \
    f"换人拆分与总差额对不上：{_SW_IN_M:.1f} + {_SW_OUT_M:.1f} vs {TWO_GAIN:.1f}"
assert 11 < _RT3 < 13 and 16 < _RT5 < 18, f"分期限平均利率异常：{_RT3:.2f} / {_RT5:.2f}"

# ---- 页签⑤ 交互块的默认档（回收率 30%、净息差 6%），禁 JS 时也能看到完整图与数 ----
EGB = dict(w=900, h=260, l=80, r=28, t=16, b=42)
_egx = lambda v: EGB["l"] + v / 0.20 * (EGB["w"] - EGB["l"] - EGB["r"])
_ri, _mi = EPV["ri0"], EPV["mi0"]
_eep, _eth = _ep_m[_ri], _th_m[_ri]
_GAP3 = float(_eep[EMARS.index(0.03)] - _eth[EMARS.index(0.03)])   # 净息差 3% 时 逐笔与单线的差距

# 把差额换算成业务能比成本的规模口径：每 100 亿元年放款额、每年多赚多少
_BOOK_M = float(loan.sum() / 1e4)                                  # 测试集放款额合计（万元）
_GAIN100 = TWO_GAIN / _BOOK_M * 1e6                                # 中性口径：每 100 亿年放款额的年度增益（万元）
_GAIN100C = _GAP3 / _BOOK_M * 1e6                                  # 净息差 3% 口径
assert 2000 < _GAIN100 < 2600, f"每 100 亿年放款的增益异常：{_GAIN100:,.0f} 万元"
assert 650 < _GAIN100C < 950, f"保守口径的每 100 亿增益异常：{_GAIN100C:,.0f} 万元"
assert 80 < _GAP3 < 100, f"净息差 3% 档差距异常：{_GAP3:.0f} 万元"
_ehi = max(_eep + _eth) * 1.10
_elo = min(0.0, min(_eep + _eth))
_egy = lambda v: EGB["t"] + (1 - (v - _elo) / (_ehi - _elo)) * (EGB["h"] - EGB["t"] - EGB["b"])

_eg = []
for _k in range(5):
    _v = _elo + (_ehi - _elo) * _k / 4
    _y = _egy(_v)
    _eg.append(f'<line x1="{EGB["l"]}" y1="{_y:.1f}" x2="{EGB["w"] - EGB["r"]}" y2="{_y:.1f}" stroke="#e8ecf1"/>')
    _eg.append(f'<text x="{EGB["l"] - 8}" y="{_y + 4:.1f}" font-size="11" fill="#5b6470" '
               f'text-anchor="end">{_v:,.0f}</text>')
_eg.append(f'<line x1="{EGB["l"]}" y1="{_egy(0):.1f}" x2="{EGB["w"] - EGB["r"]}" y2="{_egy(0):.1f}" '
           f'stroke="#9aa6b4" stroke-dasharray="4 4"/>')
for _j, _v in enumerate(EMARS):
    if _j % 2:
        continue
    _x = _egx(_v)
    _eg.append(f'<line x1="{_x:.1f}" y1="{EGB["t"]}" x2="{_x:.1f}" y2="{EGB["h"] - EGB["b"]}" stroke="#f1f4f7"/>')
    _eg.append(f'<text x="{_x:.1f}" y="{EGB["h"] - EGB["b"] + 16}" font-size="11" fill="#5b6470" '
               f'text-anchor="middle">{_v * 100:.0f}%</text>')
_eg.append(f'<text x="{(EGB["l"] + EGB["w"] - EGB["r"]) / 2:.0f}" y="{EGB["h"] - 6}" font-size="12" '
           f'fill="#5b6470" text-anchor="middle">年净息差</text>')
_eg.append(f'<text x="16" y="{EGB["t"] + 8}" font-size="12" fill="#5b6470">收益（万元）</text>')

_epts = lambda a: " ".join(f"{_egx(_v):.1f},{_egy(a[_j]):.1f}" for _j, _v in enumerate(EMARS))
DEF["E_GRID"] = "".join(_eg)
DEF["E_EP"] = _epts(_eep)
DEF["E_TH"] = _epts(_eth)
DEF["E_MX"] = f'{_egx(EMARS[_mi]):.1f}'
DEF["E_EPV"] = f'{_eep[_mi]:,.0f}'
DEF["E_THV"] = f'{_eth[_mi]:,.0f}'
DEF["E_GAP"] = f'{_eep[_mi] - _eth[_mi]:,.0f}'
DEF["E_T"] = f'{_t_m[_ri][_mi]:.3f}'
GGB = dict(w=900, h=190, l=80, r=28, t=18, b=36)
_ggx = lambda v: GGB["l"] + v / 0.20 * (GGB["w"] - GGB["l"] - GGB["r"])
_gap_m = [_eep[_j] - _eth[_j] for _j in range(len(EMARS))]
_ghi = max(_gap_m) * 1.18 or 1.0
_ggy = lambda v: GGB["t"] + (1 - v / _ghi) * (GGB["h"] - GGB["t"] - GGB["b"])
_gg = []
for _k in range(4):
    _v = _ghi * _k / 3
    _y = _ggy(_v)
    _gg.append(f'<line x1="{GGB["l"]}" y1="{_y:.1f}" x2="{GGB["w"] - GGB["r"]}" y2="{_y:.1f}" stroke="#e8ecf1"/>')
    _gg.append(f'<text x="{GGB["l"] - 8}" y="{_y + 4:.1f}" font-size="11" fill="#5b6470" '
               f'text-anchor="end">{_v:,.0f}</text>')
for _j, _v in enumerate(EMARS):
    if _j % 2:
        continue
    _x = _ggx(_v)
    _gg.append(f'<line x1="{_x:.1f}" y1="{GGB["t"]}" x2="{_x:.1f}" y2="{GGB["h"] - GGB["b"]}" stroke="#f1f4f7"/>')
    _gg.append(f'<text x="{_x:.1f}" y="{GGB["h"] - GGB["b"] + 16}" font-size="11" fill="#5b6470" '
               f'text-anchor="middle">{_v * 100:.0f}%</text>')
_gg.append(f'<text x="{(GGB["l"] + GGB["w"] - GGB["r"]) / 2:.0f}" y="{GGB["h"] - 6}" font-size="12" '
           f'fill="#5b6470" text-anchor="middle">年净息差</text>')
_gg.append(f'<text x="16" y="{GGB["t"] + 8}" font-size="12" fill="#5b6470">逐笔 − 单一线（万元）</text>')
DEF["E_GAPGRID"] = "".join(_gg)
DEF["E_GAPPTS"] = " ".join(f"{_ggx(_v):.1f},{_ggy(_gap_m[_j]):.1f}" for _j, _v in enumerate(EMARS))
DEF["E_GAPMX"] = f'{_ggx(EMARS[_mi]):.1f}'
DEF["E_GAPMY1"] = f'{GGB["t"]}'
DEF["E_GAPMY2"] = f'{GGB["h"] - GGB["b"]}'
DEF["E_GAPLX"] = f'{_ggx(EMARS[_mi]) + 6:.1f}'
DEF["E_GAPLY"] = f'{max(GGB["t"] + 10, _ggy(_gap_m[_mi]) - 6):.1f}'
DEF["E_GAPLT"] = f'{_gap_m[_mi]:,.0f} 万元'
DEF["E_NOTE"] = (
    f'回收率 <b>{ERECS[_ri] * 100:.0f}%</b>、净息差 <b>{EMARS[_mi] * 100:.0f}%</b>：'
    f'逐笔算账 <b>{_wan(_eep[_mi])} 万元</b>，最优单一线 <b>{_wan(_eth[_mi])} 万元</b>'
    f'（t*={_t_m[_ri][_mi]:.3f}），逐笔多赚 <b>{_wan(_eep[_mi] - _eth[_mi])} 万元</b>。'
    f'净息差越薄，一条线管所有人丢得越多：净息差 3% 时两者分别是 '
    f'{_wan(_eep[EMARS.index(0.03)])} 与 {_wan(_eth[EMARS.index(0.03)])} 万元。')

DEF["ADVICE"] = (
    f'<span class="tone">{_tone5}</span>'
    '在"抓违约"和"误伤好客户"之间取折中，是日常风控的常规档位，也是这套模型（AUC 0.7249）下比较稳的区间。'
    f'<div class="pick"><b>只挑一条审批线，建议设 {_best_t:.2f}：</b>'
    f'拒绝 {_best_g["reject_rate"] * 100:.1f}% 的申请人，净增益 <b>{_wan(_best_g["gain"])} 万元</b>。'
    f'你现在拖到的（0.50）比它少赚 <b>{_wan(_best_g["gain"] - _g5["gain"])} 万元</b>。</div>')

DATA = {
    "n": N, "n_pos": N_POS, "n_neg": N_NEG,
    # 金额只把③的净增益口径发给前端（两个底数给滑块现算）；概率×金额的期望口径归页签④
    "grid": [{k: v for k, v in _g.items() if k not in ("profit", "ep_pos", "ep_neg")} for _g in grid],
    "metrics": {k: float(_fm[k]) for k in ["accuracy", "precision", "recall", "f1", "auc"]},
    "profit_all": round(float(_el["全放（不筛选）"]), 1),
    "epv": EPV,
}

TERM_TBL_HTML = ("<div class=\"scroll\"><table class=\"tbl\"><thead><tr>"
                  + "".join(f"<th>{c}</th>" for c in TERM_ROWS[0])
                  + "</tr></thead><tbody>"
                  + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r.values()) + "</tr>"
                            for r in TERM_ROWS)
                  + "</tbody></table></div>")

_CMP35 = [
    ("规则长什么样", "分数 ≥ __T55__ 就拒", "期望利润 ≤ 0 就拒"),
    ("用到的信息", "只有分数（排序）", "分数、金额、期限、利率、回收率"),
    ("线有几条", "单一线：全批人共用 1 条", "拆线：每个期限 1 条"),
    ("刻度落在哪", "原始概率轴 0.55", "校准概率轴 3 年期 0.2045、5 年期 0.3000，换成原始概率轴就是 0.50、0.62"),
    ("线是怎么来的", "把曲线扫一遍，取最高点", "从一笔账解出来"),
    ("明天来的客户", "照搬这条线", "照搬公式，需要金额、期限、利率"),
    ("换客群或换参数", "曲线最高点会移动，得在旧数据上重挑", "参数换掉重算即可"),
    ("概率刻度漂了", "还能用（只依赖排序）", "会崩（依赖概率数值）"),
    ("上限（中性口径的期望利润，最优点 t*=__FINE_T__）", "__TH_M__ 万元", "__EPV_EP__ 万元"),
    ("落地成本", "审批系统支持一个阈值即可", "按期限出两张评分表"),
]
CMP35_HTML = ("<div class=\"scroll\"><table class=\"tbl\"><thead><tr>"
              "<th>对比项</th><th>③ 单一线</th><th>④ 拆线（逐笔算账）</th></tr></thead><tbody>"
              + "".join(f"<tr><td>{_a}</td><td>{_b}</td><td>{_c}</td></tr>" for _a, _b, _c in _CMP35)
              + "</tbody></table></div>")

# ----------------------------------------------------------------------------
# 2) 表格渲染
# ----------------------------------------------------------------------------
def render_table(df: pd.DataFrame, headers: dict | None = None,
                 pct_cols: set[str] | None = None, int_cols: set[str] | None = None,
                 table_id: str | None = None, row_attrs=None) -> str:
    pct_cols = pct_cols or set()
    int_cols = int_cols or set()
    headers = headers or {}
    tid = f' id="{table_id}"' if table_id else ""
    parts = [f'<div class="scroll"><table class="tbl"{tid}><thead><tr>']
    for c in df.columns:
        parts.append(f"<th>{headers.get(c, c)}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        parts.append(f"<tr{row_attrs(row) if row_attrs else ''}>")
        for c in df.columns:
            v = row[c]
            if pd.isna(v):
                cell = '<span class="na">—</span>'
            elif c in pct_cols:
                cell = f"{float(v) * 100:.1f}%"
            elif c in int_cols:
                cell = f"{int(v):,}"
            elif isinstance(v, (int, float, np.integer, np.floating)):
                cell = f"{float(v):.3f}" if abs(float(v)) < 1000 else f"{float(v):,.0f}"
            else:
                cell = str(v)
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def figure(rel: str, caption: str, cls: str = "") -> str:
    blob = base64.b64encode((OUT / rel).read_bytes()).decode()
    return (f'<figure class="{cls}"><img src="data:image/png;base64,{blob}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


CHARTS = json.loads((W4 / "dashboard_charts.json").read_text(encoding="utf-8"))
def svg_figure(svg: str, caption: str, cls: str = "fig") -> str:
    """内联 SVG 图：每个数据点带 data-tip，鼠标悬浮即读数（无需外部图表库）"""
    return (f'<figure class="{cls}">{svg}'
            f'<figcaption>{caption}</figcaption></figure>')


# ----------------------------------------------------------------------------
# 1b) ①② 的内联 SVG 图：W1/W2/W3 的静态 PNG 换成可悬浮读数的矢量图
# ----------------------------------------------------------------------------
C = CHARTS
_pc = lambda v: f"{v * 100:.1f}%"          # noqa: E731
_cnt = lambda v: f"{v:,.0f}"               # noqa: E731

_t = C["target"]
FIG_TARGET = svg_figure(
    SC.chart_bars(
        [("正常 0", _t["normal"], f"正常 0：{_cnt(_t['normal'])} 笔，占 {_pc(_t['normal'] / _t['total'])}"),
         ("违约 1", _t["default"], f"违约 1：{_cnt(_t['default'])} 笔，占 {_pc(_t['default'] / _t['total'])}")],
        w=520, h=300, ylab="笔数", bar_color=lambda i: SC.BLUE if i == 0 else SC.RED),
    f"图 1｜目标变量分布：违约 {_pc(_t['default'] / _t['total'])}、正常 {_pc(_t['normal'] / _t['total'])}，"
    f"约 1:4。属类别不平衡，建模时做了加权")

_DIST_NAME = {"loanAmnt": "贷款金额", "interestRate": "利率", "installment": "分期额",
              "annualIncome": "年收入", "dti": "负债收入比", "revolUtil": "额度使用率"}
FIG_DIST = svg_figure(
    SC.chart_minis([{"title": f"{_DIST_NAME[c]}　{c}",
                     "svg": SC.chart_hist(C["dist"][c], tip_prefix=f"{_DIST_NAME[c]} ")}
                    for c in C["dist"]], cols=3),
    "图 2｜六个核心数值变量（清洗后训练集）：金额、分期额、利率单峰右偏；年收入极端长尾，"
    "一半人在 6.5 万以下、最高 1,100 万；额度使用率封顶在 100。量纲差几个数量级，建模前要做变换。")

_CMP_UNIT = {"loanAmnt": " 元", "interestRate": "%", "installment": " 元",
             "annualIncome": " 元", "dti": "%", "revolUtil": ""}
FIG_CMP = svg_figure(
    SC.chart_box([{"title": _DIST_NAME[c] + ("（log 刻度）" if c == "annualIncome" else ""),
                   "unit": _CMP_UNIT[c], "log": c == "annualIncome",
                   "groups": [("正常", C["cmp"][c]["normal"]), ("违约", C["cmp"][c]["default"])]}
                  for c in C["cmp"]], cols=3),
    "图 3｜违约组与正常组的对比（箱体＝四分位区间，白线＝中位数，须＝极值）：违约组的利率、金额、"
    "负债收入比中位数分别高 24%、19%、15%；年收入低 8%。年收入用 log 刻度。")

_g = C["grade"]
FIG_GRADE = svg_figure(
    SC.chart_bars([(n, v, f"等级 {n}：违约率 {_pc(v / 100)}，样本 {_cnt(s)} 笔")
                   for n, v, s in _g],
                  w=520, h=300, ylab="违约率（%）", val_fmt=lambda v: f"{v:.1f}%",
                  bar_color=lambda i: ["#e8b4b8", "#e0a0a6", "#d78c94", "#cd7882", "#c3646f",
                                       "#b8505d", "#a93c4b"][i]),
    f"图 4｜各信用等级的违约率：从 A 的 {_g[0][1]:.1f}% 单调升到 {_g[-1][1]:.1f}%，"
    f"差 {_g[-1][1] / _g[0][1]:.1f} 倍。这是最强的单变量分层，业务最该先看")

_corr_lab = {"loanAmnt": "贷款金额", "interestRate": "利率", "installment": "分期额",
             "annualIncome": "年收入", "dti": "负债收入比", "revolUtil": "额度使用率",
             "ficoRangeLow": "FICO 下限", "isDefault": "是否违约"}
_corr_names = [_corr_lab[x] for x in C["corr"]["labels"]]
FIG_CORR = svg_figure(
    SC.chart_heatmap(_corr_names, C["corr"]["matrix"],
                     cell_tip=lambda i, j, v: f"{_corr_names[i]} × {_corr_names[j]}：相关系数 {v:.2f}"),
    "图 5｜相关性热力图（7 个关键数值特征 + 目标）：金额与分期额 0.95 最高——分期额是按金额、利率、"
    "期限算出的月供（99.8% 的行能对上），两者信息重复；其余两两相关最高 0.46；与违约相关性最高的是利率 0.26。")

_inc = C["income"]
FIG_INCOME = svg_figure(
    SC.chart_minis([
        {"title": "年收入（原始刻度）", "svg": SC.chart_hist(_inc["raw"], tip_prefix="年收入 ",
                                                        x_fmt=lambda v: f"{v / 10000:.1f} 万")},
        {"title": "年收入（log10 变换后）", "svg": SC.chart_hist(_inc["log"], tip_prefix="log10 年收入 ",
                                                            x_fmt=lambda v: f"{v:.1f}")}], cols=2),
    "图 6｜年收入原始刻度的偏度是 46.4，log 变换后降到 0.20，接近对称。这是 W2 特征工程里对年收入取对数的依据")

# ---- ② 模型成绩图 ----
_roc = C["roc"]
FIG_ROC = svg_figure(
    SC.chart_lines(
        [{"name": f"ROC 曲线（AUC {_roc['auc']:.4f}）", "color": SC.BLUE, "pts": _roc["pts"],
          "tip": lambda f, t: f"FPR {_pc(f)}：TPR {_pc(t)}"},
         {"name": "随机猜", "color": "#9aa6b4", "dash": "6 4", "pts": [[0, 0], [1, 1]],
          "tip": lambda f, t: f"随机猜：FPR {_pc(f)} → TPR {_pc(t)}"}],
        w=560, h=330, xlab="假阳性率（正常人被拒的比例）", ylab="真阳性率（违约者被拦的比例）",
        x_fmt=lambda v: f"{v * 100:.0f}%", y_fmt=lambda v: f"{v * 100:.0f}%", marker_every=12),
    f"图 7｜测试集 ROC 曲线（AUC {_roc['auc']:.4f}）：曲线越靠左上越好。"
    f"AUC 只看排序，与概率刻度无关，校准前后一样；沿曲线悬浮可读数")

_pr = C["pr"]
FIG_PR = svg_figure(
    SC.chart_lines(
        [{"name": f"PR 曲线（AP {_pr['ap']:.3f}）", "color": SC.BLUE, "pts": _pr["pts"],
          "tip": lambda r, p: f"召回 {_pc(r)}：精确率 {_pc(p)}"},
         {"name": f"随机基线 {_pr['base']:.3f}", "color": "#9aa6b4", "dash": "6 4",
          "pts": [[0, _pr["base"]], [1, _pr["base"]]],
          "tip": lambda r, p: f"不排序、随机拒：精确率≈违约率 {_pc(p)}"}],
        w=560, h=330, xlab="召回率（违约者被拦下的比例）", ylab="精确率（被拒者里真违约的比例）",
        marker_every=12,
        x_fmt=lambda v: f"{v * 100:.0f}%", y_fmt=lambda v: f"{v * 100:.0f}%"),
    f"图 8｜测试集 PR 曲线（AP {_pr['ap']:.3f}）：类别不平衡时更该看它，基线不是对角线而是违约率 {_pc(_pr['base'])}")

FIG_CM = svg_figure(
    SC.chart_confusion(C["cm"]),
    f"图 9｜测试集混淆矩阵（阈值 0.50）：拒 {_cnt(C['cm']['tp'] + C['cm']['fp'])} 人，"
    f"其中真违约 {_cnt(C['cm']['tp'])} 人（精确率 {_pc(C['cm']['tp'] / (C['cm']['tp'] + C['cm']['fp']))}）；"
    f"漏放 {_cnt(C['cm']['fn'])} 人（召回率 {_pc(C['cm']['tp'] / (C['cm']['tp'] + C['cm']['fn']))}）。悬浮格子看人数")

_lc = C["learn"]
FIG_LEARN = svg_figure(
    SC.chart_lines(
        [{"name": "训练集 AUC", "color": SC.ORANGE, "pts": [[r[0], r[1]] for r in _lc],
          "tip": lambda n, v: f"训练 {_cnt(n)} 笔：训练集 AUC {v:.4f}"},
         {"name": "验证集 AUC", "color": SC.BLUE, "pts": [[r[0], r[2]] for r in _lc],
          "tip": lambda n, v: f"训练 {_cnt(n)} 笔：验证集 AUC {v:.4f}"}],
        w=560, h=330, xlab="训练样本量", ylab="AUC",
        x_fmt=lambda v: f"{v / 10000:.0f} 万", y_pad=0.004),
    "图 10｜学习曲线：训练集成绩随样本量下降、验证集成绩一路升到 0.7294，两条线还在收窄，说明没有过拟合，"
    "再加数据仍有小幅收益")

_tune_names = {"Grid": "网格", "Random": "随机", "Bayesian(TPE)": "贝叶斯"}
# 不同方法配不同线型：XGB 的网格与贝叶斯成绩完全相同，实线叠在一起会互相盖住
_tune_dash = {"Grid": None, "Random": "7 4", "Bayesian(TPE)": "2 3"}
FIG_TUNING = svg_figure(
    SC.chart_lines(
        [{"name": f"{s['model'].replace('XGBoost', 'XGB').replace('LightGBM', 'LGB')}·{_tune_names.get(s['method'], s['method'])}",
          "color": SC.PALETTE[i % len(SC.PALETTE)],
          "dash": _tune_dash.get(s["method"]),
          "pts": s["pts"],
          "tip": (lambda ss: lambda t, v: f"{ss['model']}·{ss['method']} 第 {int(t)} 次：当前最优 CV AUC {v:.4f}")(s)}
         for i, s in enumerate(C["tuning"])],
        w=900, h=330, xlab="调参试验序号（每个方法各 24 次，等预算）", ylab="当前最优 CV AUC",
        x_fmt=lambda v: f"{v:.0f}", y_pad=0.0004, marker_every=6),
    "图 11｜2 个模型 × 3 种调参方法（20 万行子样本、3 折、各 24 次预算）：XGBoost 的网格与贝叶斯命中同一组参数、"
    "并列 0.7217；LightGBM 三种方法并列 0.7210。同一模型内三种方法上限一致；这张图只比调参方法，不比模型优劣")

_imp = C["imp"]
FIG_IMP = svg_figure(
    SC.chart_hbars([(n, v, f"{n}：重要性 {v:.4f}") for n, v in _imp],
                   w=560, h=430, color=SC.BLUE, xlab="随机森林重要性"),
    "图 12｜随机森林特征重要性（W2，前 15）：子等级、利率、等级三项占了大头，与后面的 SHAP 结论一致")

_shap = C["shap"]
FIG_SHAP = svg_figure(
    SC.chart_hbars([(n, v, f"{n}：平均 |SHAP| {v:.4f}") for n, v in _shap],
                   w=560, h=430, color=SC.GREEN, xlab="平均 |SHAP|（对预测的贡献）"),
    "图 13｜SHAP 全局归因（W3，前 15）：子等级、期限、住房情况贡献最大。"
    "SHAP 看的是每个特征把预测推高多少，和随机森林重要性排序接近，互为印证")

_fc = C["fc"]
FIG_FC = svg_figure(
    SC.chart_lines(
        [{"name": "训练集 AUC", "color": SC.ORANGE, "pts": [[r[0], r[1]] for r in _fc],
          "tip": lambda n, v: f"用前 {int(n)} 个特征：训练集 AUC {v:.4f}"},
         {"name": "验证集 AUC", "color": SC.BLUE, "pts": [[r[0], r[2]] for r in _fc],
          "tip": lambda n, v: f"用前 {int(n)} 个特征：验证集 AUC {v:.4f}"}],
        w=900, h=330, xlab="使用特征数（按 W2 重要性从高到低加入）", ylab="AUC",
        x_fmt=lambda v: f"{v:.0f}", y_pad=0.0015),
    "图 14｜特征数量 vs AUC：前 3 个特征就到验证集 0.7005，加到 20 个 0.7253，之后每加一个只涨不到 0.001。"
    "想省采集成本可以砍到 20 个左右，代价约 0.004 AUC。悬浮看每个特征数")

# ---- ④ 敏感性热力图：把 16 行表压成一张图 ----
_probes = C["sens"]["probes"]
_HUE_CN = {"Interest income (full rate)": "利息全额（上限）", "Net margin 9%": "净息差 9%",
           "Net margin 6%": "净息差 6%", "Net margin 3%": "净息差 3%"}
_rows = ["Interest income (full rate)", "Net margin 9%", "Net margin 6%", "Net margin 3%"]
_cols_rec = [0.2, 0.3, 0.4, 0.5]
_lookup = {(p["name"], round(p["rec"], 2)): p for p in _probes}


def _sens_cell(i, j, v):
    p = _lookup[(_rows[i], round(_cols_rec[j], 2))]
    return (f"{_HUE_CN[_rows[i]]} · 回收率 {_cols_rec[j]:.0%}[[BR]]"
            f"拒绝率 {_pc(p['reject'])}｜拦截率 {_pc(p['catch'])}｜被拒者中真违约 {_pc(p['prec'])}[[BR]]"
            f"隐含线（校准概率轴）3 年 {p['p3']:.4f}、5 年 {p['p5']:.4f}")


FIG_SENS_HEAT = svg_figure(
    '<div class="heatnote"><span><i style="background:#c0392b"></i>颜色越深＝拒得越多</span>'
    '<span>纵向换收益口径，横向换回收率；每格悬浮看隐含审批线与拦截率</span>'
    f'<span>格里的隐含线算在<b>校准概率轴</b>上；审批系统实际用的是原始概率轴，'
    f'中性口径同一批人换算过去是 3 年 {RAW3:.2f}、5 年 {RAW5:.2f}</span></div>'
    + SC.chart_heatmap([f"回收率 {r:.0%}" for r in _cols_rec],
                       [[_lookup[(r, round(c, 2))]["reject"] for c in _cols_rec] for r in _rows],
                       cell=132, cell_h=62, vmin=0.0, vmax=0.72, fmt=lambda v: _pc(v),
                       row_labels=[_HUE_CN[r] for r in _rows], cell_tip=_sens_cell),
    "图 17｜期望损失敏感性（拒绝率）：同一行往右，回收率越高越敢放；纵向往下的影响大得多。"
    "每格的隐含线算在校准概率轴上。"
    "回收率 20% 那一列，收益口径从利息全额换到净息差 3%，拒绝率从 3.6% 涨到 70.3%",
    cls="fig heat")



gen = pd.read_csv(W4 / "generalization_compare.csv", index_col=0).rename(
    index={"train": "训练集", "val": "验证集", "test": "测试集"})
gen = gen.reset_index().rename(columns={"index": "数据集", "accuracy": "准确率", "precision": "精确率",
                                        "recall": "召回率", "f1": "F1", "auc": "AUC"})

cmp_df = pd.read_csv(OUT / "W3" / "model_comparison_optimized.csv").rename(columns={
    "model": "模型", "accuracy": "准确率", "precision": "精确率", "recall": "召回率",
    "f1": "F1", "auc": "AUC"})

thr = pd.read_csv(W4 / "threshold_table.csv").rename(columns={
    "threshold": "阈值", "precision": "精确率", "recall": "召回率", "f1": "F1",
    "default_rate": "拒绝率"}).assign(阈值=lambda d: d["阈值"].map(lambda x: f"{x:.2f}"))

# 预设档位表（带钱）：同一批测试借款人的指标 + 金额，参数取中性口径
thr2 = pd.DataFrame([{
    "阈值": f'{g["t"]:.2f}',
    "拒绝率": g["reject_rate"],
    "被拒者精确率": g["precision"],
    "拦截率": g["recall"],
    "少放贷款": f'{g["loans_rej"]:,.0f}',
    "避免坏账": f'{(1 - REC_BASE) * g["avoided_base"]:,.0f}',
    "放弃收入": f'{V_NEUTRAL * g["foregone_base"]:,.0f}',
    "净增益": f'{g["gain"]:,.0f}',
} for g in grid if 0.30 - 1e-9 <= g["t"] <= 0.70 + 1e-9])

# ---- 补充分析 1：区分度（KS 与头部捕获率）----
hc = pd.read_csv(W4 / "head_capture.csv")
_ks = float(pd.read_csv(W4 / "ks_summary.csv").iloc[0]["值"])
_ks_chk = float(np.max(np.abs(np.cumsum(y[np.argsort(prob)]) / N_POS
                              - np.cumsum(1 - y[np.argsort(prob)]) / N_NEG)))
assert abs(_ks - _ks_chk) < 1e-9, f"KS 与测试集现算不一致：{_ks:.4f} vs {_ks_chk:.4f}"
assert 0.25 < _ks < 0.40, f"KS 落在预期外：{_ks:.4f}"
HEAD_TBL_HTML = render_table(pd.DataFrame([{
    "头部占比": f'前 {row["头部占比"] * 100:.0f}%',
    "人数": f'{int(row["头部人数"]):,}',
    "抓到违约者": f'{int(row["命中违约人数"]):,}',
    "捕获率": f'{row["捕获率"] * 100:.1f}%',
    "是随机挑的": f'{row["lift"]:.1f} 倍',
    "头部内违约率": f'{row["头部内违约率"] * 100:.1f}%',
} for _, row in hc.iterrows()]))

# ---- 补充分析 2：时间外验证（按放款日期切）----
oot = pd.read_csv(W4 / "oot_validation.csv").set_index("口径")
_RAND, _OOT = "随机切分（对照）", "时间外（2017-01 起）"
assert set(oot.index) == {_RAND, _OOT}, f"oot_validation.csv 口径异常：{list(oot.index)}"
assert oot.loc[_OOT, "AUC"] < oot.loc[_RAND, "AUC"], "时间外 AUC 不应高于随机切分"
assert oot.loc[_OOT, "违约率"] > oot.loc[_RAND, "违约率"], "时间外违约率不应低于随机切分"
OOT_TBL_HTML = render_table(pd.DataFrame([{
    "切分方式": name,
    "样本数": f'{int(oot.loc[name, "样本数"]):,}',
    "违约率": f'{oot.loc[name, "违约率"] * 100:.1f}%',
    "AUC": f'{oot.loc[name, "AUC"]:.4f}',
    "KS": f'{oot.loc[name, "KS"]:.4f}',
    "前 10% 捕获率": f'{oot.loc[name, "前10%捕获率"] * 100:.1f}%',
    "阈值 0.5 拒绝率": f'{oot.loc[name, "阈值0.5拒绝率"] * 100:.1f}%',
} for name in [_RAND, _OOT]]))

# ---- 补充分析 3：人工复审带（拆线附近的人交人工）----
rb = pd.read_csv(W4 / "review_band.csv")
_rbk = pd.read_csv(W4 / "review_band_key.csv").set_index("项目")["值"]
_band = np.zeros(N, bool)
_far = np.zeros(N, bool)
for _bt in sorted(set(term)):
    _bps = V_NEUTRAL * _bt / (V_NEUTRAL * _bt + 1 - REC_BASE)
    _bm = term == _bt
    _band |= _bm & (p_cal >= _bps) & (p_cal < _bps * 1.2)
    _far |= _bm & (p_cal >= _bps * 1.2)
_ep_band = float(((1 - p_cal) * V_NEUTRAL * term * loan
                  - p_cal * (1 - REC_BASE) * loan)[_band].sum() / 1e4)
assert int(_band.sum()) == int(_rbk["复审带人数"]), "复审带人数与 review_band_key.csv 不一致"
assert abs(_ep_band - float(_rbk["复审带全放期望利润(万元)"])) < 0.5, \
    f"复审带钱账与 review_band_key.csv 不一致：{_ep_band:.1f}"
REVIEW_TBL_HTML = render_table(pd.DataFrame([{
    "分工": row["分工"],
    "人数": f'{int(row["人数"]):,}',
    "占测试集": f'{row["占测试集"] * 100:.1f}%',
    "实际违约率": f'{row["实际违约率"] * 100:.1f}%',
    "说明": row["说明"],
} for _, row in rb.iterrows()]))
RB_N = int(rb.loc[rb["分工"] == "人工复审", "人数"].iloc[0])
RB_DECIDED_PCT = float(rb.loc[rb["分工"] != "人工复审", "占测试集"].sum())
RB_DR = float(rb.loc[rb["分工"] == "人工复审", "实际违约率"].iloc[0])
RB_APPR_DR = float(rb.loc[rb["分工"] == "直接放行", "实际违约率"].iloc[0])
RB_FAR_DR = float(rb.loc[rb["分工"] == "直接拒绝", "实际违约率"].iloc[0])
_band30 = np.zeros(N, bool)
for _bt in sorted(set(term)):
    _bps = V_NEUTRAL * _bt / (V_NEUTRAL * _bt + 1 - REC_BASE)
    _bm = term == _bt
    _band30 |= _bm & (p_cal >= _bps) & (p_cal < _bps * 1.3)
RB_W30 = int(_band30.sum())
assert RB_W30 > RB_N, "带宽 30% 的复审人数不应少于 20%"
assert 0.4 < (RB_APPR_DR / RB_DR) < 0.6, "放行段与复审带的违约率关系异常"

cal_cmp = pd.read_csv(W4 / "calibration_compare.csv")[["stage", "note", "brier", "auc", "mean_pred", "reject_rate", "precision", "recall"]]
cal_cmp = cal_cmp.rename(columns={"stage": "方法", "note": "说明", "brier": "Brier 分数", "auc": "AUC",
                                  "mean_pred": "平均预测概率", "reject_rate": "拒绝率",
                                  "precision": "精确率", "recall": "召回率"})

cal_bins = pd.read_csv(W4 / "calibration_bins.csv").rename(columns={
    "bin": "分箱", "count": "人数", "mean_pred_raw": "未校准平均概率",
    "mean_pred_cal": "校准后平均概率", "actual_rate": "实际违约率"})
cal_bins["档位"] = "第 " + (cal_bins["分箱"] + 1).astype(int).astype(str) + " 档"
cal_bins = cal_bins[["档位", "人数", "未校准平均概率", "校准后平均概率", "实际违约率"]]

el = pd.read_csv(W4 / "expected_loss_summary.csv")[
    ["策略", "批准占比", "被拒占比", "拦截率(真违约被拒占比)", "被拒者精确率(里面真违约占比)",
     "误伤率(正常人被拒占比)", "总期望利润(万元)", "说明"]]
el = el.rename(columns={"拦截率(真违约被拒占比)": "拦截率", "被拒者精确率(里面真违约占比)": "被拒者精确率",
                        "误伤率(正常人被拒占比)": "误伤率"})
el["说明"] = el["说明"].replace({"第 10 节原口径": "对照：一条线切在原始概率 0.5 上"})

sens = pd.read_csv(W4 / "expected_loss_sensitivity.csv")
sens["收益口径"] = sens["收益口径"].replace({
    "Interest income (full rate)": "利息全额（上限）",
    "Net margin 9%": "净息差 9%",
    "Net margin 6%": "净息差 6%",
    "Net margin 3%": "净息差 3%"})
_ss = sens.set_index(["收益口径", "回收率"])

# 敏感性表的每一行对应滑块上的一个位置（收益口径里的三档净息差 × 回收率）；
# "利息全额（上限）"是个上界假设，滑块上找不到对应点，所以不给 data 属性。
_SENS_MAR = {"净息差 9%": 0.09, "净息差 6%": 0.06, "净息差 3%": 0.03}


def _sens_attrs(row) -> str:
    mar = _SENS_MAR.get(row["收益口径"])
    if mar is None:
        return ""
    return f' data-mar="{mar:.2f}" data-rec="{float(row["回收率"]):.2f}"'
_S_OPT = float(_ss.loc[("利息全额（上限）", 0.2), "拒绝率"])      # 最松口径 + 最差催收
_S_CONS = float(_ss.loc[("净息差 3%", 0.2), "拒绝率"])           # 最紧口径 + 最差催收
_S_N_REJ = float(_ss.loc[("净息差 6%", 0.3), "拒绝率"])          # 中性口径 = 页头用的那一行
_S_N_REC = float(_ss.loc[("净息差 6%", 0.3), "拦截率"])
_S_N_PRE = float(_ss.loc[("净息差 6%", 0.3), "被拒者精确率"])
_S_N_IMP3 = float(_ss.loc[("净息差 6%", 0.3), "隐含阈值 3 年"])
_S_N_IMP5 = float(_ss.loc[("净息差 6%", 0.3), "隐含阈值 5 年"])
assert abs(_S_N_IMP3 - 0.204545) < 1e-4 and abs(_S_N_IMP5 - 0.30) < 1e-4, \
    f"中性口径的隐含阈值异常：{_S_N_IMP3:.4f} / {_S_N_IMP5:.4f}"
_S_6_20 = float(_ss.loc[("净息差 6%", 0.2), "拒绝率"])            # 同一赚头口径，催收最差
_S_6_50 = float(_ss.loc[("净息差 6%", 0.5), "拒绝率"])            # 同一赚头口径，催收最好
_S_CONS30 = float(_ss.loc[("净息差 3%", 0.3), "拒绝率"])          # 保守口径，回收率取中性那档
_S_OPT30 = float(_ss.loc[("利息全额（上限）", 0.3), "拒绝率"])     # 乐观口径，回收率取中性那档

# 刻度不准的代价：同一套逐笔规则，换成未校准概率会怎样（expected_loss_summary.csv 最后一行）
_EL_RAW_M = float(el.loc[el["策略"] == "期望损失框架（未校准概率）", "总期望利润(万元)"].iloc[0])
_EL_RAW_REJ = float(el.loc[el["策略"] == "期望损失框架（未校准概率）", "被拒占比"].iloc[0]) * 100
_EL_NEU_M = float(el.loc[el["策略"] == "期望损失框架 · 中性（净息差 6%）", "总期望利润(万元)"].iloc[0])
_EL_NEU_REJ = float(el.loc[el["策略"] == "期望损失框架 · 中性（净息差 6%）", "被拒占比"].iloc[0]) * 100
_SCALE_GAIN = _EL_NEU_M - _EL_RAW_M
assert abs(_SCALE_GAIN - 4502) < 5, f"修刻度带来的差额异常：{_SCALE_GAIN:.1f} 万元"
assert abs(_EL_RAW_REJ - 87.7) < 0.3, f"未校准规则拒绝率异常：{_EL_RAW_REJ:.1f}%"
assert 0.02 < _S_OPT < 0.05 and 0.65 < _S_CONS < 0.75, f"敏感性两端异常：{_S_OPT:.3f} / {_S_CONS:.3f}"
assert abs(_S_N_REJ - 0.3516) < 0.005 and abs(_S_N_PRE - 0.339) < 0.005, "中性行与既有结论对不上"
assert _S_6_20 / _S_6_50 > 1.8 and _S_CONS30 / _S_OPT30 > 20, \
    f"敏感性两根杠杆的倍数不对：{_S_6_20/_S_6_50:.2f} / {_S_CONS30/_S_OPT30:.1f}"

# ---- 滑块的实时读数：同一套规则在 21×21 参数网格上的结果，前端查表即可 ----
# 规则与 expected_loss_framework.py 逐字一致：EP = (1−p)·v·T − p·(1−rec)，EP ≤ 0 就拒；
# 等价写法 d_i = v·T·(1−p)/p，d_i ≤ 1−rec 就拒。
_is_bad = (y == 1)
_d_base = term * (1 - p_cal) / p_cal
SENS_LIVE = []
for _rec in ERECS:
    _lim = 1 - _rec
    _rrow = []
    for _v in EMARS:
        _rej = (_v * _d_base) <= _lim
        _n_rej = int(_rej.sum())
        # 保留 6 位小数：只留 4 位时 20.4545% 会被前端四舍五入成 20.4%，和敏感性表那一行的 20.5% 打架
        _thr = _v * term / (_v * term + _lim)
        _rrow.append([round(float(np.median(_thr[term == 3])), 6),
                      round(float(np.median(_thr[term == 5])), 6),
                      round(_n_rej / N, 6),
                      round(float(_is_bad[_rej].sum()) / N_POS, 6),
                      round(float(_is_bad[_rej].mean()) if _n_rej else 0.0, 6)])
    SENS_LIVE.append(_rrow)

# 和 expected_loss_sensitivity.csv 的 12 个格子逐一对照（9% / 6% / 3% × 回收率 20–50%）
for _lab, _v in [("净息差 9%", 0.09), ("净息差 6%", 0.06), ("净息差 3%", 0.03)]:
    for _rec in (0.2, 0.3, 0.4, 0.5):
        _g = SENS_LIVE[ERECS.index(round(_rec, 2))][EMARS.index(_v)]
        _r = _ss.loc[(_lab, _rec)]
        assert abs(_g[0] - _r["隐含阈值 3 年"]) < 1e-3 and abs(_g[1] - _r["隐含阈值 5 年"]) < 1e-3 \
            and abs(_g[2] - _r["拒绝率"]) < 1e-4 and abs(_g[3] - _r["拦截率"]) < 1e-4 \
            and abs(_g[4] - _r["被拒者精确率"]) < 1e-4, \
            f"滑块网格与敏感性表的 {_lab} / {_rec} 对不上：{_g} vs {_r.to_dict()}"

_SLV = SENS_LIVE[ERECS.index(0.30)][EMARS.index(0.06)]      # 滑块默认档 = 中性口径
assert abs(_SLV[0] - _S_N_IMP3) < 1e-3 and abs(_SLV[1] - _S_N_IMP5) < 1e-3 \
    and abs(_SLV[2] - _S_N_REJ) < 1e-3 and abs(_SLV[4] - _S_N_PRE) < 1e-3, \
    f"滑块默认档与中性行不一致：{_SLV}"
DEF["E_IMP3"] = f"{_SLV[0] * 100:.1f}%"
DEF["E_IMP5"] = f"{_SLV[1] * 100:.1f}%"
DEF["E_REJ"] = f"{_SLV[2] * 100:.1f}%"
DEF["E_REC"] = f"{_SLV[3] * 100:.1f}%"
DEF["E_PRE"] = f"{_SLV[4] * 100:.1f}%"
# ④ 滑块的实时读数：[回收率档][净息差档] = [3 年期线, 5 年期线, 拒绝率, 拦截率, 被拒者精确率]
DATA["sens"] = SENS_LIVE

fc = pd.read_csv(W4 / "feature_count_auc.csv").rename(columns={
    "n_features": "使用特征数", "train_auc": "训练集 AUC", "val_auc": "验证集 AUC",
    "seconds": "训练秒数", "added": "本步新增特征"}).rename(columns={"﻿n_features": "使用特征数"})

PCT = {"拒绝率", "精确率", "召回率", "F1", "准确率", "AUC", "批准占比", "被拒占比", "拦截率",
       "被拒者精确率", "误伤率", "平均预测概率", "拒绝率", "实际违约率", "未校准平均概率",
       "校准后平均概率", "回收率", "隐含阈值 3 年", "隐含阈值 5 年",
       "训练集 AUC", "验证集 AUC", "Brier 分数"}
INT = {"人数", "使用特征数"}

# ----------------------------------------------------------------------------
# 3) 拼 HTML
# ----------------------------------------------------------------------------
CSS = """
:root{--ink:#1b1f24;--sub:#5b6470;--line:#e3e7ec;--bg:#f5f7f9;--card:#fff;--blue:#1f5fbf;--red:#c0392b;--green:#1e8449;--orange:#c77b1a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
header{padding:22px 26px 16px;background:#fff;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:21px}
h2{font-size:17px;margin:26px 0 10px}
h3{font-size:15px;margin:18px 0 8px;color:var(--sub)}
.sub{margin:0;color:var(--sub);font-size:13px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:12px 0 4px}
.facts>div{background:#fff;border:1px solid var(--line);border-radius:8px;padding:9px 12px;font-size:12.5px;line-height:1.65;color:#3c4550}
.facts>div b{color:var(--blue);margin-right:6px}
.hint{margin:8px 0 0;color:var(--sub);font-size:12px}
nav{position:sticky;top:0;z-index:9;display:flex;flex-wrap:wrap;gap:6px;padding:10px 20px;background:#fff;border-bottom:1px solid var(--line)}
nav label{border:1px solid var(--line);background:#fff;color:var(--ink);padding:7px 14px;border-radius:18px;font-size:13px;cursor:pointer;user-select:none}
nav label:hover{border-color:#b9c6d6}
.navsep{width:1px;background:var(--line);margin:4px 8px;align-self:stretch}
.verdict{margin:10px 0 6px;padding:10px 14px;background:#f2f7fd;border-left:3px solid var(--blue);
  border-radius:4px;font-size:15px;line-height:1.7}
.tabin{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}
#t1:checked~nav label[for=t1],#t2:checked~nav label[for=t2],#t3:checked~nav label[for=t3],#t4:checked~nav label[for=t4]{background:var(--blue);border-color:var(--blue);color:#fff}
main{padding:18px 20px 40px;max-width:1180px;margin:0 auto}
.panel{display:none}
#t1:checked~main #p1,#t2:checked~main #p2,#t3:checked~main #p3,#t4:checked~main #p4{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.cards.hero{grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:12px;margin:16px 0 10px}
.cards.hero .kpi{padding:12px 15px;border-left:3px solid var(--blue);box-shadow:0 1px 2px rgba(27,31,36,.04)}
.cards.hero .kpi .k{font-size:12.5px}
.cards.hero .kpi .v{font-size:23px;letter-spacing:-.2px}
.statline{display:flex;flex-wrap:wrap;gap:0 0;align-items:baseline;margin:2px 0 12px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px 4px}
.statline>div{flex:1 1 0;min-width:150px;padding:0 14px;border-right:1px solid var(--line)}
.statline>div:last-child{border-right:0}
.statline .k{font-size:12px;color:var(--sub)}
.statline .v{font-size:17px;font-weight:600;margin-top:1px}
.statline .n{font-size:11.5px;color:var(--sub)}
.kpi{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .k{font-size:12px;color:var(--sub)}
.kpi .v{font-size:22px;font-weight:600;margin-top:2px}
.kpi .n{font-size:12px;color:var(--sub);margin-top:2px}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl thead th{background:#f0f3f7;color:var(--sub);font-weight:600}
.tbl tbody tr:nth-child(even){background:#fafbfc}
.tbl tr.sel td{background:#e8f0fe}
.tbl tr.sel td:first-child{box-shadow:inset 3px 0 0 var(--blue);font-weight:600}
.scroll{overflow-x:auto}
.tbl .na{color:#aab3bd}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.grid2.tiles{align-items:stretch}
h4{margin:0 0 6px;font-size:14px}
.cht{padding:12px 14px 8px}
.cht svg{border:0;border-radius:0;background:transparent}
.cht h4 .cht-sub{font-weight:400;color:var(--sub);font-size:12.5px;margin-left:8px}
.cht .lg{font-size:11.5px;gap:6px 14px;margin:8px 0 2px}
.sliders3{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:2px 26px}
.sliders3 .slider-row{margin:4px 0 8px}
.sliders3 .thr-val{font-size:19px;min-width:58px}
figure.fig.pngcap{max-width:780px;margin-left:auto;margin-right:auto}
figure.fig.heat{max-width:734px;margin-left:auto;margin-right:auto}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;display:flex;flex-direction:column}
.tile h2{margin:0 0 8px;font-size:15px}
.tile figure{margin:0;flex:1;display:flex;flex-direction:column;justify-content:center}
.tile figure.fig>svg{width:100%;display:block}
#p1 .tile figure.fig>svg{max-height:300px}
.tile figcaption{font-size:12px;line-height:1.55}
.tile.wide{grid-column:1/-1}
figure{margin:12px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px}
figure img{width:100%;display:block;border-radius:4px}
figcaption{margin-top:6px;color:var(--sub);font-size:12px}
.sim{display:grid;grid-template-columns:1.35fr 1fr;gap:14px;align-items:start}
.slider-row{display:flex;align-items:center;gap:12px;margin:6px 0 12px}
input[type=range]{flex:1;accent-color:var(--blue)}
.thr-val{font-size:22px;font-weight:600;min-width:64px;text-align:right}
.ledger{background:#f7f9fc;border-left:3px solid var(--blue);padding:10px 12px;border-radius:4px;font-size:13px;margin-top:10px}
.advice{margin-top:10px;background:#f7f9fc;border-left:3px solid var(--orange);padding:10px 12px;border-radius:4px;font-size:13px}
.advice .tone{font-weight:600;display:block;margin-bottom:4px}
.advice .pick{background:#fff;border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin-top:6px}
.keybox{background:#f7f9fc;border:1px solid var(--line);border-left:3px solid var(--blue);border-radius:6px;padding:10px 14px;font-size:13px;margin:10px 0}
.keybox ol,.keybox ul{margin:6px 0;padding-left:20px}
.keybox li{margin:6px 0}
.keybox p{margin:6px 0}
.keybox .tail{border-top:1px dashed var(--line);padding-top:8px;margin-top:10px}
.keybox ul.ways{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:6px 18px;margin:8px 0 2px;padding-left:0;list-style:none}
.keybox ul.ways li{margin:0}
.code-line{margin:8px 0 8px 12px}
.code-line code{background:#eef2f7;padding:3px 7px;border-radius:4px;font-size:13px}
details.how{margin-top:12px;background:#fbfcfd;border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:13px}
details.how summary{cursor:pointer;color:var(--blue);font-size:13px}
details.how ul{margin:6px 0;padding-left:20px}
details.how code{background:#eef2f7;padding:1px 4px;border-radius:3px}
.slider-row .sub{min-width:72px}
.cm{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.cm-cell{border-radius:8px;padding:10px;color:#fff}
.cm-cell .n{font-size:20px;font-weight:600;display:block}
.cm-cell .l{font-size:11.5px;opacity:.92}
.cm-tn{background:#2e7d5b}.cm-tp{background:#1f5fbf}.cm-fp{background:#c77b1a}.cm-fn{background:#8e3b3b}
.lg{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--sub);margin:6px 0}
.lg i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
svg{width:100%;height:auto;background:#fff;border:1px solid var(--line);border-radius:10px}
footer{color:var(--sub);font-size:12px;padding:0 20px 40px;max-width:1180px;margin:0 auto}
ol,ul{padding-left:20px}
/* 仪表盘交互：图片放大、说明折叠、图表悬浮读数 */
figure img{cursor:zoom-in}
figure:hover img{box-shadow:0 4px 16px rgba(20,40,80,.16)}
#zoom{display:none;position:fixed;inset:0;background:rgba(14,18,24,.9);z-index:99;padding:22px;align-items:center;justify-content:center;cursor:zoom-out}
#zoom.on{display:flex}
#zoom img{max-width:100%;max-height:100%;width:auto;border-radius:6px;background:#fff}
.tools{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0 0}
.tools button{border:1px solid var(--line);background:#fff;border-radius:16px;padding:5px 13px;font-size:12.5px;color:var(--ink);cursor:pointer}
.tools button:hover{border-color:var(--blue);color:var(--blue)}
.ctip{position:absolute;pointer-events:none;background:rgba(23,28,35,.93);color:#fff;font-size:12px;line-height:1.6;padding:5px 9px;border-radius:5px;white-space:nowrap;opacity:0;transition:opacity .12s;z-index:6}
.ctip.fl{position:fixed;z-index:120;white-space:normal;max-width:min(360px,calc(100vw - 24px))}
figure svg text{pointer-events:none}
.ctip i{display:block;font-style:normal;opacity:.72;font-size:11.5px}
.ctip.on{opacity:1}
details.fold{margin:8px 0;background:#fbfcfd;border:1px solid var(--line);border-radius:8px;padding:6px 12px}
details.fold>summary{cursor:pointer;color:var(--ink);font-size:13px;line-height:1.65;outline:none}
details.fold>summary:hover{color:var(--blue)}
.ftag{display:inline-block;background:#e9f0fb;color:var(--blue);border-radius:4px;font-size:11.5px;
  font-weight:600;padding:1px 6px;margin-right:8px;vertical-align:1px}
details.fold[open]{background:#fff}
details.fold>p{margin:8px 0 6px}
figure.fig svg{border:0;border-radius:0;background:transparent}
figure.fig{padding:10px 12px 6px}
.minis{display:grid;grid-template-columns:repeat(var(--cols,3),minmax(0,1fr));gap:10px;margin:6px 0}
.mini{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 8px 2px}
.mini .mt{font-size:12px;color:var(--sub);margin:2px 0 3px}
.mini svg{border:0;border-radius:0;background:transparent}
.heatnote{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--sub);margin:2px 0 4px}
.heatnote i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:5px}
@media (max-width:820px){.grid2{grid-template-columns:1fr}.sim{grid-template-columns:1fr}.sliders3{grid-template-columns:1fr}main{padding:14px}}
@media print{nav{display:none}.panel{display:block!important;page-break-inside:avoid}body{background:#fff}.card,figure{border-color:#ccc}}
"""

JS = """
const D = __DATA__;
const vb = {w: 540, h: 225, l: 46, r: 12, t: 14, b: 40};
const ix = t => vb.l + (t - 0.05) / 0.90 * (vb.w - vb.l - vb.r);
const iy = v => vb.t + (1 - v) * (vb.h - vb.t - vb.b);
const pct = x => (x * 100).toFixed(1) + '%';
const num = x => x.toLocaleString('en-US');
const wan = x => x.toLocaleString('en-US', {maximumFractionDigits: 0}) + ' 万';
const line = key => D.grid.map(g => ix(g.t).toFixed(1) + ',' + iy(g[key]).toFixed(1)).join(' ');

function axes() {
  let s = '';
  [0, 0.25, 0.5, 0.75, 1].forEach(v => {
    const yy = iy(v);
    s += `<line x1="${vb.l}" y1="${yy}" x2="${vb.w - vb.r}" y2="${yy}" stroke="#e8ecf1"/>`;
    s += `<text x="${vb.l - 8}" y="${yy + 4}" font-size="11" fill="#5b6470" text-anchor="end">${v.toFixed(2)}</text>`;
  });
  D.grid.forEach((g, i) => {
    if (i % 2) return;
    const xx = ix(g.t);
    s += `<line x1="${xx}" y1="${vb.t}" x2="${xx}" y2="${vb.h - vb.b}" stroke="#f1f4f7"/>`;
    s += `<text x="${xx}" y="${vb.h - vb.b + 16}" font-size="11" fill="#5b6470" text-anchor="middle">${g.t.toFixed(2)}</text>`;
  });
  s += `<text x="${(vb.l + vb.w - vb.r) / 2}" y="${vb.h - 6}" font-size="12" fill="#5b6470" text-anchor="middle">审批阈值（模型报的概率 ≥ 阈值即拒）</text>`;
  return s;
}

// 金额账：把「回收率 / 净息差」两个业务参数从数据里提出来，拖动时只做一次乘加
function money(g, rec, mar) {
  return {
    loans: g.loans_rej,
    avoided: (1 - rec) * g.avoided_base,
    foregone: mar * g.foregone_base,
    gain: (1 - rec) * g.avoided_base - mar * g.foregone_base,
  };
}

const MG = {w: 540, h: 196, l: 50, r: 12, t: 14, b: 38};
const mgx = t => MG.l + (t - 0.05) / 0.90 * (MG.w - MG.l - MG.r);

function drawMoney(rec, mar, cur) {
  const vals = D.grid.map(g => money(g, rec, mar).gain);
  let hi = Math.max(...vals), lo = Math.min(...vals);
  const pad = (hi - lo) * 0.12 || 1;
  hi += pad; lo = Math.min(lo - pad, 0);
  const mgy = v => MG.t + (1 - (v - lo) / (hi - lo)) * (MG.h - MG.t - MG.b);
  let s = '';
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = mgy(v);
    s += `<line x1="${MG.l}" y1="${y.toFixed(1)}" x2="${MG.w - MG.r}" y2="${y.toFixed(1)}" stroke="#e8ecf1"/>`;
    s += `<text x="${MG.l - 8}" y="${(y + 4).toFixed(1)}" font-size="11" fill="#5b6470" text-anchor="end">${num(Math.round(v))}</text>`;
  }
  s += `<line x1="${MG.l}" y1="${mgy(0).toFixed(1)}" x2="${MG.w - MG.r}" y2="${mgy(0).toFixed(1)}" stroke="#9aa6b4" stroke-dasharray="4 4"/>`;
  s += `<text x="${MG.w - MG.r}" y="${(mgy(0) - 6).toFixed(1)}" font-size="11" fill="#5b6470" text-anchor="end">0（不赚不亏）</text>`;
  D.grid.forEach((g, i) => {
    if (i % 2) return;
    const x = mgx(g.t);
    s += `<line x1="${x.toFixed(1)}" y1="${MG.t}" x2="${x.toFixed(1)}" y2="${MG.h - MG.b}" stroke="#f1f4f7"/>`;
    s += `<text x="${x.toFixed(1)}" y="${MG.h - MG.b + 16}" font-size="11" fill="#5b6470" text-anchor="middle">${g.t.toFixed(2)}</text>`;
  });
  s += `<text x="${((MG.l + MG.w - MG.r) / 2).toFixed(0)}" y="${MG.h - 6}" font-size="12" fill="#5b6470" text-anchor="middle">审批阈值（模型报的概率 ≥ 阈值即拒）</text>`;
  document.getElementById('mGrid').innerHTML = s;
  document.getElementById('mLine').setAttribute('points',
    D.grid.map((g, i) => mgx(g.t).toFixed(1) + ',' + mgy(vals[i]).toFixed(1)).join(' '));
  document.getElementById('mPts').innerHTML =
    D.grid.map((g, i) => `<circle cx="${mgx(g.t).toFixed(1)}" cy="${mgy(vals[i]).toFixed(1)}" r="2.8" fill="#1e8449"/>`).join('');
  const bi = vals.indexOf(Math.max(...vals));
  document.getElementById('mBest').setAttribute('cx', mgx(D.grid[bi].t).toFixed(1));
  document.getElementById('mBest').setAttribute('cy', mgy(vals[bi]).toFixed(1));
  document.getElementById('mMarker').setAttribute('x1', mgx(cur.t).toFixed(1));
  document.getElementById('mMarker').setAttribute('x2', mgx(cur.t).toFixed(1));
  const curGain = money(cur, rec, mar).gain;
  document.getElementById('mTip').textContent =
    `当前阈值 ${cur.t.toFixed(2)}：净增益 ${wan(curGain)}元（全批放款 = 0）｜`
    + `最优 ${D.grid[bi].t.toFixed(2)}：净增益 ${wan(vals[bi])}元`;
  return {best: D.grid[bi], bestGain: vals[bi], curGain: curGain};
}

function adviceText(g, m, info) {
  let tone, text;
  if (g.t <= 0.35) {
    tone = '宽进档：抓得全，但误伤多';
    text = '阈值偏低，几乎把违约者一网打尽（召回率高），代价是拒掉大量正常客户。适合获客扩张期或风险容忍度高的场景，坏账损失会上升。';
  } else if (g.t <= 0.65) {
    tone = '平衡档：日常风控常规档';
    text = '在"抓违约"和"误伤好客户"之间取折中，是日常风控的常规档位，也是这套模型（AUC 0.7249）下比较稳的区间。';
  } else {
    tone = '严审档：批得准，但放走的多';
    text = '被拒的人里违约者比例高（精确率高、误伤少），但会漏掉大量违约者、放走的业务量多。适合高风险客群或逾期压力大的收缩期。';
  }
  if (info.bestGain <= 0) {
    return `<span class="tone">当前参数下怎么设都是亏</span>`
      + `把回收率 / 净息差拖成现在这样之后，<b>任何阈值</b>算出来的净增益都不超过 0`
      + `（最好也只有 ${wan(info.bestGain)}元），问题出在参数假设本身，不在阈值。`
      + `先把回收率、净息差换成机构的真实值，再回头看阈值建议。`;
  }
  const first = Math.abs(info.best.t - g.t) < 1e-9
    ? '你现在拖到的就是它。'
    : `当前 ${g.t.toFixed(2)} 的净增益比它${info.bestGain >= m.gain ? '少' : '多'} <b>${wan(Math.abs(info.bestGain - m.gain))}元</b>。`;
  return `<span class="tone">${tone}</span>${text}`
    + `<div class="pick"><b>只挑一条审批线，建议设 ${info.best.t.toFixed(2)}：</b>`
    + `拒绝 ${pct(info.best.reject_rate)} 的申请人，净增益 <b>${wan(info.bestGain)}元</b>。${first}</div>`;
}

// 页签④：回收率 × 净息差 → 逐笔算账 vs 最优单一线（数据在生成阶段算好，这里只查表画图）
const EGB = {w: 900, h: 260, l: 80, r: 28, t: 16, b: 42};
const GGB = {w: 900, h: 190, l: 80, r: 28, t: 18, b: 36};
const egx = v => EGB.l + v / 0.20 * (EGB.w - EGB.l - EGB.r);
const nearest = (arr, v) => arr.reduce((b, x, i) =>
  Math.abs(x - v) < Math.abs(arr[b] - v) ? i : b, 0);

function eRefresh() {
  const rec = parseFloat(document.getElementById('eRec').value);
  const mar = parseFloat(document.getElementById('eMar').value);
  const ri = nearest(D.epv.recs, rec), mi = nearest(D.epv.mars, mar);
  const ep = D.epv.ep[ri], th = D.epv.th[ri], tv = D.epv.t[ri], mars = D.epv.mars;
  const hi = Math.max(...ep, ...th) * 1.10;
  const lo = Math.min(0, ...ep, ...th);
  const egy = v => EGB.t + (1 - (v - lo) / (hi - lo)) * (EGB.h - EGB.t - EGB.b);
  let s = '';
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = egy(v);
    s += `<line x1="${EGB.l}" y1="${y.toFixed(1)}" x2="${EGB.w - EGB.r}" y2="${y.toFixed(1)}" stroke="#e8ecf1"/>`;
    s += `<text x="${EGB.l - 8}" y="${(y + 4).toFixed(1)}" font-size="11" fill="#5b6470" text-anchor="end">${num(Math.round(v))}</text>`;
  }
  s += `<line x1="${EGB.l}" y1="${egy(0).toFixed(1)}" x2="${EGB.w - EGB.r}" y2="${egy(0).toFixed(1)}" stroke="#9aa6b4" stroke-dasharray="4 4"/>`;
  mars.forEach((v, j) => {
    if (j % 2) return;
    const x = egx(v);
    s += `<line x1="${x.toFixed(1)}" y1="${EGB.t}" x2="${x.toFixed(1)}" y2="${EGB.h - EGB.b}" stroke="#f1f4f7"/>`;
    s += `<text x="${x.toFixed(1)}" y="${EGB.h - EGB.b + 16}" font-size="11" fill="#5b6470" text-anchor="middle">${(v * 100).toFixed(0)}%</text>`;
  });
  s += `<text x="${((EGB.l + EGB.w - EGB.r) / 2).toFixed(0)}" y="${EGB.h - 6}" font-size="12" fill="#5b6470" text-anchor="middle">年净息差</text>`;
  s += `<text x="16" y="${EGB.t + 8}" font-size="12" fill="#5b6470">收益（万元）</text>`;
  document.getElementById('eGrid').innerHTML = s;
  const pts = a => mars.map((v, j) => egx(v).toFixed(1) + ',' + egy(a[j]).toFixed(1)).join(' ');
  document.getElementById('eLineEp').setAttribute('points', pts(ep));
  document.getElementById('eLineTh').setAttribute('points', pts(th));
  const mx = egx(mars[mi]).toFixed(1);
  ['x1', 'x2'].forEach(k => document.getElementById('eMarker').setAttribute(k, mx));
  document.getElementById('eRecVal').textContent = (D.epv.recs[ri] * 100).toFixed(0) + '%';
  document.getElementById('eMarVal').textContent = (mars[mi] * 100).toFixed(0) + '%';
  const sv = D.sens[ri][mi];
  document.getElementById('eImp').textContent =
    (sv[0] * 100).toFixed(1) + '% ／ ' + (sv[1] * 100).toFixed(1) + '%';
  document.getElementById('eRej').textContent = (sv[2] * 100).toFixed(1) + '%';
  document.getElementById('eRec2').textContent = (sv[3] * 100).toFixed(1) + '%';
  document.getElementById('ePre').textContent = (sv[4] * 100).toFixed(1) + '%';
  // 下面的敏感性表里，与滑块当前档位一致的那一行跟着亮
  document.querySelectorAll('#tblSens tbody tr').forEach(r => {
    const rm = r.dataset.mar, rr = r.dataset.rec;
    r.classList.toggle('sel', !!rm
      && Math.abs(parseFloat(rm) - mars[mi]) < 0.005
      && Math.abs(parseFloat(rr) - D.epv.recs[ri]) < 0.026);
  });
  document.getElementById('eEp').textContent = num(Math.round(ep[mi]));
  document.getElementById('eTh').textContent = num(Math.round(th[mi]));
  document.getElementById('eGap').textContent = num(Math.round(ep[mi] - th[mi]));
  document.getElementById('eT').textContent = tv[mi].toFixed(3);
  const gapv = ep.map((v, j) => v - th[j]);
  const ghi = Math.max(...gapv) * 1.18 || 1;
  const ggx = v => GGB.l + v / 0.20 * (GGB.w - GGB.l - GGB.r);
  const ggy = v => GGB.t + (1 - v / ghi) * (GGB.h - GGB.t - GGB.b);
  let s2 = '';
  for (let k = 0; k <= 3; k++) {
    const v = ghi * k / 3, y = ggy(v);
    s2 += `<line x1="${GGB.l}" y1="${y.toFixed(1)}" x2="${GGB.w - GGB.r}" y2="${y.toFixed(1)}" stroke="#e8ecf1"/>`;
    s2 += `<text x="${GGB.l - 8}" y="${(y + 4).toFixed(1)}" font-size="11" fill="#5b6470" text-anchor="end">${num(Math.round(v))}</text>`;
  }
  mars.forEach((v, j) => {
    if (j % 2) return;
    const x = ggx(v);
    s2 += `<line x1="${x.toFixed(1)}" y1="${GGB.t}" x2="${x.toFixed(1)}" y2="${GGB.h - GGB.b}" stroke="#f1f4f7"/>`;
    s2 += `<text x="${x.toFixed(1)}" y="${GGB.h - GGB.b + 16}" font-size="11" fill="#5b6470" text-anchor="middle">${(v * 100).toFixed(0)}%</text>`;
  });
  s2 += `<text x="${((GGB.l + GGB.w - GGB.r) / 2).toFixed(0)}" y="${GGB.h - 6}" font-size="12" fill="#5b6470" text-anchor="middle">年净息差</text>`;
  s2 += `<text x="16" y="${GGB.t + 8}" font-size="12" fill="#5b6470">逐笔 − 单一线（万元）</text>`;
  document.getElementById('eGapGrid').innerHTML = s2;
  document.getElementById('eGapLine').setAttribute('points',
    mars.map((v, j) => ggx(v).toFixed(1) + ',' + ggy(gapv[j]).toFixed(1)).join(' '));
  const gmx = ggx(mars[mi]).toFixed(1);
  const gmk = document.getElementById('eGapMarker');
  gmk.setAttribute('x1', gmx); gmk.setAttribute('x2', gmx);
  gmk.setAttribute('y1', GGB.t); gmk.setAttribute('y2', GGB.h - GGB.b);
  const glb = document.getElementById('eGapLabel');
  glb.setAttribute('x', (Number(gmx) + 6).toFixed(1));
  glb.setAttribute('y', Math.max(GGB.t + 10, ggy(gapv[mi]) - 6).toFixed(1));
  glb.textContent = num(Math.round(gapv[mi])) + ' 万元';
  document.getElementById('eNote').innerHTML =
    `回收率 <b>${(D.epv.recs[ri] * 100).toFixed(0)}%</b>、净息差 <b>${(mars[mi] * 100).toFixed(0)}%</b>：`
    + `逐笔算账 <b>${wan(ep[mi])}元</b>，最优单一线 <b>${wan(th[mi])}元</b>（t*=${tv[mi].toFixed(3)}），`
    + `逐笔多赚 <b>${wan(ep[mi] - th[mi])}元</b>。`;
}

function refresh() {
  const t = parseFloat(document.getElementById('thr').value);
  const rec = parseFloat(document.getElementById('rec').value);
  const mar = parseFloat(document.getElementById('mar').value);
  const g = D.grid.reduce((a, b) => Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a);
  const m = money(g, rec, mar);
  document.getElementById('thrVal').textContent = g.t.toFixed(2);
  document.getElementById('recVal').textContent = (rec * 100).toFixed(0) + '%';
  document.getElementById('marVal').textContent = (mar * 100).toFixed(0) + '%';
  document.getElementById('kReject').textContent = pct(g.reject_rate);
  document.getElementById('kRejectN').textContent = num(g.tp + g.fp) + ' 人被拒';
  document.getElementById('kPrec').textContent = pct(g.precision);
  document.getElementById('kPrecN').textContent = '被拒者中真违约占比';
  document.getElementById('kRec').textContent = pct(g.recall);
  document.getElementById('kRecN').textContent = '抓到 ' + num(g.tp) + ' / ' + num(D.n_pos) + ' 名违约者';
  document.getElementById('kF1').textContent = g.f1.toFixed(3);
  document.getElementById('kAuc').textContent = D.metrics.auc.toFixed(4);
  document.getElementById('kAucN').textContent = '全阈值下排序能力（不随阈值变）';
  document.getElementById('kLoans').textContent = wan(m.loans);
  document.getElementById('kAvoided').textContent = wan(m.avoided);
  document.getElementById('kForegone').textContent = wan(m.foregone);
  document.getElementById('kGain').textContent = wan(m.gain);
  document.getElementById('cm-tn').textContent = num(g.tn);
  document.getElementById('cm-fp').textContent = num(g.fp);
  document.getElementById('cm-fn').textContent = num(g.fn);
  document.getElementById('cm-tp').textContent = num(g.tp);
  document.getElementById('ledger').innerHTML =
    `按阈值 <b>${g.t.toFixed(2)}</b> 审批：拒绝 <b>${pct(g.reject_rate)}</b> 的申请人（${num(g.tp + g.fp)} 人），`
    + `其中真违约 <b>${num(g.tp)}</b> 人、误伤正常客户 <b>${num(g.fp)}</b> 人；`
    + `全部 ${num(D.n_pos)} 名违约者中拦下 <b>${pct(g.recall)}</b>，漏放 ${num(g.fn)} 人；`
    + `少放贷款 <b>${wan(m.loans)}元</b>，避免坏账 <b>${wan(m.avoided)}元</b>、`
    + `放弃收入 <b>${wan(m.foregone)}元</b>，两者相抵 <b>${m.gain >= 0 ? '净赚' : '净亏'} ${wan(Math.abs(m.gain))}元</b>。`;
  const x = ix(g.t);
  document.getElementById('marker').setAttribute('x1', x);
  document.getElementById('marker').setAttribute('x2', x);
  document.querySelectorAll('#ptPrec circle').forEach((c, i) => {
    c.setAttribute('cy', iy(D.grid[i].precision)); c.setAttribute('cx', ix(D.grid[i].t)); });
  document.getElementById('tip').textContent =
    `阈值 ${g.t.toFixed(2)}｜精确率 ${pct(g.precision)}｜召回率 ${pct(g.recall)}｜F1 ${g.f1.toFixed(3)}`;
  const info = drawMoney(rec, mar, g);
  document.getElementById('advice').innerHTML = adviceText(g, m, info);
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('grid').innerHTML = axes();
  document.getElementById('lnPrec').setAttribute('points', line('precision'));
  document.getElementById('lnRec').setAttribute('points', line('recall'));
  document.getElementById('lnF1').setAttribute('points', line('f1'));
  document.getElementById('ptPrec').innerHTML =
    D.grid.map(g => `<circle cx="${ix(g.t).toFixed(1)}" cy="${iy(g.precision).toFixed(1)}" r="2.8" fill="#1f5fbf"/>`).join('');
  document.getElementById('thr').addEventListener('input', refresh);
  document.getElementById('rec').addEventListener('input', refresh);
  document.getElementById('mar').addEventListener('input', refresh);
  document.getElementById('eRec').addEventListener('input', eRefresh);
  document.getElementById('eMar').addEventListener('input', eRefresh);
  eRefresh();
  refresh();

  // ---------- 图表悬浮读数 ----------
  const tipEl = document.createElement('div');
  tipEl.className = 'ctip';
  const toView = (svg, ev) => {
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  };
  function bindHover(svg, fn) {
    if (!svg) return;
    const box = svg.parentElement;
    box.style.position = 'relative';
    svg.addEventListener('mousemove', ev => {
      const html = fn(toView(svg, ev));
      if (!html) { tipEl.classList.remove('on'); return; }
      box.appendChild(tipEl);
      tipEl.innerHTML = html;
      const r = box.getBoundingClientRect();
      let x = ev.clientX - r.left + 14, y = ev.clientY - r.top - 8;
      if (x + tipEl.offsetWidth > r.width) x = ev.clientX - r.left - tipEl.offsetWidth - 14;
      tipEl.style.left = Math.max(0, x) + 'px';
      tipEl.style.top = Math.max(0, y) + 'px';
      tipEl.classList.add('on');
    });
    svg.addEventListener('mouseleave', () => tipEl.classList.remove('on'));
  }
  const gAt = t => D.grid.reduce((a, b) => Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a);
  const moneyHtml = g => {
    const rec = parseFloat(document.getElementById('rec').value);
    const mar = parseFloat(document.getElementById('mar').value);
    const m = money(g, rec, mar);
    return `阈值 ${g.t.toFixed(2)} ｜ 净增益 <b>${num(Math.round(m.gain))}</b> 万元` +
      `<br>被拒 ${num(g.tp + g.fp)} 人 ｜ 避免坏账 ${num(Math.round(m.avoided))} 万 ｜ 放弃收入 ${num(Math.round(m.foregone))} 万`;
  };
  // ③ 阈值—指标曲线
  bindHover(document.getElementById('lnPrec').ownerSVGElement, v => {
    const t = 0.05 + (v.x - vb.l) / (vb.w - vb.l - vb.r) * 0.90;
    if (t < 0.05 || t > 0.95) return '';
    const g = gAt(t);
    return `阈值 ${g.t.toFixed(2)}<br>精确率 ${pct(g.precision)} ｜ 召回率 ${pct(g.recall)} ｜ F1 ${pct(g.f1)}` +
      `<br>被拒 ${num(g.tp + g.fp)} 人（其中真违约 ${num(g.tp)}）`;
  });
  // ③ 净增益曲线
  bindHover(document.getElementById('mLine').ownerSVGElement, v => {
    const t = 0.05 + (v.x - MG.l) / (MG.w - MG.l - MG.r) * 0.90;
    if (t < 0.05 || t > 0.95) return '';
    return moneyHtml(gAt(t));
  });
  // ④ 两种做法的收益曲线
  const epvHtml = v => {
    const mar = (v.x - EGB.l) / (EGB.w - EGB.l - EGB.r) * 0.20;
    if (mar < 0 || mar > 0.20) return '';
    const ri = nearest(D.epv.recs, parseFloat(document.getElementById('eRec').value));
    const mi = nearest(D.epv.mars, mar);
    return `年净息差 ${(D.epv.mars[mi] * 100).toFixed(0)}%（回收率 ${(D.epv.recs[ri] * 100).toFixed(0)}%）` +
      `<br>逐笔算账 <b>${num(Math.round(D.epv.ep[ri][mi]))}</b> 万元 ｜ 最优单一线 ${num(Math.round(D.epv.th[ri][mi]))} 万元` +
      `<br>逐笔多赚 ${num(Math.round(D.epv.ep[ri][mi] - D.epv.th[ri][mi]))} 万元`;
  };
  bindHover(document.getElementById('eLineEp').ownerSVGElement, epvHtml);
  bindHover(document.getElementById('eGapLine').ownerSVGElement, v => {
    const mar = (v.x - GGB.l) / (GGB.w - GGB.l - GGB.r) * 0.20;
    if (mar < 0 || mar > 0.20) return '';
    const ri = nearest(D.epv.recs, parseFloat(document.getElementById('eRec').value));
    const mi = nearest(D.epv.mars, mar);
    const gap = D.epv.ep[ri][mi] - D.epv.th[ri][mi];
    return `年净息差 ${(D.epv.mars[mi] * 100).toFixed(0)}% ｜ 逐笔比单线多赚 <b>${num(Math.round(gap))}</b> 万元`;
  });

  // ---------- 图片点击放大 ----------
  const zmBox = document.getElementById('zoom'), zmImg = document.getElementById('zimg');
  document.querySelectorAll('figure img').forEach(im => im.addEventListener('click', () => {
    zmImg.src = im.src; zmBox.classList.add('on');
  }));
  zmBox.addEventListener('click', () => zmBox.classList.remove('on'));
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { zmBox.classList.remove('on'); tipEl.classList.remove('on'); }
  });

  // ---------- ①② 内联 SVG 图：逐点读数（统一的 data-tip 悬浮层）----------
  const inlineTip = document.createElement('div');
  inlineTip.className = 'ctip fl';
  document.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('mouseenter', ev => {
      document.body.appendChild(inlineTip);
      inlineTip.innerHTML = el.dataset.tip.split('[[BR]]').join('<br>');
      inlineTip.classList.add('on');
    });
    el.addEventListener('mousemove', ev => {
      const w = inlineTip.offsetWidth, h = inlineTip.offsetHeight;
      let x = ev.clientX + 14, y = ev.clientY - h - 10;
      if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
      if (x < 8) x = 8;
      if (y < 6) y = ev.clientY + 16;
      inlineTip.style.left = x + 'px';
      inlineTip.style.top = y + 'px';
    });
    el.addEventListener('mouseleave', () => inlineTip.classList.remove('on'));
  });

  // ---------- 说明折叠：一键展开 / 收起；打印前全部展开 ----------
  const btn = document.getElementById('foldAll');
  btn.addEventListener('click', () => {
    const ds = [...document.querySelectorAll('details.fold')];
    const open = ds.some(d => !d.open);
    ds.forEach(d => d.open = open);
    btn.textContent = open ? '收起全部说明' : '展开全部说明';
  });
  window.addEventListener('beforeprint', () => {
    document.querySelectorAll('details.fold').forEach(d => d.open = true);
  });
});
"""

CHART_SVG = """
<svg viewBox="0 0 540 225" preserveAspectRatio="xMidYMid meet" role="img" aria-label="阈值—指标曲线">
  <g id="grid"></g>
  <polyline id="lnPrec" fill="none" stroke="#1f5fbf" stroke-width="2.4" points="__DEF_PT_PREC__"/>
  <polyline id="lnRec" fill="none" stroke="#c0392b" stroke-width="2.4" points="__DEF_PT_REC__"/>
  <polyline id="lnF1" fill="none" stroke="#1e8449" stroke-width="2.4" stroke-dasharray="6 4" points="__DEF_PT_F1__"/>
  <g id="ptPrec">__DEF_CIRCLES__</g>
  <line id="marker" x1="__DEF_MX__" y1="14" x2="__DEF_MX__" y2="185" stroke="#1b1f24" stroke-width="1.4" stroke-dasharray="4 4"/>
</svg>
<div class="lg">
  <span><i style="background:#1f5fbf"></i>精确率（被拒者里真违约的比例）</span>
  <span><i style="background:#c0392b"></i>召回率（违约者被拦下的比例）</span>
  <span><i style="background:#1e8449"></i>F1（两者平衡）</span>
  <span id="tip">__DEF_TIP__</span>
</div>
"""

MONEY_SVG = """
<svg viewBox="0 0 540 196" preserveAspectRatio="xMidYMid meet" role="img" aria-label="各阈值下的净增益曲线">
  <g id="mGrid">__DEF_G_GRID__</g>
  <line id="mZero" x1="80" y1="0" x2="872" y2="0" stroke="#9aa6b4" stroke-dasharray="4 4" style="display:none"/>
  <polyline id="mLine" fill="none" stroke="#1e8449" stroke-width="2.6" points="__DEF_G_POINTS__"/>
  <g id="mPts">__DEF_G_PTS__</g>
  <circle id="mBest" cx="__DEF_G_BEST_X__" cy="__DEF_G_BEST_Y__" r="5" fill="#fff" stroke="#c0392b" stroke-width="2.6"/>
  <line id="mMarker" x1="__DEF_G_MX__" y1="14" x2="__DEF_G_MX__" y2="158" stroke="#1b1f24" stroke-width="1.4" stroke-dasharray="4 4"/>
</svg>
<div class="lg">
  <span><i style="background:#1e8449"></i>净增益 = 避免的坏账 − 放弃的利息收入（相对"全批放款"）</span>
  <span><i style="background:#c0392b"></i>净增益最高的一档（曲线最高点）</span>
  <span id="mTip"></span>
</div>
"""

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>信贷违约预测 · 业务仪表板</title>
<style>__CSS__</style>
</head>
<body>
<input class="tabin" type="radio" name="tab" id="t1" checked>
<input class="tabin" type="radio" name="tab" id="t2">
<input class="tabin" type="radio" name="tab" id="t3">
<input class="tabin" type="radio" name="tab" id="t4">
<header>
  <h1>信贷违约预测 · 业务仪表板</h1>
  <p class="sub">阿里线上数据分析实习项目（W1–W4）｜天池信贷违约数据集：一行一笔贷款申请，train 80 万行有标签、testA 20 万行无标签。</p>
  <div class="cards hero">
    <div class="kpi"><div class="k">建议审批线（按期限拆两条）</div><div class="v">3 年 __RAW3__ ／ 5 年 __RAW5__</div><div class="n">原始概率轴（模型直接输出）。只能设一条线就用 __T55__</div></div>
    <div class="kpi"><div class="k">中性口径期望利润</div><div class="v">__TWO_M__ 万元</div><div class="n">拒 __TWO_REJ_N__ 人（__TWO_REJ_PCT__）｜比最优单线多 267 万元</div></div>
    <div class="kpi"><div class="k">测试集 AUC</div><div class="v">__AUC__</div><div class="n">LightGBM（W3 调优）｜79,850 人，从未参与训练</div></div>
    <div class="kpi"><div class="k">全放不筛人</div><div class="v">2,085 万元</div><div class="n">同一口径下的零策略参照</div></div>
  </div>
  <details class="fold"><summary><span class="ftag">口径</span>数据范围、模型、金额口径；审批线怎么读、两种做法差在哪</summary>
    <div class="facts">
      <div><b>数据</b>天池信贷违约数据集，一行一笔贷款；train 80 万行有标签，testA 20 万行没有</div>
      <div><b>口径</b>只用 train（清洗后 798,498 行，7:2:1 切分）；成绩和金额都在这批测试集 79,850 人上算</div>
      <div><b>模型</b>LightGBM（W3 调优），测试集 AUC __AUC__；净息差 6%、回收率 30% 为模拟值</div>
    </div>
    <div class="keybox">
      <p><b>审批线</b>就是一条概率门槛：达到线的人拒贷，低于线的放贷。落地有两种做法：</p>
      <ul class="ways">
        <li><b>方案①（页签③）一条线管所有人</b>——只依赖排序，简单，也是兜底；拖一下滑块就能看钱随线怎么变。</li>
        <li><b>方案②（页签④）按每笔贷款的账定线</b>——正常还完赚息差、违约亏本金，按概率加权，期望为正就放；本批只有 3、5 年期，落地成两条线，贴着线的一段转人工复审。</li>
      </ul>
      <p class="tail">Ctrl/⌘+P 另存 PDF；本文件离线，双击即开。</p>
    </div>
  </details>
</header>

<nav>
  <label class="tab" for="t1">① 数据概览</label>
  <label class="tab" for="t2">② 模型表现</label>
  <label class="tab" for="t3">③ 放贷方案① · 单一线（可拖动）</label>
  <span class="navsep"></span>
  <label class="tab" for="t4">④ 放贷方案② · 拆线：按期限设两条线</label>
</nav>

<main>
  <div class="tools">
    <button id="foldAll" type="button">展开全部说明</button>
    <span class="hint" style="margin:0">图都能悬浮读数，③④ 跟着滑块走。</span>
  </div>
  <section id="p1" class="panel">
    <div class="statline">
      <div><div class="k">建模样本</div><div class="v">798,498 行</div><div class="n">来自 train 80 万，清洗删 0.19%</div></div>
      <div><div class="k">原始字段</div><div class="v">47 个</div><div class="n">加工后 38 个进模型，见页签②</div></div>
      <div><div class="k">训练集违约率</div><div class="v">19.96%</div><div class="n">类别不平衡，模型做了加权</div></div>
      <div><div class="k">数据划分</div><div class="v">7:2:1</div><div class="n">558,948 / 159,700 / 79,850</div></div>
    </div>
    <div class="grid2 tiles">
      <div class="tile">
        <h2>目标变量：是否违约</h2>
    __FIG_TARGET__
      </div>
      <div class="tile">
        <h2>信用等级是最强的分层变量</h2>
    __FIG_GRADE__
      </div>
      <div class="tile wide">
        <h2>核心数值变量的分布</h2>
    __FIG_DIST__
      </div>
      <div class="tile wide">
        <h2>违约组与正常组的差异</h2>
    __FIG_CMP__
      </div>
      <div class="tile">
        <h2>相关性总览</h2>
    __FIG_CORR__
      </div>
      <div class="tile">
        <h2>年收入右偏与 log 变换</h2>
    __FIG_INCOME__
      </div>
    </div>
    <details class="fold"><summary><span class="ftag">口径</span>本页是 W1 口径：六张图都在 47 个原始字段上做，字段 n0–n14 是匿名计数特征</summary>
      <p>本页六张图都在加工前的原始字段上做，还没进特征工程；加工之后的 38 个建模特征见页签②。</p>
      <p>字段 n0–n14 为数据集提供方给出的匿名计数特征，官方未公布具体含义，本项目只做统计处理、不猜测其业务含义。</p>
    </details>
  </section>

  <section id="p2" class="panel">
    <h2>测试集成绩单（阈值 0.50）</h2>
    <p class="sub">模型输入 <b>38 个特征</b>（由 47 个原始字段加工而来），特征数量实验见本页最后一节。</p>
    <div class="cards">
      <div class="kpi"><div class="k">AUC</div><div class="v">__AUC__</div><div class="n">排序能力：随机抽一违约一正常，模型把违约者排前面的概率</div></div>
      <div class="kpi"><div class="k">KS</div><div class="v">__KS__</div><div class="n">违约者与正常人的分数分布拉开的最大距离，风控常用的门槛是 0.30</div></div>
      <div class="kpi"><div class="k">准确率</div><div class="v">__ACC__</div><div class="n">整体判断正确比例</div></div>
      <div class="kpi"><div class="k">精确率</div><div class="v">__PRE__</div><div class="n">被拒的人里真违约的比例</div></div>
      <div class="kpi"><div class="k">召回率</div><div class="v">__REC__</div><div class="n">违约者被拦住的比例</div></div>
      <div class="kpi"><div class="k">F1</div><div class="v">__F1__</div><div class="n">精确率与召回率的平衡</div></div>
    </div>
    <h3>头部区分度：KS 与头部捕获率</h3>
    <p class="sub">AUC 在整批人上算平均，看不出头部的表现。审批更关心名单头部：分数最高的那 10% 里抓到多少违约者。
       下表是四个头部位置的表现，<b>捕获率</b>指这批人里的违约者占全部违约者的比例，<b>头部内违约率</b>指这批人自己的违约比例。</p>
    <details class="fold"><summary><span class="ftag">数据表</span>分数最高的 1% / 5% / 10% / 20% 各抓到多少违约者</summary>
        __TBL_HEAD__</details>
    <p class="sub">KS = <b>__KS__</b>，指违约者和正常人的分数分布拉开的最大距离，风控常用的门槛是 0.30。
       这两个指标都只看排序，与概率刻度无关，校准前后一样。</p>
    <h3>泛化能力：训练 / 验证 / 测试三集对比</h3>
    <p class="sub">三个集合的成绩很接近，说明没有过拟合，模型在没见过的人身上同样有效。</p>
    <details class="fold"><summary><span class="ftag">数据表</span>训练 / 验证 / 测试三集的六项指标</summary>
        __TBL_GEN__</details>
    <h3>模型性能图：ROC、PR、混淆矩阵、学习曲线</h3>
    <div class="grid2 tiles">
      <div class="tile">__FIG_ROC__</div>
      <div class="tile">__FIG_PR__</div>
      <div class="tile">__FIG_CM__</div>
      <div class="tile">__FIG_LEARN__</div>
    </div>
    <h3>时间外验证：拿过去训练，给未来打分</h3>
    <p class="sub">上面的成绩都出自随机切分：同一批放款随机分到训练和测试，两边行情一样，成绩会偏乐观。
       真实上线只能用过去的数据训练、给未来上门的客户打分。</p>
    <details class="fold"><summary><span class="ftag">口径</span>训练用过去、测试用未来，为什么必须这么切</summary>
        <p class="sub">按放款日期切一次：训练用 2016 年底之前的 __OOT_NTR__ 笔，测试用 2017-01 到 2018-06 的 __OOT_NTE__ 笔。
       2018-07 之后的 8,980 笔没放进来，那段违约率只有 7.5%，是真变好还是违约没暴露，从数据里分不清。</p></details>
    <details class="fold"><summary><span class="ftag">数据表</span>随机切分与按日期切分的成绩对照</summary>
        __TBL_OOT__</details>
    <p class="sub">排序能力掉了一点，没伤到根：AUC 从 __RND_AUC__ 到 __OOT_AUC__，前 10% 捕获率从 __RND_CAP10__ 到 __OOT_CAP10__，
       仍远高于随机挑的 10%。</p>
    <details class="fold"><summary><span class="ftag">读数</span>时间外这段的行情偏移与阈值漂移</summary>
        <p class="sub">更该留意行情那一层：时间外这段的违约率是 __OOT_DR__，比随机切分的 __RND_DR__ 高 __DR_GAP__ 个百分点，
       同一个阈值在新数据上拒的人会多 __REJ_GAP__ 个百分点。④ 拿概率数值算钱，那条概率轴要定期用最近到期的贷款重新标定。</p></details>
    <h3>调参：两个模型 × 三种方法的等预算对照</h3>
    <p class="sub">调参的分数在训练集内部算，跟下面那张用验证集出的选型表不是一个口径，不能横比。正式定参用随机搜索：同一个模型换三种方法，最好成绩极差不到 0.0002。</p>
    __FIG_TUNING__
    <h3>模型对比（验证集，W3 优化后）</h3>
    <details class="fold"><summary><span class="ftag">数据表</span>六个候选模型在验证集上的准确率 / 精确率 / 召回率 / F1 / AUC</summary>
        __TBL_CMP__</details>
    <div class="keybox">
      <p><b>方式、参数、模型怎么定的</b>：</p>
      <ul>
        <li><b>调参方式＝随机搜索</b>（XGB 与 LGB 各 8 组、5 折交叉验证）。依据是上面那组对照：同一个模型换三种方法，最优成绩极差不到 0.0002，
            说明这类小离散空间里没有更多可挖的区域；而 8 组预算下贝叶斯前 5 次是热身期，实测还慢 35%、要多引一个 optuna 依赖，
            随机搜索 sklearn 自带、透明可复现。</li>
        <li><b>最终参数</b>（LightGBM，5 折平均 AUC 0.7249）：num_leaves=63、n_estimators=200、max_depth=8、
            learning_rate=0.05、colsample_bytree=0.7、subsample=0.8。</li>
        <li><b>为什么最终用 LGB（优化），而不是分最高的 Stacking</b>：验证集 AUC 上 Stacking 0.7298 比 LGB 0.7294 高 0.0004，
            统计检验说这个差距是真的，但量级不到万分之五；代价是训练 371 秒 vs 5.6 秒（66 倍）、打分慢一倍，
            还要维护 4 个基模型加 1 个元模型。多付六十几倍成本换 0.0004，不划算。</li>
      </ul>
    </div>
    <h3>特征重要性：随机森林与 SHAP</h3>
    <div class="grid2 tiles">
      <div class="tile">__FIG_IMP__</div>
      <div class="tile">__FIG_SHAP__</div>
    </div>
    <h3>特征数量的边际收益（逐步加特征）</h3>
    <details class="fold"><summary><span class="ftag">数据表</span>Top 3 到 38 个特征的验证集 AUC</summary>
        __TBL_FC__</details>
    __FIG_FC__
  </section>

  <section id="p3" class="panel">
    <h2>放贷方案① · 单一线：把阀门拧到哪，能多赚多少 / 少亏多少</h2>
    <details class="fold"><summary><span class="ftag">定义</span>审批线是什么，两条线怎么来的</summary>
        <p class="sub">阈值就是"审批线"：模型给每个人算一个违约概率，达到这个数就拒贷。调低抓得全、误伤多；调高放得准、漏放多。
       这一页取的阈值是模型直接报出来的未校准概率，页签④ 会把同一个数重新刻度成真实违约率，两种刻度落在同一批人身上，排序一样。
       下面所有数字都在同一个测试集（__N__ 人）上实时计算，金额、期限、是否违约来自真实数据，只有"回收率"和"净息差"是模拟值（真实值得找催收和财务要）。</p></details>
    <p class="sub">下面九张 KPI 分两种：拒绝率、少放贷款只数人数和金额，不碰标签；精确率、召回率、F1、AUC、避免坏账、放弃收入、净增益
       都得拿历史批次里"谁真的违约了"去数，属于回看。能上线、明天来客户就能用的规则在页签④。</p>
    <p class="sub">所以这一页不需要校准：概率在这里只干一件事——排名，金额那几栏用的是答案（谁真违约、谁其实正常）。
       换成"概率 × 金额"算钱是另一条路，p 直接进了算式，才必须先校准，见页签④。</p>
    <h3>这条线是怎么定出来的</h3>
    <div class="keybox">
      <ol>
        <li><b>先看默认值是怎么来的。</b>0.50 是数学默认值：模型说概率过半就拒。它跟业务没关系，
            换一批客群、换一个净息差，它也还是 0.50。</li>
        <li><b>把 0.05 到 0.95 逐档扫一遍。</b>每一档都算一次"避免坏账 − 放弃收入"，画成下面第 ② 条曲线。</li>
        <li><b>取曲线最高点。</b>最高的一档是 <b>__T55__</b>：拒绝 __BEST_REJ_N__ 人（__BEST_REJ_PCT__），
            净增益 <b>__BEST_GAIN__ 万元</b>，比 0.50 多 __BEST_GAIN_DIFF__ 万元。
            曲线按 0.05 步长画；再细扫到 0.001 步长，最优点在 <b>__FINE_T__</b>，净增益 <b>__FINE_GAIN__ 万元</b>，比 0.55 这一档多 __FINE_GAIN_DIFF__ 万元。</li>
      </ol>
      <p class="tail">换一批数据还站得住吗：把挑线这件事搬到验证集上重做一遍，选出 <b>__VAL_T__</b>，
         搬到测试集净增益 <b>__VAL_GAIN__ 万元</b>，与测试集自己扫出的最优 __FINE_GAIN__ 万元差 __VAL_GAP__ 万元（__VAL_GAP_PCT__）；
         0.54 到 0.58 是一段平台区，区间内最低也有峰值的 __PLAT_LOW_PCT__，挑哪个都行。</p>
      <p class="tail">数字只能在这根轴上读：挑线只靠排序，而校准是把概率换个刻度（单调变换），谁排在谁前面不变，
         所以两把刻度上挑出来的是同一条线——原始轴 __FINE_T__ 换算到校准轴是 <b>__C55_FINE__</b>，拒的还是那 __N_FINE_REJ__ 人。
         跨轴直接搬数会出事：把 __FINE_T__ 拿到校准轴上用，只拒 __REJ_XAXIS__ 的人，钱白丢。</p>
    </div>
    <h3>拖动阀门，看每个阈值赚多少、亏多少</h3>
    <div class="card">
      <div class="sliders3">
        <div class="slider-row">
          <span class="sub">审批阈值</span>
          <input id="thr" type="range" min="0.05" max="0.95" step="0.05" value="0.5">
          <span class="thr-val" id="thrVal">0.50</span>
        </div>
        <div class="slider-row">
          <span class="sub">违约回收率</span>
          <input id="rec" type="range" min="0" max="1" step="0.05" value="0.3">
          <span class="thr-val" id="recVal">30%</span>
        </div>
        <div class="slider-row">
          <span class="sub">年净息差</span>
          <input id="mar" type="range" min="0" max="0.2" step="0.01" value="0.06">
          <span class="thr-val" id="marVal">6%</span>
        </div>
      </div>
        <div class="ledger" id="ledger">__DEF_LEDGER__</div>
        <details class="how">
          <summary><span class="ftag">算法</span>净增益这笔账怎么算</summary>
          <p>基准是"不用模型、全批放款"。拒掉一个人有两笔后果：</p>
          <ul>
            <li><b>避免坏账</b> = 贷款金额 ×（1 − 回收率），只在"这个人真的违约"时才算数；</li>
            <li><b>放弃收入</b> = 贷款金额 × 年净息差 × 期限年数，只在"这个人其实是正常客户"时才算数。</li>
          </ul>
          <p><b>净增益 = 避免坏账 − 放弃收入</b>（相对全批放款的增量）。
             阈值定得太低，误伤的正常客户太多，放弃的收入超过避免的坏账，净增益甚至为负；定得太高，
             该拒的违约者没拒，坏账重新爬上来。中间某一点最大，就是最优阈值。</p>
          <details class="fold"><summary><span class="ftag">口径</span>换一种算法，这个数会变</summary>
        <p><b>换一个算法会得到另一个数。</b>不数答案，改成给每笔贷款算
             <code>(1−p)×净息差×期限 − p×(1−回收率)</code>，为正就批，这是页签④ 的口径，
             明天来一个客户也算得出来。两套账的绝对值不一样，但在同一批数据上挑出来的线是同一条：
             净增益最高在 0.55，概率乘金额的期望利润最高在 0.553。同一档位两个数不会完全相等：
             __FINE_T__ 处回看口径比全放多 __FINE_GAIN__ 万元，期望口径比全放多 __FINE_EP_GAIN__ 万元
             （__FINE_M__ − __BASE_M__）；差的 __GAIN_GAP__ 万元来自被拒人群实际违约率 __GAIN_GAP_DR__，
             比模型估的 __GAIN_GAP_PD__ 略低。</p></details>
          <p class="sub">回收率、净息差是模拟值；换成机构的真实值后，这里的金额结论要重算。</p>
        </details>
    </div>
    <div class="grid2">
      <div class="card cht">
        <h4>① 阈值—指标曲线<span class="cht-sub">技术视角：抓得全还是抓得准</span></h4>
        __CHART__
      </div>
      <div class="card cht">
        <h4>② 各阈值下的净增益曲线（万元）<span class="cht-sub">业务视角：阀门拧到哪最赚</span></h4>
        <p class="sub" style="margin:0 0 2px">拖动上面的滑块，两条曲线、最优阈值和下面的建议一起变。</p>
        __MONEY__
      </div>
    </div>
    <div class="sim">
      <div class="card">
        <h4>这一档的读数</h4>
        <div class="cards" style="margin:0">
          <div class="kpi"><div class="k">拒绝率</div><div class="v" id="kReject">__DEF_K_REJECT__</div><div class="n" id="kRejectN">__DEF_K_REJECT_N__</div></div>
          <div class="kpi"><div class="k">精确率</div><div class="v" id="kPrec">__DEF_K_PREC__</div><div class="n" id="kPrecN">被拒者中真违约占比</div></div>
          <div class="kpi"><div class="k">召回率</div><div class="v" id="kRec">__DEF_K_REC__</div><div class="n" id="kRecN">__DEF_K_REC_N__</div></div>
          <div class="kpi"><div class="k">F1</div><div class="v" id="kF1">__DEF_K_F1__</div><div class="n">综合平衡分</div></div>
          <div class="kpi"><div class="k">AUC</div><div class="v" id="kAuc">__DEF_K_AUC__</div><div class="n" id="kAucN">全阈值下排序能力（不随阈值变）</div></div>
          <div class="kpi"><div class="k">少放贷款</div><div class="v" id="kLoans">__DEF_K_LOANS__</div><div class="n">被拒者的贷款金额合计（万元）</div></div>
          <div class="kpi"><div class="k">避免坏账</div><div class="v" id="kAvoided">__DEF_K_AVOIDED__</div><div class="n">被拒者里的真违约者 ×（1−回收率）（万元）</div></div>
          <div class="kpi"><div class="k">放弃收入</div><div class="v" id="kForegone">__DEF_K_FOREGONE__</div><div class="n">被拒者里的正常客户本可赚的息差（万元）</div></div>
          <div class="kpi"><div class="k">净增益</div><div class="v" id="kGain">__DEF_K_GAIN__</div><div class="n" id="kGainN">避免坏账 − 放弃收入，相对全批放款不筛选的增量（万元）</div></div>
        </div>
      </div>
      <div class="card">
        <h4>混淆矩阵（当前阈值）</h4>
        <div class="cm">
          <div class="cm-cell cm-tn"><span class="n" id="cm-tn">__DEF_CM_TN__</span><span class="l">正常 · 批准（正确放款）</span></div>
          <div class="cm-cell cm-fp"><span class="n" id="cm-fp">__DEF_CM_FP__</span><span class="l">正常 · 拒绝（误伤）</span></div>
          <div class="cm-cell cm-fn"><span class="n" id="cm-fn">__DEF_CM_FN__</span><span class="l">违约 · 批准（漏放）</span></div>
          <div class="cm-cell cm-tp"><span class="n" id="cm-tp">__DEF_CM_TP__</span><span class="l">违约 · 拒绝（抓到）</span></div>
        </div>
        <p class="sub">纵轴是真实结果、横轴是模型决定；颜色只用于区分四类结果。</p>
        <div class="advice" id="advice">__DEF_ADVICE__</div>
      </div>
    </div>
    <h3>③ 两条曲线叠在一起看</h3>
    __FIG_SWEEP__
    <h3>预设档位对照表（带钱）</h3>
    __TBL_THR__
    <p class="hint">线上审批的时候没有答案，上面这些回看数字只能用来挑一条线，当不了规则用。
       页签④ 补上这件事：先做概率校准，再给出不依赖答案的逐笔规则。</p>
    <h3>这个做法的三条短板</h3>
    <div class="keybox">
      <ol>
        <li><b>一条线管所有人，金额、期限、利率都不进规则。</b>一笔 5 年期贷款正常还完能赚 5 年净息差，一笔 3 年期的只赚 3 年，
            前者本该容忍更高的违约率。按 __T55__ 这条线，5 年期拒掉 __G5R_REJ__，按逐笔的账只要拒 __G5R_WANT__；
            3 年期反过来，只拒 __G3_REJ__，该拒 __G3_WANT__。</li>
        <li><b>挑线本身要靠答案。</b>"避免坏账"得数出被拒的人里谁真的违约了，只有历史数据做得到。
            换一批客群、换一个净息差，曲线最高点会移动，得重新在旧数据上挑一遍。</li>
        <li><b>它交出一个数字，交不出判断标准。</b>0.55 回答不了"这笔贷款为什么值得放"，也说不出"利率提到多少就能放"
            这类问题。明天来的客户，这条线照搬，一视同仁。</li>
      </ol>
      <p class="tail">三条短板指向同一件事：<b>这条线只看分数，不看这笔贷款本身。</b>
         页签④ 换了个起点，从一笔贷款的账出发，一开头就给出它的结论：<b>3 年期 __RAW3__、5 年期 __RAW5__</b>。两套规则对 <b>__DIFF_N__ 人（__DIFF_PCT__）</b>的判定不一致。</p>
    </div>
  </section>

  <section id="p4" class="panel">
    <div class="keybox">
      <p><b>放贷方案② 的业务建议</b></p>
      <p class="sub">每笔贷款单独算期望利润，为正就放：正常还完赚 <b>金额 × 年净息差 × 期限</b>，违约亏 <b>金额 ×（1 − 回收率）</b>。
         本批只有 3 年期、5 年期两种贷款，净息差和回收率全批共用，规则落地成按期限的两条线。</p>
      <ol>
        <li><b>能拆线就拆两条</b>：3 年期 ≥ <b>__RAW3__</b> 拒、5 年期 ≥ <b>__RAW5__</b> 拒（原始概率轴，同 ③）。
            共拒 <b>__TWO_REJ_N__ 人（__TWO_REJ_PCT__）</b>，中性口径期望利润 <b>__TWO_M__ 万元</b>，比单一线多 <b>__TWO_GAIN__ 万元</b>。</li>
        <li><b>只能维持一条线就用 __T55__</b>：期望利润 <b>__TH55__ 万元</b>；0.001 步长细扫的最优点在 __FINE_T__，<b>__TH_M__ 万元</b>。</li>
      </ol>
      <p class="tail">金额是模拟值（净息差 6%、回收率 30%），上线前换机构的真实口径重算。__TWO_GAIN__ 万元约 4%，
         每 100 亿元年放款额一年约多 <b>__GAIN100__ 万元</b>，拿去比改造成本。推进顺序是先按 __T55__ 把单一线跑起来，
         口径和审批系统改造都到位后再切两条线；改造前先确认审批系统能按期限出两张表、财务给得出真实的净息差与回收率、校准层有人维护。</p>
    </div>
    <h2>放贷方案② · 第一步：概率校准，让"概率"能直接乘金额</h2>
    <p class="sub">页签③ 的"避免坏账""放弃收入""净增益"都要拿答案去数（回看），线上放贷没有答案。
       这一节换成事前就能算的规则：把概率乘进金额。p 不准账就错，不校准直接算会拒掉 87.7% 的申请人。</p>
    <p class="sub">校准器用验证集拟合（有答案），测试集只做检验。</p>
    <details class="how">
      <summary><span class="ftag">方法</span>校准器怎么拟合、什么时候会失效</summary>
      <details class="fold"><summary><span class="ftag">前提</span>校准器只在有答案的数据上拟合一次，然后冻住</summary>
        <p>校准器只在有标签的数据上拟合一次：拿验证集（有答案）算出这条映射，然后冻住不动。
         这一版就是 <code>p_cal = sigmoid(a×logit(p_raw) + b)</code> 两个数；之后对任何新批次，
         套这条曲线就得到校准概率，<b>不需要知道谁真的违约</b>。这也是校准的业务意义：
         只有校准后，0.20 才真的是"这批人里 20% 会违约"，才能拿去算定价、风险成本（PD × LGD）和拨备。</p></details>
      <p>要留意的是基础违约率会漂移：映射里含了"这批人整体违约多少"的水平，换了客群或宏观环境就会偏，
         确认它偏了只能等贷款到期拿回标签，中间可以先用"模型输出概率的分布有没有移动"来预警。</p>
    </details>
    <div class="card">
      <details class="fold"><summary><span class="ftag">读数</span>模型报的概率与实际差多少</summary>
        <p><b>模型报出来的概率和实际差多少</b><br>
        测试集上，模型平均报 <b>0.4495（45%）</b>，这批人里实际只有 <b>19.95%（20%）</b> 真违约，风险被高估了。
        排序是准的（AUC 0.7249），偏的只是刻度：像一支温度计，冷热方向对，度数整体偏高。</p></details>
      <p><b>为什么非得校准</b><br>
        "p ≥ 0.5 就拒"这类规则只看排序，刻度不准也能用，0.5 那条线本身不受校准影响。
        算钱要用"概率 × 金额"，概率大一倍每笔账都错，所以得把刻度拉回真实水平。</p>
    </div>
    __TBL_CALCMP__
    <p class="hint">三行的拒绝率都在<b>各自概率轴的 0.5</b> 上算，不能直接横比；要横比得换等价阈值（原始 0.55 对应校准 0.2382，拒的是同一批人）。
       与阈值无关、可以直接比的是平均预测概率、Brier、AUC 三列。</p>
    <h3>分箱对照：每个分数段说多少、实际多少</h3>
    <p class="sub"><b>实际违约率</b>从 3.8% 升到 48.1%，排序是对的；
       未校准时模型平均偏高 25 个百分点，校准后十档平均只差 0.5 个百分点。</p>
    <details class="fold"><summary><span class="ftag">数据表</span>10 档：模型报多少 vs 实际多少</summary>
    __TBL_CALBIN__</details>
    <details class="how">
      <summary><span class="ftag">读法</span>这张表怎么做、每一档怎么读</summary>
      <details class="fold"><summary><span class="ftag">做法</span>按原始概率分十档，再看每档的实际违约率</summary>
        <p>做法：按模型报出的<b>原始概率</b>把测试集 79,850 人分成 10 等份，每份 7,985 人，
         第 1 档是模型认为最安全的那 10%，第 10 档是最危险的那 10%。每一档看三个数：
         模型<b>未校准</b>时平均报多少、<b>校准后</b>平均报多少、<b>实际</b>有多少人违约。</p></details>
      <p>升序排列说明排序是对的。未校准时模型报得普遍偏高，差得最多的第 9 档高出 33 个百分点。
         校准之后这一列和实际几乎重合，最大差 0.9 个百分点。</p>
    </details>
    __FIG_CALCURVE__

    <h2>第二步：逐笔算账，解出按期限的两条线</h2>
    <p class="sub">一笔贷款放出去，赚和亏能直接算：正常还，赚 <b>金额 × 年净息差 × 期限</b>；违约，亏 <b>金额 ×（1 − 回收率）</b>。
       两件事按概率加权，就是这笔贷款的期望利润。</p>
    <p class="code-line"><code>期望利润 = (1 − p) × 金额 × 年净息差 × 期限 − p × 金额 ×（1 − 回收率）</code></p>
    <p class="sub">金额是公因子，约掉不影响正负号。让期望利润大于 0，把概率解出来：</p>
    <p class="code-line"><code>p* = 年净息差 × 期限 ÷（年净息差 × 期限 + 1 − 回收率）</code></p>
    <details class="fold"><summary><span class="ftag">定义</span>p* 是每笔贷款自己的审批线</summary>
        <p class="sub">p* 就是这笔贷款自己的审批线：概率高过它，放出去就是亏的。净息差与回收率全批共用，所以 p* 只随期限变，
       中性口径下 3 年期 <b>0.2045</b>、5 年期 <b>0.3000</b>（校准概率轴）。③ 挑出来的数只对那批数据成立，这条公式换一批客户也能套用。</p></details>
    <details class="fold"><summary><span class="ftag">前提</span>逐笔规则比的是概率数值，不是排序</summary>
        <p class="sub">这条规则是拿概率的<b>数值</b>去比 p*，不比排序，所以刻度必须准。同一套逐笔规则，把未校准概率直接代进去，
       会拒掉 <b>__EL_RAW_REJ__</b> 的人、期望利润只剩 <b>__EL_RAW_M__ 万元</b>；换成校准概率，拒绝率降到 <b>__EL_NEU_REJ__</b>、
       期望利润 <b>__EL_NEU_M__ 万元</b>。这 <b>__SCALE_GAIN__ 万元</b>是修好刻度的贡献；按期限把线拆成两条，相对最优单线只多
       <b>__TWO_GAIN__ 万元</b>——期限决定的是每条线画在哪，不是这笔差额的来源。</p></details>
    <h3>和 ③ 的区别</h3>
    <details class="fold"><summary><span class="ftag">数据表</span>方案① 与方案② 的逐档对照</summary>
    __TBL_CMP35__</details>
    <p class="sub">为什么有了 ③ 还不够：③ 只回答了"线设在哪"。同一个 __T55__ 分数，3 年期和 5 年期的两笔贷款在 ③ 眼里一样，
       在这里一个该拒 28.9%、一个该拒 54.9%。</p>
    <p class="sub">③ 也不能删：它只依赖排序，刻度漂了还能用，落地成本也低一个数量级。审批系统按期限出两张评分表（3 年期 __RAW3__、5 年期 __RAW5__），③ 那条单线当兜底。</p>
    <h3>调参数看敏感度：回收率和净息差一变，两种做法各值多少</h3>
    <div class="sim">
      <div class="card" style="margin-top:0">
        <div class="slider-row">
          <span class="sub">违约回收率（催收能力决定，去要真实值）</span>
          <input id="eRec" type="range" min="0" max="1" step="0.05" value="0.3">
          <span class="thr-val" id="eRecVal">30%</span>
        </div>
        <div class="slider-row">
          <span class="sub">年净息差（含你自己能定的放贷利率）</span>
          <input id="eMar" type="range" min="0" max="0.2" step="0.01" value="0.06">
          <span class="thr-val" id="eMarVal">6%</span>
        </div>
        <svg style="margin-top:14px" viewBox="0 0 900 260" preserveAspectRatio="xMidYMid meet" role="img" aria-label="逐笔算账与最优单一线的收益对比">
          <g id="eGrid">__DEF_E_GRID__</g>
          <polyline id="eLineTh" fill="none" stroke="#1f5fbf" stroke-width="2.6" points="__DEF_E_TH__"/>
          <polyline id="eLineEp" fill="none" stroke="#1e8449" stroke-width="2.6" points="__DEF_E_EP__"/>
          <line id="eMarker" x1="__DEF_E_MX__" y1="16" x2="__DEF_E_MX__" y2="218" stroke="#1b1f24" stroke-width="1.4" stroke-dasharray="4 4"/>
        </svg>
        <div class="lg">
          <span><i style="background:#1e8449"></i>逐笔算账（每笔期望利润为正才放）</span>
          <span><i style="background:#1f5fbf"></i>最优单一线（一条线管所有人）</span>
        </div>
        <p class="hint" style="margin-top:14px">两条线贴得很近：差额最大 295 万元，而纵轴最高点是它的 160 倍，图上几乎看不出来。
           下面把差额单独画一遍（回收率 30% 时，差额在净息差 7% 附近最高；净息差越厚差额越小）。</p>
        <svg viewBox="0 0 900 190" preserveAspectRatio="xMidYMid meet" role="img" aria-label="逐笔算账比最优单一线多赚多少">
          <g id="eGapGrid">__DEF_E_GAPGRID__</g>
          <polyline id="eGapLine" fill="none" stroke="#c0392b" stroke-width="2.6" points="__DEF_E_GAPPTS__"/>
          <line id="eGapMarker" x1="__DEF_E_GAPMX__" y1="__DEF_E_GAPMY1__" x2="__DEF_E_GAPMX__" y2="__DEF_E_GAPMY2__"
                stroke="#1b1f24" stroke-width="1.4" stroke-dasharray="4 4"/>
          <text id="eGapLabel" x="__DEF_E_GAPLX__" y="__DEF_E_GAPLY__" font-size="12" fill="#c0392b">__DEF_E_GAPLT__</text>
        </svg>
      </div>
      <div>
        <div class="cards" style="grid-template-columns:1fr 1fr;margin-top:0">
          <div class="kpi"><div class="k">隐含阈值（3 年期 / 5 年期）</div><div class="v" id="eImp">__DEF_E_IMP3__ ／ __DEF_E_IMP5__</div><div class="n">校准概率轴；按期限各解一条</div></div>
          <div class="kpi"><div class="k">拒绝率</div><div class="v" id="eRej">__DEF_E_REJ__</div><div class="n">这套线在测试集上会拒掉多少人</div></div>
          <div class="kpi"><div class="k">拦截率</div><div class="v" id="eRec2">__DEF_E_REC__</div><div class="n">真违约者里拦下多少</div></div>
          <div class="kpi"><div class="k">被拒者精确率</div><div class="v" id="ePre">__DEF_E_PRE__</div><div class="n">被拒的人里真违约占比</div></div>
        </div>
        <div class="cards" style="grid-template-columns:1fr 1fr;margin-top:10px">
          <div class="kpi"><div class="k">逐笔算账上限</div><div class="v" id="eEp">__DEF_E_EPV__</div><div class="n">万元</div></div>
          <div class="kpi"><div class="k">最优单一线</div><div class="v" id="eTh">__DEF_E_THV__</div><div class="n">万元</div></div>
          <div class="kpi"><div class="k">逐笔多赚</div><div class="v" id="eGap">__DEF_E_GAP__</div><div class="n">万元</div></div>
          <div class="kpi"><div class="k">单一线最优阈值</div><div class="v" id="eT">__DEF_E_T__</div><div class="n">0.001 步长细扫的最优点，取整到 0.05 就是 0.55</div></div>
        </div>
        <div class="ledger" id="eNote">__DEF_E_NOTE__</div>
      </div>
    </div>
    <p class="hint">两张图同一批测试集、同一套参数，差别只在规则：蓝线从候选审批线里挑最赚的那条，绿线每笔按自己的期限解一条
       （中性口径就是 3 年期与 5 年期），期望利润为正才放。横轴是净息差，拖滑块换回收率。</p>
    <p class="hint">多出来的 <b>__TWO_GAIN__ 万元</b>是换人换出来的：② 把 ① 拒掉的 <b>__SW_IN_N__ 人</b>（平均利率 __SW_IN_RT__ 的 5 年期）
       放进来，这批的期望利润 <b>+__SW_IN_M__ 万元</b>；同时把 ① 会放的 <b>__SW_OUT_N__ 人</b>（平均利率 __SW_OUT_RT__ 的 3 年期）改判为拒，
       这批放出去是 <b>−__SW_OUT_M__ 万元</b>，拒掉就避开了。</p>
    <p class="hint">四张小卡跟着滑块变，说的是这套参数解出来的两条线本身：线画在哪（校准概率轴）、会拒掉多少人、拦下多少违约者、被拒的人里多少是真违约。</p>
    <h3>把两条线摆到同一根轴上比</h3>
    <p class="sub">页签③ 那条线换算到校准轴上是<b>一个数 __C55__</b>（原始轴 __T55__），不分期限。
       逐笔的线按期限分成两条，正好把它夹在中间。</p>
    __TBL_TERM__
    <p class="sub">__DEF_TERM_CONC__</p>
    <details class="fold"><summary><span class="ftag">来历</span>两条线：长期限放宽，短期限收紧</summary>
        <p class="sub">这就是开头那两条线的来历：<b>长期限放宽，短期限收紧</b>（3 年期 __RAW3__、5 年期 __RAW5__），5 年期多赚两年净息差，才接得住更高的违约率。
       拆线的期望利润 <b>__TWO_M__ 万元</b>，正好等于逐笔算账的上限 __EPV_EP__ 万元，不必改审批系统就能拿到。</p></details>
    <details class="how">
      <summary><span class="ftag">口径</span>这个账的两处简化</summary>
      <details class="fold"><summary><span class="ftag">口径</span>净息差取常数后，逐笔规则退化成按期限两条线</summary>
        <p>中性口径下净息差取常数、金额只做乘法不改正负号，逐笔规则实际只按期限分叉，所以拆成两条线就等于逐笔算账的上限。
         收益按「本金 × 年净息差 × 期限」算，等于假设本金整段占用（真实业务里本金分期归还，平均占用只有一半上下）；
         两种期限也用了同一个模拟净息差 6%。数据里 5 年期平均利率 __RT5__%、3 年期 __RT3__%，真按每笔利率算，5 年期这条线只会比 __RAW5__ 更松。</p></details>
      <p>回收率由催收能力决定，得去要真实值；净息差里放贷利率那一半在数据里（每笔贷款都有 interestRate），另一半资金成本要去问财务。
         两个滑块用来看假设敏不敏感，不是决策杠杆。</p>
    </details>
    __TBL_EL__
    <p class="hint">表里的"乐观／中性／保守"指<b>收益假设</b>的松紧：乐观＝利息全赚、中性＝净息差 6%、保守＝净息差 3%，三行用的是同一套校准概率。
       前两行是固定线（"全放"不筛人，"固定阈值 0.5"切在原始概率 0.5 上），其余四行是逐笔规则，没有统一阈值，"被拒占比"在那几行是结果。</p>
    <h3>敏感性：换回收率与收益口径</h3>
    <p class="sub">这张图回答参数一变、审批会松还是紧。纵向是收益口径（利息全额是不扣资金成本的上限，净息差 3% 是保守），
       横向是回收率（催收能力），格子里是拒绝率。中性一格（净息差 6%、回收率 30%）拒绝 __S_N_REJ__、
       拦截 __S_N_REC__、被拒者里真违约 __S_N_PRE__，就是页头和 ③④ 用的那套数。</p>
    __FIG_SENS_HEAT__
    <p class="hint">行和滑块是同一套规则：滑块落在表里的档位（净息差 9%／6%／3%，回收率 20%／30%／40%／50%）上，那一行会亮，
       四张小卡就是那一格的四个数。</p>
    <details class="fold"><summary><span class="ftag">数据表</span>4 种收益口径 × 4 档回收率的完整数字（隐含线、拦截率、被拒者精确率）</summary>
    __TBL_SENS__</details>
    <details class="fold"><summary><span class="ftag">读法</span>赚头（纵向）是粗调，催收（横向）是细调</summary>
        <p class="sub">净息差 6% 那行，回收率从 20% 提到 50%，拒绝率从 __S_6_20__ 降到 __S_6_50__；
       回收率 30% 那列，赚头从保守换到乐观，拒绝率在 __S_CONS30__ 和 __S_OPT30__ 之间跑。
       页头那套建议只在中性那一格成立，两个参数都是模拟值，上线前换成真实值重算。</p></details>
    <h3>靠线的人交人工：__RB_N__ 人的复审带（方案② 执行细节）</h3>
    <p class="sub">两条线把 __RB_DECIDED_PCT__ 的人分得很干脆，剩下的卡在线上，模型也拿不准。这撮人送人工复审，其余机器直接判。</p>
    <details class="fold"><summary><span class="ftag">口径</span>复审带取线上方 20% 以内</summary>
        <p class="sub">带宽取线上方 20% 以内：校准概率轴上 3 年期 0.2045 到 0.2455、5 年期 0.3000 到 0.3600 进复审，更上面的直接拒。
       带宽多宽是审批政策，不是模型算出来的：放宽到 30%，复审人数从 __RB_N__ 涨到 __RB_W30__。</p></details>
    __TBL_REVIEW__
    <details class="fold"><summary><span class="ftag">读数</span>复审带的违约率落在放行段与直接拒段之间</summary>
        <p class="sub">复审带里实际违约 __RB_DR__，落在放行段的 __RB_APPR_DR__ 和直接拒段的 __RB_FAR_DR__ 中间；
       __RB_N__ 人里正常客户 __RB_GOOD__ 人。模型给这三段报的是 12.1%、25.9%、38.2%（校准概率的三段均值），
       和实际差不到 1 个百分点，这一段概率刻度也是准的。</p></details>
    <details class="fold"><summary><span class="ftag">读数</span>这段带子值多少：两头各算一次</summary>
        <p class="sub">这一段值多少，看两头的数：不挑不拣全放，按模型口径亏 __RB_LOSS__ 万元；把里面的 __RB_GOOD__ 名正常人挑出来放，息差是 __RB_GOOD_INT__ 万元。
       复审带的价值就在这两头之间，转人工就是把它往好的那头拉；全拒则这批生意不做。</p></details>
    <h3>上线时哪些要重做</h3>
    <p class="sub">前面几张表的收益，都是拿历史批次里"谁真的违约了"数出来的，属于回看。真正上线以后，只有两件事需要在有标签的历史数据上重做，其余环节都在前线跑，不需要答案。</p>
    <div class="scroll"><table class="tbl"><thead><tr><th>环节</th><th>要不要答案</th><th>什么时候做</th></tr></thead><tbody>
      <tr><td>挑 ③ 那条线（扫曲线找 0.55）</td><td>要</td><td>上线前在带标签的历史批次上做一次，换客群要重挑</td></tr>
      <tr><td>拟合校准曲线（a、b 两个数）</td><td>要</td><td>上线前在验证集上拟合一次，之后冻住</td></tr>
      <tr><td>核对收益（6,611 万、多赚 267 万）</td><td>要</td><td>回看，只有历史数据做得到</td></tr>
      <tr><td>日常审批：分数 ≥ 线就拒</td><td>不要</td><td>每笔在线做</td></tr>
      <tr><td>逐笔算账：算这笔贷款自己的线</td><td>不要，但要金额、期限、利率</td><td>每笔在线做</td></tr>
      <tr><td>盯概率刻度有没有漂</td><td>不要（看分数分布）</td><td>定期</td></tr>
    </tbody></table></div>
  </section>
</main>

<footer>
  <p>数据：阿里天池"违约贷款数据集"（只用 train 80 万建模，testA 20 万无标签）｜模型：LightGBM（W3 调优）｜
     本页由 <code>build_dashboard_html.py</code> 生成，数字可在 <code>outputs/W4/</code> 的 CSV 核对；交互版见 <code>app_dashboard.py</code>。</p>
</footer>

<script>__JS__</script>
<div id="zoom"><img id="zimg" alt="放大的图"></div>

</body>
</html>
"""

html = (HTML
        .replace("__CSS__", CSS)
        .replace("__JS__", JS.replace("__DATA__", json.dumps(DATA, ensure_ascii=False)))
        .replace("__N__", f"{N:,}")
        .replace("__ACC__", f"{DATA['metrics']['accuracy']:.3f}")
        .replace("__AUC__", f"{DATA['metrics']['auc']:.4f}")
        .replace("__EPV_EP__", f"{_EPV_EP:,.0f}")
        .replace("__BASE_M__", f"{_BASE_M:,.0f}")
        .replace("__G50_N__", f"{_G50_N:,}")
        .replace("__G50_M__", f"{_G50_M:,.0f}")
        .replace("__D1__", f"{_D1:,.0f}")
        .replace("__D2__", f"{_D2:,.0f}")
        .replace("__TBL_TERM__", TERM_TBL_HTML)
        .replace("__KS__", f"{_ks:.4f}")
        .replace("__TBL_HEAD__", HEAD_TBL_HTML)
        .replace("__OOT_NTR__", f'{int(oot.loc[_OOT, "训练样本数"]):,}')
        .replace("__OOT_NTE__", f'{int(oot.loc[_OOT, "样本数"]):,}')
        .replace("__TBL_OOT__", OOT_TBL_HTML)
        .replace("__RND_AUC__", f'{oot.loc[_RAND, "AUC"]:.4f}')
        .replace("__OOT_AUC__", f'{oot.loc[_OOT, "AUC"]:.4f}')
        .replace("__RND_CAP10__", f'{oot.loc[_RAND, "前10%捕获率"] * 100:.1f}%')
        .replace("__OOT_CAP10__", f'{oot.loc[_OOT, "前10%捕获率"] * 100:.1f}%')
        .replace("__OOT_DR__", f'{oot.loc[_OOT, "违约率"] * 100:.1f}%')
        .replace("__RND_DR__", f'{oot.loc[_RAND, "违约率"] * 100:.1f}%')
        .replace("__DR_GAP__", f'{(oot.loc[_OOT, "违约率"] - oot.loc[_RAND, "违约率"]) * 100:.1f}')
        .replace("__REJ_GAP__", f'{(oot.loc[_OOT, "阈值0.5拒绝率"] - oot.loc[_RAND, "阈值0.5拒绝率"]) * 100:.1f}')
        .replace("__TBL_REVIEW__", REVIEW_TBL_HTML)
        .replace("__RB_N__", f"{RB_N:,}")
        .replace("__RB_DECIDED_PCT__", f"{RB_DECIDED_PCT * 100:.1f}%")
        .replace("__RB_DR__", f"{RB_DR * 100:.1f}%")
        .replace("__RB_APPR_DR__", f"{RB_APPR_DR * 100:.1f}%")
        .replace("__RB_FAR_DR__", f"{RB_FAR_DR * 100:.1f}%")
        .replace("__RB_GOOD__", f'{int(_rbk["复审带内正常人"]):,}')
        .replace("__RB_LOSS__", f'{abs(float(_rbk["复审带全放期望利润(万元)"])):,.0f}')
        .replace("__RB_GOOD_INT__", f'{float(_rbk["复审带正常人全放息差(万元)"]):,.0f}')
        .replace("__RB_SHARE__", f'{abs(float(_rbk["复审带全放期望利润(万元)"])) / _EPV_EP * 100:.1f}%')
        .replace("__RB_W30__", f"{RB_W30:,}")
        .replace("__C55__", f"{_C55:.4f}")
        .replace("__T55__", f"{_T55:.2f}")
        .replace("__FINE_T__", f"{_FINE_T:.3f}")
        .replace("__FINE_M__", f"{_FINE_M:,.0f}")
        .replace("__FINE_DIFF__", f"{_FINE_DIFF:,.1f}")
        .replace("__FINE_GAIN__", f"{_FINE_GAIN:,.0f}")
        .replace("__FINE_GAIN_DIFF__", f"{_FINE_GAIN_DIFF:,.1f}")
        .replace("__C55_FINE__", f"{_C55_FINE:.4f}")
        .replace("__EL_RAW_REJ__", f"{_EL_RAW_REJ:.1f}%")
        .replace("__EL_RAW_M__", f"{_EL_RAW_M:,.0f}")
        .replace("__EL_NEU_REJ__", f"{_EL_NEU_REJ:.1f}%")
        .replace("__EL_NEU_M__", f"{_EL_NEU_M:,.0f}")
        .replace("__SCALE_GAIN__", f"{_SCALE_GAIN:,.0f}")
        .replace("__SW_IN_N__", f"{_SW_IN_N:,}")
        .replace("__SW_IN_M__", f"{_SW_IN_M:,.1f}")
        .replace("__SW_IN_RT__", f"{_SW_IN_RT:.1f}%")
        .replace("__SW_OUT_N__", f"{_SW_OUT_N:,}")
        .replace("__SW_OUT_M__", f"{_SW_OUT_M:,.1f}")
        .replace("__SW_OUT_RT__", f"{_SW_OUT_RT:.1f}%")
        .replace("__GAIN100__", f"{_GAIN100:,.0f}")
        .replace("__GAIN100C__", f"{_GAIN100C:,.0f}")
        .replace("__N_FINE_REJ__", f"{_N_FINE_REJ:,}")
        .replace("__REJ_XAXIS__", f"{_REJ_XAXIS:.1f}%")
        .replace("__FINE_EP_GAIN__", f"{_FINE_M - _BASE_M:,.0f}")
        .replace("__BASE_M__", f"{_BASE_M:,.0f}")
        .replace("__GAIN_GAP__", f"{_FINE_M - _BASE_M - _FINE_GAIN:,.0f}")
        .replace("__GAIN_GAP_DR__", f"{y[prob >= _FINE_T].mean() * 100:.1f}%")
        .replace("__GAIN_GAP_PD__", f"{p_cal[prob >= _FINE_T].mean() * 100:.1f}%")
        .replace("__VAL_T__", f"{_VAL_T:.2f}")
        .replace("__VAL_GAIN__", f"{_VAL_GAIN:,.0f}")
        .replace("__VAL_GAP__", f"{_VAL_GAP:,.0f}")
        .replace("__VAL_GAP_PCT__", f"{_VAL_GAP_PCT:.1f}%")
        .replace("__PLAT_LOW_PCT__", f"{_PLAT_LOW_PCT:.1f}%")
        .replace("__RAW3__", f"{RAW3:.2f}")
        .replace("__RAW5__", f"{RAW5:.2f}")
        .replace("__RAW3_4__", f"{RAW3:.4f}")
        .replace("__RT3__", f"{_RT3:.1f}")
        .replace("__GAP3__", f"{_GAP3:,.0f}")
        .replace("__S_OPT__", f"{_S_OPT * 100:.1f}%")
        .replace("__S_CONS__", f"{_S_CONS * 100:.1f}%")
        .replace("__S_N_REJ__", f"{_S_N_REJ * 100:.1f}%")
        .replace("__S_N_REC__", f"{_S_N_REC * 100:.1f}%")
        .replace("__S_N_PRE__", f"{_S_N_PRE * 100:.1f}%")
        .replace("__S_N_IMP3__", f"{_S_N_IMP3 * 100:.1f}%")
        .replace("__S_N_IMP5__", f"{_S_N_IMP5 * 100:.1f}%")
        .replace("__S_6_20__", f"{_S_6_20 * 100:.1f}%")
        .replace("__S_6_50__", f"{_S_6_50 * 100:.1f}%")
        .replace("__S_CONS30__", f"{_S_CONS30 * 100:.1f}%")
        .replace("__S_OPT30__", f"{_S_OPT30 * 100:.1f}%")
        .replace("__RT5__", f"{_RT5:.1f}")
        .replace("__TWO_REJ_N__", f"{TWO_REJ_N:,}")
        .replace("__TWO_REJ_PCT__", f"{TWO_REJ_N / N * 100:.1f}%")
        .replace("__TWO_M__", f"{TWO_M:,.0f}")
        .replace("__TWO_GAIN__", f"{TWO_GAIN:,.0f}")
        .replace("__TH_M__", f"{_EPV_TH:,.0f}")
        .replace("__TH55__", f"{_T55_FINE:,.0f}")
        .replace("__TBL_CMP35__", CMP35_HTML
                 .replace("__T55__", f"{_T55:.2f}")
                 .replace("__TH_M__", f"{_EPV_TH:,.0f}")
                 .replace("__FINE_T__", f"{_FINE_T:.3f}")
                 .replace("__EPV_EP__", f"{_EPV_EP:,.0f}"))
        .replace("__BEST_REJ_N__", f'{_best_g["reject_rate"] * N:,.0f}')
        .replace("__BEST_REJ_PCT__", f'{_best_g["reject_rate"] * 100:.1f}%')
        .replace("__BEST_GAIN__", f'{_best_g["gain"]:,.0f}')
        .replace("__BEST_GAIN_DIFF__", f'{_best_g["gain"] - _g5["gain"]:,.0f}')
        .replace("__G5R_REJ__", f'{_g5r["③ 拒多少"]}')
        .replace("__G5R_WANT__", f'{_g5r["逐笔该拒多少"]}')
        .replace("__G3_REJ__", f'{_g3["③ 拒多少"]}')
        .replace("__G3_WANT__", f'{_g3["逐笔该拒多少"]}')
        .replace("__DIFF_N__", f'{DIFF_N:,}')
        .replace("__DIFF_PCT__", f'{DIFF_PCT:.1f}%')
        .replace("__PRE__", f"{DATA['metrics']['precision']:.3f}")
        .replace("__REC__", f"{DATA['metrics']['recall']:.3f}")
        .replace("__F1__", f"{DATA['metrics']['f1']:.3f}")
        .replace("__CHART__", CHART_SVG)
        .replace("__MONEY__", MONEY_SVG)
        .replace("__TBL_GEN__", render_table(gen, pct_cols=PCT))
        .replace("__TBL_CMP__", render_table(cmp_df, pct_cols=PCT))
        .replace("__TBL_THR__", render_table(thr2, pct_cols={"拒绝率", "被拒者精确率", "拦截率"}))
        .replace("__TBL_CALCMP__", render_table(cal_cmp, pct_cols=PCT))
        .replace("__TBL_CALBIN__", render_table(cal_bins, headers={"档位": "档位（按原始概率从低到高）"},
                                                pct_cols=PCT, int_cols=INT))
        .replace("__TBL_EL__", render_table(el, pct_cols=PCT))
        .replace("__TBL_SENS__", render_table(sens, pct_cols=PCT, table_id="tblSens",
                                              row_attrs=_sens_attrs))
        .replace("__TBL_FC__", render_table(fc, pct_cols={"训练集 AUC", "验证集 AUC"}, int_cols=INT))
        .replace("__FIG_TARGET__", FIG_TARGET)
        .replace("__FIG_DIST__", FIG_DIST)
        .replace("__FIG_CMP__", FIG_CMP)
        .replace("__FIG_GRADE__", FIG_GRADE)
        .replace("__FIG_CORR__", FIG_CORR)
        .replace("__FIG_INCOME__", FIG_INCOME)
        .replace("__FIG_ROC__", FIG_ROC)
        .replace("__FIG_PR__", FIG_PR)
        .replace("__FIG_CM__", FIG_CM)
        .replace("__FIG_LEARN__", FIG_LEARN)
        .replace("__FIG_IMP__", FIG_IMP)
        .replace("__FIG_SHAP__", FIG_SHAP)
        .replace("__FIG_TUNING__", FIG_TUNING)
        .replace("__FIG_FC__", FIG_FC)
        .replace("__FIG_CALCURVE__", figure("W4/calibration_curve.png",
                                            "图 16｜校准曲线：校准后（折线贴近对角线）概率才可直接算钱",
                                            cls="fig pngcap"))
        .replace("__FIG_SENS_HEAT__", FIG_SENS_HEAT)
        .replace("__FIG_SWEEP__", figure("W4/threshold_sweep.png",
                                         f"图 15｜阈值扫描（净增益口径）：横轴是审批线，两种概率刻度叠画（同一条曲线、刻度不同，见页签④），"
                                         f"标出最优单一线 t*={_FINE_T:.3f}、旧口径 0.50 和逐笔算账的上限",
                                         cls="fig pngcap"))
        )

for _k, _v in DEF.items():
    html = html.replace(f"__DEF_{_k}__", _v)

assert "__DEF_" not in html, "还有未替换的预渲染占位符：" + str(
    [m for m in __import__("re").findall(r"__DEF_[A-Z_]+__", html)][:5])
_LEAK = sorted(set(__import__("re").findall(r"__[A-Z][A-Z0-9_]*__", html)))
assert not _LEAK, f"还有未替换的占位符：{_LEAK[:6]}"
assert html.count('class="tabin"') == 4 and html.count("<label class=\"tab\"") == 4, "页签数量应为 4"
assert 'id="p5"' not in html, "⑤ 已并入 ④，不应再有 p5 面板"
assert '<div class="advice" id="advice">' in html and 'mBest' in html, "建议框或金额曲线没进 HTML"
assert html.count("避免坏账") >= 2, "预设档位表没有带钱"

DST.write_text(html, encoding="utf-8")
kb = DST.stat().st_size / 1024
print(f"已生成：{DST}")
print(f"文件大小：{kb:,.0f} KB（单文件，含 {HTML.count('__FIG_')} 张内嵌图）")
print(f"测试集 {N:,} 人｜违约 {N_POS:,} / 正常 {N_NEG:,}｜阈值网格 {len(grid)} 档（0.05–0.95）")
print(f"自检：阈值 0.50 → 精确率 {_g5['precision']:.4f}、召回率 {_g5['recall']:.4f}，与 model_final_metrics.csv 一致")
