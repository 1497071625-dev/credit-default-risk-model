"""
app_dashboard.py - 业务仪表板（W4 交付物，阶段 6）
项目：信贷违约预测模型
运行：streamlit run scripts/W4/app_dashboard.py
页面：1) 数据概览  2) 模型表现  3) 阈值模拟器（业务交互核心）
      4) 概率校准（W4）  5) 期望损失框架（W4）

复用已有产物：
  图：outputs/W1/*.png（EDA）、outputs/W2/*.png（特征重要性）、
      outputs/W3/*.png（SHAP）、outputs/W4/*.png（最终评估、学习曲线）
  表：outputs/W3/model_comparison_optimized.csv、outputs/W4/*.csv
  概率：outputs/W4/test_predictions.csv（由 prepare_test_predictions.py 生成，
        供阈值滑块实时计算）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "outputs").is_dir()),
    Path(__file__).resolve().parents[2],
)
OUT = BASE_DIR / "outputs"

st.set_page_config(page_title="信贷违约预测仪表板", layout="wide")

# ---------- 全局加载（st.cache_data：只读一次，滑动滑块不重读） ----------
@st.cache_data
def load_metrics():
    return pd.read_csv(OUT / "W4" / "model_final_metrics.csv").iloc[0]

@st.cache_data
def load_generalization():
    return pd.read_csv(OUT / "W4" / "generalization_compare.csv")

@st.cache_data
def load_model_compare():
    return pd.read_csv(OUT / "W3" / "model_comparison_optimized.csv")

@st.cache_data
def load_predictions():
    return pd.read_csv(OUT / "W4" / "test_predictions.csv")

@st.cache_data
def load_threshold_table():
    return pd.read_csv(OUT / "W4" / "threshold_table.csv")

@st.cache_data
def load_calibration_compare():
    return pd.read_csv(OUT / "W4" / "calibration_compare.csv")

@st.cache_data
def load_calibration_bins():
    return pd.read_csv(OUT / "W4" / "calibration_bins.csv")

@st.cache_data
def load_expected_loss():
    return pd.read_csv(OUT / "W4" / "expected_loss_summary.csv")

@st.cache_data
def load_predictions_calibrated():
    return pd.read_csv(OUT / "W4" / "test_predictions_calibrated.csv")

@st.cache_data
def load_feature_count():
    return pd.read_csv(OUT / "W4" / "feature_count_auc.csv")

@st.cache_data
def load_head_capture():
    return pd.read_csv(OUT / "W4" / "head_capture.csv")

@st.cache_data
def load_ks():
    return float(pd.read_csv(OUT / "W4" / "ks_summary.csv").iloc[0]["值"])

@st.cache_data
def load_oot_validation():
    return pd.read_csv(OUT / "W4" / "oot_validation.csv")

@st.cache_data
def load_review_band():
    return pd.read_csv(OUT / "W4" / "review_band.csv")

@st.cache_data
def load_review_band_key():
    return pd.read_csv(OUT / "W4" / "review_band_key.csv")

metrics = load_metrics()
gen_df = load_generalization()
cmp_df = load_model_compare()
pred = load_predictions()
thr_df = load_threshold_table()
cal_df = load_calibration_compare()
cal_bins = load_calibration_bins()
el_df = load_expected_loss()
fc_df = load_feature_count()
cal_pt = load_predictions_calibrated()
hc_df = load_head_capture()
ks_val = load_ks()
oot_df = load_oot_validation()
rb_df = load_review_band()
rb_key = load_review_band_key().set_index("项目")["值"]

# ③ 与 ④ 的对照数字：中性口径（净息差 6%、回收率 30%），评价尺子统一
_LOAN = cal_pt["loanAmnt"].to_numpy(float)
_TERM = cal_pt["term"].to_numpy(float)
_PRAW = cal_pt["prob_raw"].to_numpy(float)
_PCAL = cal_pt["prob_cal"].to_numpy(float)
_PER = ((1 - _PCAL) * 0.06 * _TERM - _PCAL * (1 - 0.30)) * _LOAN      # 每笔期望利润（元）
_PS_MAP = {float(_t): 0.06 * _t / (0.06 * _t + 0.70) for _t in sorted(set(_TERM))}
_REJ_TWO = np.zeros(len(_TERM), bool)
_RAW_LINES = {}
for _t, _ps in _PS_MAP.items():
    _m = _TERM == _t
    _r = _m & (_PCAL >= _ps)
    _REJ_TWO |= _r
    _RAW_LINES[_t] = float(_PRAW[_r].min())
TWO_M = float(_PER[~_REJ_TWO].sum() / 1e4)                              # 按期限两条线
_ORDER = np.argsort(_PRAW, kind="stable")
ONE_M = float(np.concatenate([[0.0], np.cumsum(_PER[_ORDER])]).max() / 1e4)   # 最优单一线
ONE_T = float(_PRAW[_ORDER][int(np.argmax(np.concatenate([[0.0], np.cumsum(_PER[_ORDER])]))) - 1])
_CUM_PER = np.concatenate([[0.0], np.cumsum(_PER[_ORDER])])
_IDX55 = int(np.searchsorted(_PRAW[_ORDER], 0.55))
_PROF55 = float(_CUM_PER[_IDX55] / 1e4)          # 0.55 这一档的期望利润（万元）
FINE_DIFF_55 = ONE_M - _PROF55                   # 细扫最优点比 0.55 多出来的钱（万元）
RAW3, RAW5 = _RAW_LINES[min(_RAW_LINES)], _RAW_LINES[max(_RAW_LINES)]
_two_gain = TWO_M - ONE_M
_assert_two = abs(TWO_M - float(el_df.loc[el_df["策略"].str.contains("中性"), "总期望利润(万元)"].iloc[0])) < 1.0
assert _assert_two, f"按期限两条线 {TWO_M:.1f} 应等于逐笔上限（中性口径）"

# 净息差 3% 时，逐笔规则与最优单一线的差距（回答"值不值得为 267 万改系统"）
_PER3 = ((1 - _PCAL) * 0.03 * _TERM - _PCAL * (1 - 0.30)) * _LOAN
_EP3 = float(_PER3[_PER3 > 0].sum() / 1e4)
_TH3 = float(np.concatenate([[0.0], np.cumsum(_PER3[_ORDER])]).max() / 1e4)
_GAP3 = _EP3 - _TH3
assert 80 < _GAP3 < 100, f"净息差 3% 档差距异常：{_GAP3:.0f} 万元"

# 规模换算：每 100 亿元年放款额，② 比 ① 每年多赚多少（拿它去比改造成本）
_BOOK_M = float(_LOAN.sum() / 1e4)                       # 测试集放款额合计（万元）
_GAIN100 = _two_gain / _BOOK_M * 1e6
_GAIN100C = _GAP3 / _BOOK_M * 1e6
assert 2000 < _GAIN100 < 2600 and 650 < _GAIN100C < 950, "每 100 亿元年放款口径异常"

# 这 267 万元是谁贡献的：② 与 ① 各换了一批人
_RATE = cal_pt["interestRate"].to_numpy(float)
_PS_ARR = np.where(_TERM == 5.0, 0.06 * 5 / (0.06 * 5 + 0.70), 0.06 * 3 / (0.06 * 3 + 0.70))
_ACC_A, _ACC_B = _PRAW < ONE_T, _PCAL < _PS_ARR
_SW_IN, _SW_OUT = _ACC_B & ~_ACC_A, _ACC_A & ~_ACC_B
_SW_IN_N, _SW_IN_M = int(_SW_IN.sum()), float(_PER[_SW_IN].sum() / 1e4)
_SW_OUT_N, _SW_OUT_M = int(_SW_OUT.sum()), float(-_PER[_SW_OUT].sum() / 1e4)
_SW_IN_RT, _SW_OUT_RT = float(_RATE[_SW_IN].mean()), float(_RATE[_SW_OUT].mean())
assert abs((_SW_IN_M + _SW_OUT_M) - _two_gain) < 1.0, "换人拆分与总差额对不上"

T55 = 0.55        # ③ 在 0.05 步长网格上选出的审批线（原始概率轴）
_YT = cal_pt["y_true"].to_numpy(int)


def _gain_wan(_t: float) -> float:
    """③ 的金额口径：净增益 = 避免坏账 − 放弃收入（相对全批放款，万元）。"""
    _r = _PRAW >= _t
    return float(((1 - 0.30) * _LOAN[_r & (_YT == 1)]).sum()
                 - (0.06 * _TERM[_r & (_YT == 0)] * _LOAN[_r & (_YT == 0)]).sum()) / 1e4


_GRID_FINE = np.round(np.arange(0.30, 0.8001, 0.001), 3)
_GAIN_FINE = np.array([_gain_wan(float(_t)) for _t in _GRID_FINE])
_GI = int(_GAIN_FINE.argmax())
assert abs(float(_GRID_FINE[_GI]) - ONE_T) < 0.0015, "净增益与期望利润的细扫最优点不在同一档"
FINE_GAIN = float(_GAIN_FINE[_GI])
FINE_GAIN_55 = _gain_wan(T55)
_VAL_GAIN = _gain_wan(0.560)
_PLAT_LOW_PCT = min(_gain_wan(float(_q)) for _q in np.round(np.arange(0.54, 0.5801, 0.001), 3)) / FINE_GAIN * 100
assert 0 < FINE_GAIN - FINE_GAIN_55 < 50 and 0 < FINE_GAIN - _VAL_GAIN < 40, "③ 净增益差值异常"
assert _PLAT_LOW_PCT > 95, f"平台区下限异常：{_PLAT_LOW_PCT:.1f}%"
_RT3 = float(cal_pt.loc[cal_pt["term"] == 3, "interestRate"].mean())
_RT5 = float(cal_pt.loc[cal_pt["term"] == 5, "interestRate"].mean())
assert 11 < _RT3 < 13 and 16 < _RT5 < 18, f"分期限平均利率异常：{_RT3:.2f} / {_RT5:.2f}"
_TERM_STAT = {}
for _t in sorted(set(_TERM)):
    _m = _TERM == _t
    _TERM_STAT[float(_t)] = {
        "n": int(_m.sum()),
        "default": float(_YT[_m].mean()),
        "rej55": float((_PRAW[_m] >= T55).mean()),
        "want": float((_PCAL[_m] >= _PS_MAP[float(_t)]).mean()),
    }
_PS_ARR = np.where(_TERM == 5.0, _PS_MAP[5.0], _PS_MAP[3.0])
DIFF_N = int(((_PRAW >= T55) != (_PCAL >= _PS_ARR)).sum())
DIFF_PCT = DIFF_N / len(_TERM) * 100

st.title("信贷违约预测 · 业务仪表板")
st.caption("阿里线上数据分析实习项目（W1–W4）：拿借款人的资料预测这笔贷款会不会违约，再把模型输出接到一条能落地的放贷审批线上")
_f1, _f2, _f3 = st.columns(3)
_f1.markdown("**数据**\n\n阿里天池信贷违约数据集：一行一笔贷款申请，含金额、期限、利率、信用等级、年收入等；train 80 万行有标签，testA 20 万行无标签")
_f2.markdown("**口径**\n\n建模只用 train：清洗后 798,498 行，7:2:1 切出训练 558,948、验证 159,700、测试 79,850，本页成绩与金额都在测试集上算，testA 没参与")
_f3.markdown("**模型**\n\nLightGBM（W3 随机搜索优化），测试集 AUC 0.7249；金额里的净息差 6%、回收率 30% 是模拟值")
_BASE_M = float(el_df.loc[el_df["策略"].str.contains("全放"), "总期望利润(万元)"].iloc[0])
_G50_N = int((_PRAW >= 0.50).sum())
_G50_M = float(el_df.loc[el_df["策略"].str.contains("固定阈值"), "总期望利润(万元)"].iloc[0])
assert abs(_G50_N - 33041) < 5 and abs(_G50_M - 6155) < 5, "0.50 档参照与 expected_loss_summary.csv 对不上"
st.success(
    f"**建议：审批线按期限拆成两条，3 年期 {RAW3:.2f}、5 年期 {RAW5:.2f}**（模型的概率轴），"
    f"拒 **{int(_REJ_TWO.sum()):,} 人（{_REJ_TWO.mean() * 100:.1f}%）**，中性口径期望利润 **{TWO_M:,.0f} 万元**，"
    f"比最优单一线多 **{_two_gain:,.0f} 万元**；只能设一条线就用 **{T55:.2f}**，"
    f"同一口径 **{ONE_M:,.0f} 万元**（这两个数都是页签④ 的账）。")

st.markdown(
    "**审批线**是一条分数门槛：模型给每个人算一个违约概率（0 到 1），达到这条线的人拒贷，低于的放贷。"
    "放贷方案有两个：方案①（页签③）全批设一条线，简单、只依赖排序，也是兜底；方案②（页签④）从每笔贷款的账出发"
    "（正常还完赚息差、违约亏本金，两头按概率加权，期望为正就放），本批数据只有 3 年期、5 年期两种贷款，"
    "这条规则落地成两条线。")

tab_overview, tab_model, tab_threshold, tab_method = st.tabs(
    ["① 数据概览", "② 模型表现", "③ 放贷方案① 单线（可拖动）",
     "④ 放贷方案② 拆线：按期限设两条线"])

# ============================================================
# 页签 1：数据概览
# ============================================================
with tab_overview:
    st.header("数据概览")
    st.markdown(
        "数据来自阿里天池信贷违约数据集（train 80 万条）。清洗原则：**修格子不砍样本**"
        "（缺失填充、异常封顶、仅删 1,502 行）。47 个原始字段经工程与选择后保留 38 个进模型；"
        "本页的图都是 W1 口径，用的是加工前的原始字段。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本量（清洗后）", "798,498")
    c2.metric("特征数（建模用）", "38", help="47 个原始字段加工而来")
    c3.metric("违约率", "20.0%")
    c4.metric("测试集（未参与建模）", "79,850")

    st.subheader("数据质量处理（W1）")
    q1, q2, q3, _ = st.columns([1, 1, 1, 2])
    q1.metric("缺失填充", "46,375 个", help="就业年限缺失填'未知'")
    q2.metric("异常封顶", "2,719 个", help="revolUtil>100 封顶到 100")
    q3.metric("删除行", "1,502（0.19%）", help="先检测后处理，仅删极少数越界行")
    st.caption("清洗思路：修格子不砍样本——缺失填充、异常封顶，样本保留率 99.8%。")

    st.subheader("目标变量：是否违约")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(str(OUT / "W1" / "eda_target_distribution.png"),
                 caption="违约 19.96%、正常 80.04%，约 1:4，属类别不平衡，建模时做了加权", width="stretch")
    with col_b:
        st.image(str(OUT / "W1" / "eda_grade_default_rate.png"),
                 caption="各等级违约率：从 A 的 6.0% 单调升到 G 的 49.7%，差 8 倍，这是最强的单变量分层", width="stretch")

    st.subheader("原始变量分布（W1 数据探索）")
    st.caption("以下为清洗后、特征工程前的原始变量分布；模型实际输入为特征工程后的 38 个特征（见②模型表现）。")
    col_c, col_d = st.columns(2)
    with col_c:
        st.image(str(OUT / "W1" / "eda_numeric_distributions.png"),
                 caption="六个核心变量的分布：金额、分期额、利率单峰右偏；年收入极端长尾（一半人在 6.5 万以下）；额度使用率封顶在 100，接近对称", width="stretch")
    with col_d:
        st.image(str(OUT / "W1" / "eda_income_log_compare.png"),
                 caption="年收入的原始刻度偏度 46.4，log 变换后降到 0.20，接近对称；这是做对数变换的依据",
                 width="stretch")

    st.subheader("违约组与正常组的差异")
    center2 = st.columns([1, 2, 1])[1]
    with center2:
        st.image(str(OUT / "W1" / "eda_compare_default.png"),
                 caption="违约组的利率、金额、负债收入比中位数分别高 24%、19%、15%，年收入反而低 8%；分组差异最大的是利率", width="stretch")

    st.subheader("特征相关性")
    center = st.columns([1, 2, 1])[1]
    with center:
        st.image(str(OUT / "W1" / "eda_corr_heatmap.png"),
                 caption="图上 7 个关键数值特征 + 目标。最强的一对是贷款金额和分期额 0.95（分期额本就是按金额、利率、期限算出的月供，两者信息重复）；其余两两相关最高 0.46；与违约相关性最高的是利率 0.26", width="stretch")

# ============================================================
# 页签 2：模型表现
# ============================================================
with tab_model:
    st.header("模型表现（测试集最终评估）")
    st.markdown("最终模型：**LightGBM（W3 随机搜索优化）**，阈值 0.5。测试集为模型从未见过的 79,850 人。")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("准确率 Accuracy", f"{metrics['accuracy']:.4f}")
    m2.metric("精确率 Precision", f"{metrics['precision']:.4f}")
    m3.metric("召回率 Recall", f"{metrics['recall']:.4f}")
    m4.metric("F1", f"{metrics['f1']:.4f}")
    m5.metric("AUC", f"{metrics['auc']:.4f}")
    m6.metric("KS", f"{ks_val:.4f}")

    st.subheader("头部区分度：KS 与头部捕获率")
    st.markdown(
        "AUC 在整批人上算平均，看不出头部的表现。审批更关心名单头部：分数最高的那 10% 里抓到多少违约者。"
        f"KS = **{ks_val:.4f}**，指违约者和正常人的分数分布拉开的最大距离，风控常用的门槛是 0.30。"
        "两个指标都只看排序，与概率刻度无关，校准前后一样。")
    hc_show = hc_df.copy()
    hc_show["头部占比"] = hc_show["头部占比"].map(lambda x: f"前 {x*100:.0f}%")
    hc_show["头部人数"] = hc_show["头部人数"].map(lambda x: f"{x:,}")
    hc_show["命中违约人数"] = hc_show["命中违约人数"].map(lambda x: f"{x:,}")
    hc_show["捕获率"] = hc_show["捕获率"].map(lambda x: f"{x*100:.1f}%")
    hc_show["lift"] = hc_show["lift"].map(lambda x: f"{x:.1f} 倍")
    hc_show["头部内违约率"] = hc_show["头部内违约率"].map(lambda x: f"{x*100:.1f}%")
    st.dataframe(hc_show.rename(columns={"lift": "是随机挑的", "命中违约人数": "抓到违约者",
                                         "头部内违约率": "这批人自己的违约率"}),
                 width="stretch", hide_index=True)

    st.subheader("泛化能力：训练 / 验证 / 测试对比")
    st.dataframe(gen_df, width="stretch")

    st.subheader("时间外验证：拿过去训练，给未来打分")
    st.markdown(
        "上面的成绩都出自随机切分：同一批放款随机分到训练和测试，两边行情一样，成绩会偏乐观。"
        "真实上线只能用过去的数据训练、给未来上门的客户打分。")
    _rnd_row = oot_df[oot_df["口径"] == "随机切分（对照）"].iloc[0]
    _oot_row = oot_df[oot_df["口径"] == "时间外（2017-01 起）"].iloc[0]
    st.markdown(
        f"按放款日期切一次：训练用 2016 年底之前的 {int(_oot_row['训练样本数']):,} 笔，"
        f"测试用 2017-01 到 2018-06 的 {int(_oot_row['样本数']):,} 笔。"
        "2018-07 之后的 8,980 笔没放进来，那段违约率只有 7.5%，是真变好还是违约没暴露，从数据里分不清。")
    oot_show = pd.DataFrame([{
        "切分方式": r["口径"],
        "样本数": f"{int(r['样本数']):,}",
        "违约率": f"{r['违约率']*100:.1f}%",
        "AUC": f"{r['AUC']:.4f}",
        "KS": f"{r['KS']:.4f}",
        "前 10% 捕获率": f"{r['前10%捕获率']*100:.1f}%",
        "阈值 0.5 拒绝率": f"{r['阈值0.5拒绝率']*100:.1f}%",
    } for _, r in oot_df.iterrows()])
    st.dataframe(oot_show, width="stretch", hide_index=True)
    st.markdown(
        f"排序能力掉了一点，没伤到根：AUC 从 {_rnd_row['AUC']:.4f} 到 {_oot_row['AUC']:.4f}，"
        f"前 10% 捕获率从 {_rnd_row['前10%捕获率']*100:.1f}% 到 {_oot_row['前10%捕获率']*100:.1f}%，仍远高于随机挑的 10%。\n\n"
        f"更该留意行情那一层：时间外这段的违约率是 {_oot_row['违约率']*100:.1f}%，"
        f"比随机切分的 {_rnd_row['违约率']*100:.1f}% 高 {( _oot_row['违约率']-_rnd_row['违约率'])*100:.1f} 个百分点，"
        f"同一个阈值在新数据上拒的人会多 {(_oot_row['阈值0.5拒绝率']-_rnd_row['阈值0.5拒绝率'])*100:.1f} 个百分点。"
        "④ 拿概率数值算钱，那条概率轴要定期用最近到期的贷款重新标定。")

    st.subheader("模型性能图")
    col_e, col_f = st.columns(2)
    with col_e:
        st.image(str(OUT / "W4" / "roc_curve_test.png"), caption="ROC 曲线（测试集）")
        st.image(str(OUT / "W4" / "confusion_matrix_test.png"), caption="混淆矩阵（阈值 0.5）")
    with col_f:
        st.image(str(OUT / "W4" / "pr_curve_test.png"), caption="PR 曲线（测试集）")
        st.image(str(OUT / "W4" / "learning_curve.png"), caption="学习曲线：无过拟合，加数据仍有小幅收益")

    st.subheader("特征重要性（W2 随机森林 vs W3 SHAP）")
    col_i, col_j = st.columns(2)
    with col_i:
        st.image(str(OUT / "W2" / "feature_importance.png"),
                 caption="W2 特征工程：随机森林特征重要性（筛选弱特征后保留 38 个）",
                 width="stretch")
    with col_j:
        st.image(str(OUT / "W3" / "shap_summary.png"),
                 caption="W3 SHAP 全局归因：subGrade 影响最大，其次期限、房产权属、信用分",
                 width="stretch")

    st.subheader("模型对比（验证集）")
    st.dataframe(cmp_df, width="stretch")
    st.caption("LGB(优化) 与 Stacking 几乎持平，但更可解释、易部署，故选为最终模型。")

    st.subheader("特征数量的边际收益（W4 特征数量实验）")
    st.markdown(
        "按随机森林重要性从高到低，依次只喂前 N 个特征重训同一个 LGB，看验证集 AUC 怎么变。"
        "**注意：本实验只看训练集与验证集，测试集未参与。**")
    fc_show = fc_df.rename(columns={
        "n_features": "使用特征数", "train_auc": "训练集 AUC",
        "val_auc": "验证集 AUC", "seconds": "训练秒数", "added": "本档新增特征"})
    st.dataframe(fc_show[["使用特征数", "训练集 AUC", "验证集 AUC", "训练秒数"]],
                 width="stretch", hide_index=True)
    col_fc1, col_fc2 = st.columns([3, 2])
    with col_fc1:
        st.image(str(OUT / "W4" / "feature_count_auc.png"),
                 caption="验证 AUC 在第 20–25 个特征后基本走平", width="stretch")
    with col_fc2:
        st.markdown(
            "**怎么读**\n\n"
            "- 只用 **3 个特征**（子等级、利率、等级）就拿到满血性能的 **96%**\n"
            "- **20 个特征**到 99.4%；补齐剩下 18 个只多 **+0.0041**\n"
            "- **25 个之后饱和**：25→30 一点没涨\n"
            "- 训练分一路涨、验证分早早走平 → 后面的特征更多在帮模型背训练集\n\n"
            "**业务含义**：最终保留 38 个（数据已在手，不增加线上成本）；"
            "若将来要精简采集口径，砍到 20–25 个、AUC 损失不到 0.005。")

# ============================================================
# 页签 3：阈值模拟器（业务核心）
# ============================================================
with tab_threshold:
    st.header("放贷方案① · 单一线：宽进 or 严审？")
    st.markdown(
        "模型给每个借款人一个**违约概率**。阈值就是「审批线」——概率超过它就拒贷。"
        "滑块每动一格，指标实时重算（基于 79,850 个测试借款人）。\n\n"
        "滑块和横轴取的是**模型的概率轴**，也就是模型直接报出来的未校准概率；页签④ 会把这同一个数重新刻度成真实违约率，"
        "两种刻度落在同一批人身上，排序完全一样。")

    t = st.slider("审批阈值（模型报的概率 ≥ 此值 → 拒绝）", 0.05, 0.95, 0.5, 0.05)

    prob = pred["prob_default"]
    y_true = pred["y_true"]
    y_pred = (prob >= t).astype(int)

    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    default_rate = float(y_pred.mean())
    n_reject = int(y_pred.sum())
    n_default = int(y_true.sum())
    caught = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("拒绝占比 default_rate", f"{default_rate*100:.1f}%",
              help="被拒贷的借款人占全部申请人的比例")
    r2.metric("精确率 Precision", f"{prec:.3f}",
              help="被拒的人里，真的违约者的比例")
    r3.metric("召回率 Recall", f"{rec:.3f}",
              help="所有真违约者里，被我们拦下的比例")
    r4.metric("F1", f"{f1:.3f}")

    st.subheader("当前阈值下的业务账本")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.dataframe(pd.DataFrame({
            "指标": ["拒绝人数（申请 79,850 人）", "其中真违约者", "漏掉的违约者",
                     "误伤的正常客户"],
            "人数": [f"{n_reject:,}", f"{caught:,}", f"{fn:,}", f"{fp:,}"],
            "占申请比": [f"{n_reject/len(pred)*100:.1f}%",
                        f"{caught/len(pred)*100:.1f}%",
                        f"{fn/len(pred)*100:.1f}%",
                        f"{fp/len(pred)*100:.1f}%"],
        }), width="stretch")
    with cc2:
        st.subheader("预设档位对照（阈值表）")
        st.dataframe(thr_df.round(3), width="stretch")
        st.caption("0.3=抓得多误伤多；0.7=抓得准漏得多；默认建议 0.5。")

    # ---------- 金额账（模拟参数测算） ----------
    st.subheader("金额账：模型值多少钱（模拟测算）")
    st.caption(
        "和“不用模型（全批放款）”相比，模型每拒掉一批人，既**避免了坏账**（损失避免），"
        "也**放弃了利润**（收入放弃）。净增益 = 避免坏账 − 放弃收入。"
        "以下是**模拟测算**：回收率、净息差是假设值，只展示决策方法，不构成利润结论。")

    with st.expander("口径说明：这笔账是怎么算的？"):
        st.markdown(
            "**输入**：79,850 名测试借款人，每人有 4 样信息——模型违约概率、真实是否违约、"
            "贷款金额、期限（3/5 年）。\n\n"
            "**对比基准**：不用模型、全批放款。\n\n"
            "**计算三步**：\n"
            "1. 阈值＝审批线：违约概率 ≥ 阈值 → 拒贷；\n"
            "2. 两笔账：被拒且**真违约** → 避免坏账 = 贷款金额 × (1−回收率)；"
            "被拒且**正常** → 放弃收入 = 贷款金额 × 年净息差 × 期限年；\n"
            "3. 净增益 = 避免坏账 − 放弃收入（相对全批的增量）。\n\n"
            "**为什么有最优阈值**：阈值太低 → 误伤太多正常客户，放弃的收入超过避免的坏账，"
            "净增益为负；阈值太高 → 该拒的违约者没拒，坏账重新爬升；"
            "中间某点“多拒一人避免的坏账 ≈ 放弃的收入”，即最优阈值。\n\n"
            "**两个滑块是什么**：回收率 = 违约后机构能追回本金的比例；"
            "净息差 = 放贷利率 − 机构资金成本（每年）。这两列数据集里没有，这里用的是模拟值"
            "（回收率 30% / 净息差 6%，与项目报告、PPT 的中性口径一致），"
            "填入机构真实值后，金额结论要重算。"
            "**除滑块外，所有金额均来自测试集真实数据**（贷款金额、期限、是否违约）。")

    col_rec, col_margin = st.columns(2)
    with col_rec:
        rec = st.slider("违约回收率（违约后还能追回的比例）", 0.0, 1.0, 0.3, 0.05)
    with col_margin:
        margin = st.slider("年净息差（正常客户每年带来的净利差）", 0.0, 0.20, 0.06, 0.01)

    amt = pred["loanAmnt"]
    t_years = pred["term"]  # 3 或 5 年
    rejected = (y_pred == 1)
    amt_rejected = float(amt[rejected].sum())
    avoided = float((amt[rejected & (y_true == 1)] * (1 - rec)).sum())
    foregone = float((amt[rejected & (y_true == 0)] * margin * t_years[rejected & (y_true == 0)]).sum())
    gain = avoided - foregone

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("少放贷款", f"{amt_rejected/1e8:.2f} 亿",
              help="当前阈值下被拒贷的金额合计")
    a2.metric("避免坏账", f"{avoided/1e8:.2f} 亿",
              help="被拒的人里真违约者的本金损失（×1−回收率）")
    a3.metric("放弃收入", f"{foregone/1e8:.2f} 亿",
              help="被拒的人里正常客户全期可带来的息差收入（净息差×金额×期限年）")
    a4.metric("模型净增益", f"{gain/1e8:.2f} 亿",
              delta=f"{gain/1e8:.2f} 亿", delta_color="normal",
              help="避免坏账−放弃收入；正数=模型帮机构多赚/少亏")

    # 各阈值下的增益曲线（找最优阈值）
    th_grid = np.arange(0.05, 0.96, 0.05)
    gain_rows = []
    for th in th_grid:
        rej = (prob >= th)
        av = (amt[rej & (y_true == 1)] * (1 - rec)).sum() / 1e8
        fo = (amt[rej & (y_true == 0)] * margin * t_years[rej & (y_true == 0)]).sum() / 1e8
        gain_rows.append({"阈值": th, "避免坏账(亿)": av, "放弃收入(亿)": fo,
                          "净增益(亿)": av - fo})
    gain_df = pd.DataFrame(gain_rows).set_index("阈值")
    best_th = float(gain_df["净增益(亿)"].idxmax())

    st.write("**各阈值下的增益曲线**（最高点=最优阈值）")
    st.line_chart(gain_df, height=280)
    st.markdown(
        f"- **最优阈值 ≈ {best_th:.2f}**：净增益最大（{gain_df['净增益(亿)'].max():.2f} 亿）。"
        f"低于它 → 拒掉的好客户太多，放弃的收入超过避免的坏账；"
        f"高于它 → 该拒的违约者没被拒，坏账开始超过收益。\n"
        f"- 当前阈值 **{t:.2f}**：避免坏账 {avoided/1e8:.2f} 亿、放弃收入 {foregone/1e8:.2f} 亿、"
        f"净增益 {gain/1e8:.2f} 亿。"
        f"实际决策前要用机构的真实回收率、资金成本与审核成本替换这些模拟值。")

    _g50 = float(gain_df.loc[0.50, "净增益(亿)"])
    st.subheader("这条线是怎么定出来的")
    st.markdown(
        f"1. **先看默认值。** 0.50 是数学默认值：模型说概率过半就拒。它跟业务没关系，"
        f"换一批客群、换一个净息差，它也还是 0.50。\n"
        f"2. **把 0.05 到 0.95 逐档扫一遍。** 每一档都算一次「避免坏账 − 放弃收入」，就是上面那条曲线。\n"
        f"3. **取曲线最高点。** 最高的一档是 **{best_th:.2f}**：拒绝 "
        f"{float((prob >= best_th).sum()):,.0f} 人（{float((prob >= best_th).mean())*100:.1f}%），"
        f"净增益 **{float(gain_df['净增益(亿)'].max()):.2f} 亿**，比 0.50 的 {_g50:.2f} 亿多 "
        f"{(float(gain_df['净增益(亿)'].max()) - _g50)*1e4:.0f} 万元。"
        f"曲线按 0.05 步长画；再细扫到 0.001 步长，最优点在 **{ONE_T:.3f}**，净增益 **{FINE_GAIN:,.0f} 万元**，与 0.55 这一档差 {FINE_GAIN - FINE_GAIN_55:.1f} 万元。")
    st.caption(
        "换一批数据还站得住吗：`threshold_select_on_val.py` 把挑线搬到验证集上重做一遍，选出 t=0.560，"
        f"搬到测试集净增益 {_VAL_GAIN:,.0f} 万元，与测试集自己扫出的 {FINE_GAIN:,.0f} 万元差 "
        f"{FINE_GAIN - _VAL_GAIN:,.0f} 万元。"
        f"0.54 到 0.58 是一段平台区（净增益都在最优值的 {_PLAT_LOW_PCT:.1f}% 以上），区间内挑哪个都行。")

    # 业务话术
    if t <= 0.35:
        tone = "**宽进档**：抓得全，但误伤多"
        advice = ("当前阈值低，几乎把违约者一网打尽（召回率高），但会拒掉大量正常客户"
                  "（精确率低、误伤多）。适合**获客扩张期**或风险容忍度高的场景；"
                  "坏账损失会上升，需要业务能承受。")
    elif t <= 0.65:
        tone = "**平衡档**：常规风控档位"
        advice = ("当前阈值在抓违约与误伤正常客户之间取折中，是**日常风控常用档位**。"
                  "默认 0.5 即此区间：AUC 0.7249 下这是精度与召回较均衡的平衡点。")
    else:
        tone = "**严审档**：批得准，但放走的多"
        advice = ("当前阈值高，被拒的人里违约者比例高（精确率高、误伤少），"
                  "但会漏掉大量违约者（召回率低），可能**损失业务量**。"
                  "适合高风险客群、或贷后逾期压力大时的收缩期。")

    st.info(f"{tone}\n\n{advice}")
    st.success(
        f"**当前结论**：若按此阈值审批，约 **{default_rate*100:.1f}%** 的申请会被拒绝，"
        f"其中真违约者占被拒者的 **{prec*100:.1f}%**；全部 {n_default:,} 名违约者中，"
        f"我们拦下了 **{rec*100:.1f}%**（{caught:,} 人），漏掉 {fn:,} 人。"
        f"金额账（模拟值口径）：避免坏账 **{avoided/1e8:.2f} 亿**、放弃收入 "
        f"**{foregone/1e8:.2f} 亿**，最优阈值约 **{best_th:.2f}**。"
        f"最终阈值建议由业务结合真实坏账/资金成本数据确定。")

    _s5 = _TERM_STAT[max(_TERM_STAT)]
    _s3 = _TERM_STAT[min(_TERM_STAT)]
    st.subheader("这个做法的三条短板")
    st.markdown(
        f"1. **一条线管所有人，金额、期限、利率都不进规则。** "
        f"一笔 5 年期贷款的利息要收 5 年，一笔 3 年期的只收 3 年，前者本该容忍更高的违约率。"
        f"按 {T55:.2f} 这条线，5 年期拒掉 {_s5['rej55']*100:.1f}%，按逐笔的账只要拒 {_s5['want']*100:.1f}%；"
        f"3 年期反过来，只拒 {_s3['rej55']*100:.1f}%，该拒 {_s3['want']*100:.1f}%。\n"
        f"2. **挑线本身要靠答案。** 「避免坏账」得数出被拒的人里谁真的违约了，只有历史数据做得到。"
        f"换一批客群、换一个净息差，曲线最高点会移动，得重新在旧数据上挑一遍。\n"
        f"3. **它交出一个数字，交不出判断标准。** {T55:.2f} 回答不了「这笔贷款为什么值得放」，"
        f"也说不出「利率提到多少就能放」这类问题。明天来的客户，这条线照搬，一视同仁。")
    st.warning(
        f"三条短板指向同一件事：**这条线只看分数，不看这笔贷款本身。** "
        f"页签④ 里逐笔算账换了个起点，从一笔贷款的账出发。两套规则对 **{DIFF_N:,} 人（{DIFF_PCT:.1f}%）**"
        f"的判定不一致。")


# ============================================================
# 页签 4：概率校准（W4）
# ============================================================
with tab_method:
    st.info(
        f"**方案② 的业务建议**\n\n"
        f"每笔贷款单独算期望利润，为正就放：正常还完赚 **金额 × 年净息差 × 期限**，违约亏 **金额 ×（1 − 回收率）**。"
        f"本批数据只有 3 年期、5 年期两种贷款，净息差和回收率是全批共用的模拟参数，规则落地成按期限的两条线。\n\n"
        f"1. **能拆线就拆两条**：3 年期 ≥ **{RAW3:.2f}** 拒、5 年期 ≥ **{RAW5:.2f}** 拒（原始概率轴，同 ③）。"
        f"共拒 **{int(_REJ_TWO.sum()):,} 人（{_REJ_TWO.mean()*100:.1f}%）**，中性口径期望利润 **{TWO_M:,.0f} 万元**，比单一线多 **{_two_gain:,.0f} 万元**。\n"
        f"2. **只能维持一条线就用 {ONE_T:.2f}**：期望利润 **{ONE_M:,.0f} 万元**"
        f"（0.001 步长细扫的最优点在 {ONE_T:.3f}，两者差 {FINE_DIFF_55:.1f} 万元）。\n\n"
        f"金额是模拟值（净息差 6%、回收率 30%），上线前换机构的真实口径重算。{TWO_M - ONE_M:,.0f} 万元约 4%，"
        f"每 100 亿元年放款额一年约多 **{_GAIN100:,.0f} 万元**，拿去比改造成本。"
        f"推进顺序是先按 {ONE_T:.2f} 把单一线跑起来，口径和审批系统改造都到位后再切两条线；"
        f"改造前先确认审批系统能按期限出两张表、财务给得出真实的净息差与回收率、校准层有人维护。")
    st.header("第一步：概率校准，让「概率」能直接乘金额")
    st.markdown(
        "模型输出的原始概率是**排序用**的——谁比谁风险高，它排得很准；"
        "但「45% 违约概率」这个数值本身**偏大**，不能直接拿去乘金额算期望损失。"
        "校准就是把分数重新映射成**真实的违约率**，让「说 20% 就真的违约 20%」。")
    with st.expander("校准器怎么拟合、什么时候会失效"):
        st.markdown(
            "校准器只在有标签的数据上拟合一次（验证集，有答案），然后冻住不动，"
            "这一版就是 `p_cal = sigmoid(a×logit(p_raw) + b)` 两个数。"
            "之后对任何新批次，套这条曲线就得到校准概率，**不需要知道谁真的违约**。\n\n"
            "要留意的是基础违约率会漂移：映射里含了「这批人整体违约多少」的水平，换了客群或宏观环境就会偏，"
            "确认它偏了只能等贷款到期拿回标签，中间可以先用「模型输出概率的分布有没有移动」来预警。")

    raw = cal_df[cal_df["stage"] == "raw(test)"].iloc[0]
    platt = cal_df[cal_df["stage"] == "platt(test)"].iloc[0]
    iso = cal_df[cal_df["stage"] == "isotonic(test)"].iloc[0]
    actual = cal_df[cal_df["stage"] == "actual(test)"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("校准前平均预测概率", f"{raw['mean_pred']:.4f}",
              delta=f"比实际高 {(raw['mean_pred']/actual['mean_pred']-1)*100:.0f}%",
              delta_color="inverse")
    c2.metric("校准后平均预测概率", f"{platt['mean_pred']:.4f}")
    c3.metric("测试集实际违约率", f"{actual['mean_pred']:.4f}")
    c4.metric("Brier 分数（越小越好）", f"{platt['brier']:.4f}",
              delta=f"{platt['brier']-raw['brier']:.4f} vs 校准前", delta_color="normal")

    st.subheader("校准前后对比（都在测试集上）")
    show = pd.DataFrame({
        "版本": ["校准前（原始概率）", "校准后（Platt）", "校准后（Isotonic，对照）", "实际违约率"],
        "平均预测概率": [f"{raw['mean_pred']:.4f}", f"{platt['mean_pred']:.4f}",
                   f"{iso['mean_pred']:.4f}", f"{actual['mean_pred']:.4f}"],
        "≥0.5 的占比": [f"{raw['share_ge_0.5']*100:.1f}%", f"{platt['share_ge_0.5']*100:.1f}%",
                    f"{iso['share_ge_0.5']*100:.1f}%", "—"],
        "Brier（越小越好）": [f"{raw['brier']:.4f}", f"{platt['brier']:.4f}",
                         f"{iso['brier']:.4f}", "—"],
        "AUC（排序能力）": [f"{raw['auc']:.4f}", f"{platt['auc']:.4f}",
                       f"{iso['auc']:.4f}", "—"],
    })
    st.dataframe(show, width="stretch", hide_index=True)
    st.caption(
        "阈值口径：拒绝率、精确率、召回率都在**各自概率轴的 0.5** 上算。未校准那行是原始概率过 0.5（拒 41.4%），"
        "校准后两行是校准概率过 0.5（拒 3.4% 和 4.1%）。校准把平均概率从 0.4495 拉到 0.1993，"
        "两条轴上的 0.5 落在不同位置，这两行的拒绝率不能直接比；要横比得换等价阈值：原始 0.55 对应校准 0.2382，换完拒的是同一批人。"
        "所以这张表能比的是与阈值无关的三列：平均预测概率、Brier、AUC。")
    st.caption(
        "三个数看校准效果：① 平均概率从 0.4495 降到 0.1993，和实际的 0.1995 几乎相等；"
        "② Brier 从 0.2106 降到 0.1424（大幅改善）；"
        "③ AUC 一点没变（0.7249）——校准只动「数值大小」，不动「谁排前面」。")

    col_cal1, col_cal2 = st.columns([1, 1])
    with col_cal1:
        st.image(str(OUT / "W4" / "calibration_curve.png"),
                 caption="可靠性曲线：点贴住对角线 = 说多少就真发生多少", width="stretch")
    with col_cal2:
        st.markdown("**为什么要做校准**")
        st.markdown(
            "- **概率虚高 125%**：模型平均说 45%，实际只有 20%。直接拿这个概率去算"
            "「期望损失 = 概率 × 金额 × (1−回收率)」，会**高估损失**，"
            "把本来该放的好客户也拒掉。\n"
            "- **阈值会失真**：明明 0.5 是「一半一半」的分界，但由于概率整体虚高，"
            "实际拒掉了 41% 的申请，这是概率标度失真带来的副作用，跟业务规则无关。\n"
            "- **校准不牺牲排序**：AUC 保持不变，说明「谁更危险」的信息一点没丢，"
            "只是把刻度重新对准了真实概率。")
        st.info(
            "**方法怎么选的**：在验证集内部对半再切一刀，一半训练校准器、一半比较效果——"
            f"Isotonic {iso['brier']:.5f} vs Platt {platt['brier']:.5f}，"
            "差别在万分之几，Platt 更平滑、参数更少、外推更稳，故采用 **Platt**。")

    st.subheader("分箱对照：每个分数段说多少、实际多少")
    st.markdown(
        "**实际违约率**从 3.8% 升到 48.1%，排序是对的；"
        "校准前模型平均偏高 25 个百分点，校准后十档平均只差 0.5 个百分点。")
    with st.expander("这张表怎么做、每一档怎么读"):
        st.markdown(
            "做法：按模型报出的**原始概率**把测试集 79,850 人分成 10 等份，每份 7,985 人，"
            "第 1 档是模型认为最安全的那 10%，第 10 档是最危险的那 10%。每一档看三个数："
            "模型**校准前**平均报多少、**校准后**平均报多少、**实际**有多少人违约。\n\n"
            "升序排列说明排序是对的。校准前模型报得普遍偏高，差得最多的第 9 档高出 33 个百分点。"
            "校准之后这一列和实际几乎重合，最大差 0.9 个百分点。")
    bins_show = cal_bins.rename(columns={
        "bin": "分数段", "count": "人数", "mean_pred_raw": "校准前平均概率",
        "mean_pred_cal": "校准后平均概率", "actual_rate": "实际违约率"})
    bins_show["分数段"] = [f"第 {i+1} 档" for i in range(len(bins_show))]
    st.dataframe(bins_show, width="stretch", hide_index=True)
    st.line_chart(bins_show.set_index("分数段")[["校准前平均概率", "校准后平均概率", "实际违约率"]],
                  height=280)
    st.caption("校准后那条线和实际违约率几乎重合，校准前那条线整体偏高，这就是概率虚高的样子。")


# ============================================================
# 页签 5：期望损失框架（W4）
# ============================================================
with tab_method:
    st.divider()
    st.header("放贷方案② · 逐笔算账：从「拍一条线」到「按笔算账」")
    st.subheader("逐笔算账是怎么来的：把「该不该放」写成一笔账")
    st.markdown(
        "一笔贷款放出去，赚和亏能直接算：正常还，赚 **金额 × 年净息差 × 期限**；违约，亏 **金额 ×（1 − 回收率）**。"
        "两件事按概率加权，就是这笔贷款的期望利润。"
        "金额是公因子，约掉不影响正负号；让期望利润大于 0 把概率解出来，就是这笔贷款自己的审批线。p 用校准后的值（见上一节）。")

    st.latex(r"\text{每 1 元贷款的期望利润} = (1-p)\times \text{净收益率}\times \text{期限} - p \times (1-\text{回收率})")
    st.markdown(
        "**决策规则**：期望利润 > 0 批准，≤ 0 拒绝，等价于逐笔的隐含阈值 "
        "`p* = 净收益率 × 期限 / (净收益率 × 期限 + 1 − 回收率)`。")

    st.subheader("和 ③ 的区别")
    st.dataframe(pd.DataFrame([
        ("规则长什么样", "分数 ≥ 0.55 就拒", "期望利润 ≤ 0 就拒"),
        ("用到的信息", "只有分数（排序）", "分数、金额、期限、利率、回收率"),
        ("线有几条", "单一线：全批人共用 1 条", "拆线：每个期限 1 条"),
        ("刻度落在哪", "原始概率轴 0.55", "校准概率轴 3 年期 0.2045、5 年期 0.3000，换成原始概率轴就是 "
                                     f"{RAW3:.2f}、{RAW5:.2f}"),
        ("线是怎么来的", "把曲线扫一遍，取最高点", "从一笔账解出来"),
        ("明天来的客户", "照搬这条线", "照搬公式，需要金额、期限、利率"),
        ("换客群或换参数", "曲线最高点会移动，得在旧数据上重挑", "参数换掉重算即可"),
        ("概率刻度漂了", "还能用（只依赖排序）", "会崩（依赖概率数值）"),
        ("上限（中性口径的期望利润）", f"{ONE_M:,.0f} 万元", f"{TWO_M:,.0f} 万元"),
        ("落地成本", "审批系统支持一个阈值即可", "按期限出两张评分表"),
    ], columns=["对比项", "方案① 单一线", "方案② 逐笔（拆线）"]), width="stretch", hide_index=True)
    st.markdown(
        f"**为什么有了 ③ 还不够**：③ 只回答了「线设在哪」。同一个 0.55 分数，3 年期和 5 年期的两笔贷款在 ③ 眼里一样，"
        f"在 ④ 眼里一个该拒 {_TERM_STAT[min(_TERM_STAT)]['want']*100:.1f}%、一个该拒 {_TERM_STAT[max(_TERM_STAT)]['want']*100:.1f}%。\n\n"
        f"**③ 也不能删**：它只依赖排序，刻度漂了还能用，落地成本也低一个数量级；审批系统按期限出两张评分表"
        f"（3 年期 {RAW3:.2f}、5 年期 {RAW5:.2f}），③ 那条单线当兜底。")

    st.subheader("调参数看敏感度：回收率和净息差一变，两种做法各值多少")
    _ec1, _ec2 = st.columns(2)
    _rec5 = _ec1.slider("违约回收率（催收能力决定，去要真实值）", 0.0, 1.0, 0.30, 0.05, key="el_rec")
    _v5 = _ec2.slider("年净息差（含你自己能定的放贷利率）", 0.0, 0.20, 0.06, 0.01, key="el_mar")
    _p_raw = pred["prob_default"].to_numpy(float)
    _p_cal = cal_pt["prob_cal"].to_numpy(float)
    _loan = cal_pt["loanAmnt"].to_numpy(float)
    _term = cal_pt["term"].to_numpy(float)
    # 这套参数解出来的审批线本身：四张实时读数，和 HTML 版 ④ 滑块下面那四张小卡对得上
    _lim5 = 1 - _rec5
    _den5 = _v5 * _term + _lim5
    _pstar5 = np.where(_den5 > 0, _v5 * _term / _den5, 1.0)
    _rej5 = _p_cal >= _pstar5
    _y5 = cal_pt["y_true"].to_numpy(int)
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("隐含阈值（3 年期 / 5 年期）",
               f"{np.median(_pstar5[_term == 3])*100:.1f}% / {np.median(_pstar5[_term == 5])*100:.1f}%",
               help="校准概率轴；这批数据只有 3 年期、5 年期两种贷款，按期限各解一条线")
    _m2.metric("拒绝率", f"{_rej5.mean()*100:.1f}%", help="这套线在测试集上会拒掉多少人")
    _m3.metric("拦截率", f"{_y5[_rej5].sum()/_y5.sum()*100:.1f}%", help="真违约者里拦下多少")
    _m4.metric("被拒者精确率", f"{_y5[_rej5].mean()*100:.1f}%", help="被拒的人里真违约占比")
    st.caption(
        "四张读数跟着滑块变，说的是这套参数解出来的两条线：线画在哪（校准概率轴）、会拒掉多少人、"
        "拦下多少违约者、被拒的人里多少是真违约。")
    _ep5 = (1 - _p_cal) * _v5 * _term - _p_cal * (1 - _rec5)         # 逐笔单位期望利润
    _app5 = _ep5 > 0
    _ep_profit = float((_ep5[_app5] * _loan[_app5]).sum() / 1e4)
    _tgrid = np.round(np.arange(0.001, 1.0, 0.001), 4)
    _ord = np.argsort(_p_raw, kind="stable")
    _cs = np.concatenate([[0.0], np.cumsum((((1 - _p_cal) * _term * _loan) / 1e4)[_ord])])
    _ds = np.concatenate([[0.0], np.cumsum((( _p_cal * _loan) / 1e4)[_ord])])
    _jj = np.searchsorted(_p_raw[_ord], _tgrid)
    _prof5 = _v5 * _cs[_jj] - (1 - _rec5) * _ds[_jj]
    _bi5 = int(np.argmax(_prof5))
    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric("逐笔算账上限（万元）", f"{_ep_profit:,.0f}")
    _k2.metric("最优单一线（万元）", f"{_prof5[_bi5]:,.0f}")
    _k3.metric("逐笔多赚（万元）", f"{_ep_profit - _prof5[_bi5]:,.0f}")
    _k4.metric("单一线最优阈值", f"{_tgrid[_bi5]:.3f}", help="0.001 步长细扫的最优点，取整到 0.05 就是 0.55")
    st.caption(
        f"同一批测试集、同一套参数，差别只在规则：单一线从候选审批线里挑最赚的那条，"
        f"逐笔算账每笔按自己的期限解一条（中性口径就是 3 年期与 5 年期）。净息差越薄，一条线管所有人丢得越多。")
    _mars_gap = np.round(np.arange(0.001, 0.2001, 0.005), 4)
    _ep_c, _th_c = [], []
    for _vv in _mars_gap:
        _ev = (1 - _p_cal) * _vv * _term - _p_cal * (1 - _rec5)
        _ep_c.append(float((_ev[_ev > 0] * _loan[_ev > 0]).sum() / 1e4))
        _pr = _vv * _cs[_jj] - (1 - _rec5) * _ds[_jj]
        _th_c.append(float(_pr.max()))
    _gap_c = [_a - _b for _a, _b in zip(_ep_c, _th_c)]
    _gi = int(np.argmax(_gap_c))
    st.line_chart(pd.DataFrame({"净息差": _mars_gap, "逐笔 − 单一线（万元）": _gap_c}).set_index("净息差"),
                  height=220)
    st.caption(
        f"差额图：纵轴是逐笔算账比最优单一线多赚多少。当前回收率 {_rec5*100:.0f}% 下，差额在净息差 "
        f"{_mars_gap[_gi]*100:.0f}% 附近最高，{_gap_c[_gi]:,.0f} 万元；净息差越厚差额越小，所以上面那张图看着两条线几乎重合。")
    st.markdown(
        "逐笔算账在数学上更强，一条线只是它的特例（所有人金额、期限、利率一样时就退化成一条线）。页签③ 那条线留着当兜底："
        "它只依赖排序（这套模型 AUC 0.7249），客群或宏观一变，排序还站着、概率刻度先坏。"
        "差价也不厚：中性口径多 267 万元（4.2%），净息差越厚越小，20% 时只多 38 万元（0.1%）。"
        "要不要改，看多赚这几个点值不值得动审批系统（逐笔要求系统实时拿到金额、期限、利率，不少审批引擎做不到）。\n\n"
        f"这笔差额的构成：② 把 ① 拒掉的 **{_SW_IN_N:,} 人**（平均利率 {_SW_IN_RT:.1f}% 的 5 年期）放进来，"
        f"贡献 **+{_SW_IN_M:,.1f} 万元**；把 ① 会放的 **{_SW_OUT_N:,} 人**（平均利率 {_SW_OUT_RT:.1f}% 的 3 年期）改判为拒，"
        f"避开 **−{_SW_OUT_M:,.1f} 万元**。\n\n"
        f"换算成规模：每 100 亿元年放款额，中性口径每年多约 **{_GAIN100:,.0f} 万元**，净息差 3% 时约 **{_GAIN100C:,.0f} 万元**。")

    st.subheader("把两条线摆到同一根轴上比")
    st.markdown(
        "页签③ 那条线换算到校准轴上是**一个数**，不分期限；逐笔的线按期限分成两条，正好把它夹在中间。")
    _T55 = T55
    _n55 = int((_p_raw >= _T55).sum())
    _C55 = float(np.sort(_p_cal)[::-1][_n55 - 1])
    _rows = []
    for _t in sorted(set(_term)):
        _m = _term == _t
        _ps = 0.06 * _t / (0.06 * _t + 1 - 0.30)
        _r3, _r5 = _p_raw[_m] >= _T55, _p_cal[_m] >= _ps
        _rows.append({"期限": f"{_t:.0f} 年", "人数": f"{int(_m.sum()):,}",
                      "实际违约率": f"{pred['y_true'].to_numpy()[_m].mean()*100:.1f}%",
                      "模型报高倍数": f"{_p_raw[_m].mean()/pred['y_true'].to_numpy()[_m].mean():.2f} 倍",
                      "④ 想要的线": f"{_ps:.4f}", "③ 给的线": f"{_C55:.4f}",
                      "③ 拒多少": f"{_r3.mean()*100:.1f}%", "④ 该拒多少": f"{_r5.mean()*100:.1f}%"})
    st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)
    _r3a = _p_raw >= _T55
    _r5a = _p_cal >= np.where(_term == 5.0, 0.06 * 5 / (0.06 * 5 + 0.70), 0.06 * 3 / (0.06 * 3 + 0.70))
    st.markdown(
        f"3 年期 ④ 认为该拒 **{_rows[0]['④ 该拒多少']}**，③ 只拒了 **{_rows[0]['③ 拒多少']}**，放进来太多；"
        f"5 年期 ④ 认为拒 **{_rows[-1]['④ 该拒多少']}** 就够，③ 却拒了 **{_rows[-1]['③ 拒多少']}**。"
        f"两套规则对 **{int((_r3a != _r5a).sum()):,} 人（{(_r3a != _r5a).mean()*100:.1f}%）** 的判定不一致。\n\n"
        f"这才是 ④ 给业务的动作：**长期限放宽，短期限收紧**（5 年期多赚两年净息差，才接得住更高的违约率）。\n\n"
        f"两处简化：收益按「本金 × 年净息差 × 期限」算，等于假设本金整段占用（真实业务里分期归还，平均占用只有一半上下）；"
        f"两种期限用了同一个模拟净息差 6%。数据里 5 年期平均利率 {_RT5:.1f}%、3 年期 {_RT3:.1f}%，真按每笔利率算，5 年期这条线只会比 {RAW5:.2f} 更松。"
        f"两个滑块看的是假设敏不敏感，不是决策杠杆。")

    _ps_map = {float(_t): 0.06 * _t / (0.06 * _t + 0.70) for _t in sorted(set(_term))}
    _rej_two = np.zeros(len(_term), bool)
    _raw_lines = {}
    for _t, _ps in _ps_map.items():
        _m = _term == _t
        _r = _m & (_p_cal >= _ps)
        _rej_two |= _r
        _raw_lines[_t] = float(_p_raw[_r].min())
    _two_m = float((((1 - _p_cal) * 0.06 * _term - _p_cal * 0.70) * _loan)[~_rej_two].sum() / 1e4)
    # 最优单一线：原始轴按 EP 排序累加求峰（等价 threshold_sweep.csv，精确断点 6,344.34 万元）
    _o = np.argsort(_p_raw, kind='stable')
    _per = ((1 - _p_cal) * 0.06 * _term - _p_cal * 0.70) * _loan
    _cum = np.concatenate([[0.0], np.cumsum(_per[_o])])
    _one_m = float(_cum.max() / 1e4)
    _r3v, _r5v = _raw_lines[sorted(_raw_lines)[0]], _raw_lines[sorted(_raw_lines)[-1]]
    st.markdown(
        f"把 ③ 的一条线拆成按期限的两条：**3 年期 {_r3v:.2f}、5 年期 {_r5v:.2f}**（原始概率轴），"
        f"拒绝 **{int(_rej_two.sum()):,} 人（{_rej_two.mean()*100:.1f}%）**，期望利润 **{_two_m:,.0f} 万元**，比最好的单一线多 **{_two_m - _one_m:,.0f} 万元**，"
        f"正好等于逐笔算账的上限（中性口径下净息差取常数、金额不分正负，逐笔规则实际只按期限分叉）。把评分表拆成 3 年期一张、5 年期一张就行。")

    st.subheader("策略对比（统一按中性口径评价）")
    el_show = el_df.copy()
    el_show["被拒占比"] = el_show["被拒占比"].map(lambda x: f"{x*100:.1f}%")
    el_show["拦截率(真违约被拒占比)"] = el_show["拦截率(真违约被拒占比)"].map(
        lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    el_show["被拒者精确率(里面真违约占比)"] = el_show["被拒者精确率(里面真违约占比)"].map(
        lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    el_show["总期望利润(万元)"] = el_show["总期望利润(万元)"].map(lambda x: f"{x:,.0f}")
    st.dataframe(el_show[["策略", "被拒占比", "拦截率(真违约被拒占比)",
                          "被拒者精确率(里面真违约占比)", "总期望利润(万元)", "说明"]],
                 width="stretch", hide_index=True)
    st.caption(
        "表里的「乐观／中性／保守」指收益假设的松紧：乐观＝利息全赚、中性＝净息差 6%、保守＝净息差 3%，"
        "三行用的是同一套校准概率。前两行是固定线（「全放」不筛人，「固定阈值 0.5」切在原始概率 0.5 上）；"
        "其余四行是逐笔规则，没有统一阈值，「被拒占比」在那几行是结果。")

    col_el1, col_el2 = st.columns([3, 2])
    with col_el1:
        st.image(str(OUT / "W4" / "expected_loss_sensitivity.png"),
                 caption="敏感性：收益口径 / 回收率一变，拒绝率跟着变（口径才是最大变量）",
                 width="stretch")
    with col_el2:
        st.markdown("**四条结论**")
        st.markdown(
            "1. **校准是前提**：拿未校准概率去算账会拒掉 **87.7%** 的申请，同一套公式只因为标度错了就不可用。\n"
            "2. **中性口径下比固定阈值更优**：拒绝率从 41.4% 降到 **35.2%**，总期望利润从 6,155 万升到 **6,611 万**，"
            "每笔账分开算就不必一刀切。\n"
            "3. **口径是最大敏感源**：净息差从 3% 变到 9%，拒绝率在 65% 和 18% 之间摆动，比回收率的拉动大得多。"
            "上阈值之前先和财务把口径敲定。\n"
            "4. **边界**：净收益率、回收率是**模拟值**（数据集里没有），上线前必须换成机构真实口径，本页只交付**方法**。")

    st.subheader("靠线的人交人工：复审带")
    _per20 = np.zeros(len(_TERM), bool)
    _per30 = np.zeros(len(_TERM), bool)
    for _t, _ps in _PS_MAP.items():
        _m = _TERM == _t
        _per20 |= _m & (_PCAL >= _ps) & (_PCAL < _ps * 1.2)
        _per30 |= _m & (_PCAL >= _ps) & (_PCAL < _ps * 1.3)
    _band_n = int(rb_key["复审带人数"])
    assert int(_per20.sum()) == _band_n, "复审带人数与 review_band_key.csv 不一致"
    _decided_pct = float(rb_df.loc[rb_df["分工"] != "人工复审", "占测试集"].sum())
    st.markdown(
        f"两条线把 {_decided_pct*100:.1f}% 的人分得很干脆，剩下的卡在线上，模型也拿不准。这撮人送人工复审，其余机器直接判。"
        "\n\n"
        "带宽取线上方 20% 以内（校准概率轴上 3 年期 0.2045 到 0.2455、5 年期 0.3000 到 0.3600 进复审，更上面的直接拒）。"
        f"带宽多宽是审批政策，不是模型算出来的：放宽到 30%，复审人数从 {_band_n:,} 涨到 {int(_per30.sum()):,}。")
    rb_show = rb_df.copy()
    rb_show["人数"] = rb_show["人数"].map(lambda x: f"{x:,}")
    rb_show["占测试集"] = rb_show["占测试集"].map(lambda x: f"{x*100:.1f}%")
    rb_show["实际违约率"] = rb_show["实际违约率"].map(lambda x: f"{x*100:.1f}%")
    rb_show = rb_show.rename(columns={"实际违约率": "这段人的实际违约率"})
    st.dataframe(rb_show, width="stretch", hide_index=True)
    st.markdown(
        f"复审带里实际违约 {float(rb_key['复审带内违约者'])/_band_n*100:.1f}%，"
        f"落在放行段的 {rb_df.loc[rb_df['分工'] == '直接放行', '实际违约率'].iloc[0]*100:.1f}% 和"
        f"直接拒段的 {rb_df.loc[rb_df['分工'] == '直接拒绝', '实际违约率'].iloc[0]*100:.1f}% 中间；"
        f"{_band_n:,} 人里正常客户 {int(rb_key['复审带内正常人']):,} 人。\n\n"
        f"按模型口径，这 {_band_n:,} 人全放会亏 {abs(float(rb_key['复审带全放期望利润(万元)'])):,.0f} 万元，全拒则这批生意不做。"
        f"{abs(float(rb_key['复审带全放期望利润(万元)'])):,.0f} 万元是拆线方案 {TWO_M:,.0f} 万元的 "
        f"{abs(float(rb_key['复审带全放期望利润(万元)']))/TWO_M*100:.1f}%，复审带就算判错，代价有限。")

    st.subheader("上线时哪些要重做")
    st.markdown(
        "前面几张表的收益，都是拿历史批次里「谁真的违约了」数出来的，属于回看。"
        "真正上线以后，只有两件事需要在有标签的历史数据上重做，其余环节都在前线跑，不需要答案。")
    st.dataframe(pd.DataFrame([
        ("挑 ③ 那条线（扫曲线找 0.55）", "要", "上线前在带标签的历史批次上做一次，换客群要重挑"),
        ("拟合校准曲线（a、b 两个数）", "要", "上线前在验证集上拟合一次，之后冻住"),
        ("核对收益（6,611 万、多赚 267 万）", "要", "回看，只有历史数据做得到"),
        ("日常审批：分数 ≥ 线就拒", "不要", "每笔在线做"),
        ("逐笔算账：算这笔贷款自己的线", "不要，但要金额、期限、利率", "每笔在线做"),
        ("盯概率刻度有没有漂", "不要（看分数分布）", "定期"),
    ], columns=["环节", "要不要答案", "什么时候做"]), width="stretch", hide_index=True)

    st.warning(
        "**给业务方的落点**：模型负责把风险排序，最终「放不放」要按每笔贷款的"
        "期望利润来定：收益高、期限长的客户可以放松一点，利差薄的客户要更严。"
        "但这条线的位置，取决于机构自己的资金成本和催收回收能力。")
