"""
app_dashboard.py - 业务仪表板（W4 交付物，阶段 6）
项目：信贷违约预测模型
运行：streamlit run scripts/W4/app_dashboard.py
页面：1) 数据概览  2) 模型表现  3) 阈值模拟器（业务交互核心）

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

metrics = load_metrics()
gen_df = load_generalization()
cmp_df = load_model_compare()
pred = load_predictions()
thr_df = load_threshold_table()

st.title("信贷违约预测 · 业务仪表板")
st.caption("阿里线上数据分析实习项目 | 基于 LightGBM 的信贷违约预测模型")

tab_overview, tab_model, tab_threshold = st.tabs(
    ["① 数据概览", "② 模型表现", "③ 阈值模拟器"])

# ============================================================
# 页签 1：数据概览
# ============================================================
with tab_overview:
    st.header("数据概览")
    st.markdown(
        "数据来自阿里天池信贷违约数据集（train 80 万条）。清洗原则：**修格子不砍样本**"
        "（缺失填充、异常封顶、仅删 1,502 行），特征经工程与选择后保留 38 个。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本量（清洗后）", "798,498")
    c2.metric("特征数（建模用）", "38")
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
                 caption="违约 vs 正常（约 1:4，类别不均衡）", width="stretch")
    with col_b:
        st.image(str(OUT / "W1" / "eda_grade_default_rate.png"),
                 caption="贷款等级与违约率：等级越高（A 好）违约越低", width="stretch")

    st.subheader("原始变量分布（W1 数据探索）")
    st.caption("以下为清洗后、特征工程前的原始变量分布；模型实际输入为特征工程后的 38 个特征（见②模型表现）。")
    col_c, col_d = st.columns(2)
    with col_c:
        st.image(str(OUT / "W1" / "eda_numeric_distributions.png"),
                 caption="核心数值特征分布（按违约/正常分组；收入为 log 刻度）", width="stretch")
    with col_d:
        st.image(str(OUT / "W1" / "eda_income_log_compare.png"),
                 caption="年收入原始分布严重右偏 → log 变换后近似钟形（模型使用 log 后特征）",
                 width="stretch")

    st.subheader("特征相关性")
    center = st.columns([1, 2, 1])[1]
    with center:
        st.image(str(OUT / "W1" / "eda_corr_heatmap.png"),
                 caption="特征相关性热力图", width="stretch")

# ============================================================
# 页签 2：模型表现
# ============================================================
with tab_model:
    st.header("模型表现（测试集最终评估）")
    st.markdown("最终模型：**LightGBM（W3 随机搜索优化）**，阈值 0.5。测试集为模型从未见过的 79,850 人。")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("准确率 Accuracy", f"{metrics['accuracy']:.4f}")
    m2.metric("精确率 Precision", f"{metrics['precision']:.4f}")
    m3.metric("召回率 Recall", f"{metrics['recall']:.4f}")
    m4.metric("F1", f"{metrics['f1']:.4f}")
    m5.metric("AUC", f"{metrics['auc']:.4f}")

    st.subheader("泛化能力：训练 / 验证 / 测试对比")
    st.dataframe(gen_df, width="stretch")

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

# ============================================================
# 页签 3：阈值模拟器（业务核心）
# ============================================================
with tab_threshold:
    st.header("阈值模拟器：宽进 or 严审？")
    st.markdown(
        "模型给每个借款人一个**违约概率**。阈值就是「审批线」——概率超过它就拒贷。"
        "滑块每动一格，指标实时重算（基于 79,850 个测试借款人）。")

    t = st.slider("审批阈值（违约概率 ≥ 此值 → 拒绝）", 0.05, 0.95, 0.5, 0.05)

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

    # ---------- 金额账（演示测算） ----------
    st.subheader("金额账：模型值多少钱（演示测算）")
    st.caption(
        "和“不用模型（全批放款）”相比，模型每拒掉一批人，既**避免了坏账**（损失避免），"
        "也**放弃了利润**（收入放弃）。净增益 = 避免坏账 − 放弃收入。"
        "以下为**演示性测算**，参数非真实业务数据，仅展示决策方法，不构成利润结论。")

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
            "中间某点“多拒一人避免的坏账 ≈ 放弃的收入”，即演示最优阈值。\n\n"
            "**两个滑块是什么**：回收率 = 违约后机构能追回本金的比例；"
            "净息差 = 放贷利率 − 机构资金成本（每年）。二者是机构业务参数，数据集里没有，"
            "默认值（50% / 5%）仅为行业常见参考，填入真实值后结论即生效。"
            "**除滑块外，所有金额均来自测试集真实数据**（贷款金额、期限、是否违约）。")

    col_rec, col_margin = st.columns(2)
    with col_rec:
        rec = st.slider("违约回收率（违约后还能追回的比例）", 0.0, 1.0, 0.5, 0.05)
    with col_margin:
        margin = st.slider("年净息差（正常客户每年带来的净利差）", 0.0, 0.20, 0.05, 0.01)

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

    # 各阈值下的增益曲线（找演示最优阈值）
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

    st.write("**各阈值下的增益曲线**（最高点=演示最优阈值）")
    st.line_chart(gain_df, height=280)
    st.markdown(
        f"- **演示最优阈值 ≈ {best_th:.2f}**：净增益最大（{gain_df['净增益(亿)'].max():.2f} 亿）。"
        f"低于它 → 拒掉的好客户太多，放弃的收入超过避免的坏账；"
        f"高于它 → 该拒的违约者没被拒，坏账开始超过收益。\n"
        f"- 当前阈值 **{t:.2f}**：避免坏账 {avoided/1e8:.2f} 亿、放弃收入 {foregone/1e8:.2f} 亿、"
        f"净增益 {gain/1e8:.2f} 亿。"
        f"实际决策前需用机构的真实回收率、资金成本与审核成本替换演示参数。")

    # 业务话术
    if t <= 0.35:
        tone = "🟢 **宽进策略**：抓得全，但误伤多"
        advice = ("当前阈值低，几乎把违约者一网打尽（召回率高），但会拒掉大量正常客户"
                  "（精确率低、误伤多）。适合**获客扩张期**或风险容忍度高的场景；"
                  "坏账损失会上升，需要业务能承受。")
    elif t <= 0.65:
        tone = "🟡 **平衡策略**：常规风控档位"
        advice = ("当前阈值在抓违约与误伤正常客户之间取折中，是**日常风控常用档位**。"
                  "默认 0.5 即此区间：AUC 0.7249 下这是精度与召回较均衡的平衡点。")
    else:
        tone = "🔴 **严审策略**：批得准，但放走的多"
        advice = ("当前阈值高，被拒的人里违约者比例高（精确率高、误伤少），"
                  "但会漏掉大量违约者（召回率低），可能**损失业务量**。"
                  "适合高风险客群、或贷后逾期压力大时的收缩期。")

    st.info(f"{tone}\n\n{advice}")
    st.success(
        f"**当前结论**：若按此阈值审批，约 **{default_rate*100:.1f}%** 的申请会被拒绝，"
        f"其中真违约者占被拒者的 **{prec*100:.1f}%**；全部 {n_default:,} 名违约者中，"
        f"我们拦下了 **{rec*100:.1f}%**（{caught:,} 人），漏掉 {fn:,} 人。"
        f"金额账（演示参数下）：避免坏账 **{avoided/1e8:.2f} 亿**、放弃收入 "
        f"**{foregone/1e8:.2f} 亿**，演示最优阈值约 **{best_th:.2f}**。"
        f"最终阈值建议由业务结合真实坏账/资金成本数据确定。")
