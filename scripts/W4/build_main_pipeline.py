"""
build_main_pipeline.py - 生成 notebooks/main_pipeline.ipynb
项目：信贷违约预测模型
说明：把 W1-W4 的脚本与产物整合为一个可复现的主流程 Notebook。
     快速路径：读取 data/ 与 outputs/ 中间产物，几分钟跑完；
     完整重跑：每个阶段标注对应脚本，可单独执行（更耗时）。
用法：.venv/bin/python scripts/W4/build_main_pipeline.py
"""
from pathlib import Path
import nbformat as nbf

BASE = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
NB_DIR = BASE / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ============================================================
md("""# 信贷违约预测 · 完整项目流程（main_pipeline）

**项目**：阿里线上数据分析实习——基于用户历史数据与行为特征，预测信贷违约概率，为金融机构风控决策提供支持。
**数据**：阿里天池信贷违约数据集（train 80 万条 + testA 20 万条，47 个字段，部分匿名/脱敏）。
**模型**：逻辑回归（基线）→ 随机森林 → XGBoost → LightGBM（随机搜索优化）→ 集成对比，最终选择 LightGBM。

本 Notebook 把 W1–W4 的成果整合成一条主线：

| 阶段 | 内容 | 对应交付 |
|---|---|---|
| 1 数据探索 | 业务理解、目标分布、EDA | 数据探索报告 |
| 2 数据预处理 | 缺失/异常/重复处理 | data_preprocessing.py |
| 3 特征工程 | 编码、变换、特征创建与选择 | feature_engineering.py |
| 4 模型构建与评估 | 三模型对比、多指标 | model_training.py |
| 5 模型优化 | 随机搜索、K 折、集成、SHAP | model_optimization.py |
| 6 最终评估 | 测试集成绩单、泛化、阈值选择 | model_final_eval.py |
| 7 业务结论 | 阈值解读、给业务方的话 | 最终项目报告 |

**怎么用**：默认快速路径——读取 data/ 与 outputs/ 已有产物，全程约 3–5 分钟；每个阶段末尾标注了完整重跑脚本（耗时更长，但可复现每一步）。
""")

# ============================================================
md("""## 0. 环境与项目结构

依赖见 `requirements.txt`（pandas / numpy / scikit-learn / xgboost / lightgbm / shap / streamlit 等），建议用项目 `.venv`。""")

code("""# 环境检查
import sys
print("Python", sys.version.split()[0])
import pandas as pd, numpy as np
print("pandas", pd.__version__, "| numpy", np.__version__)
import sklearn
print("scikit-learn", sklearn.__version__)
for lib in ("xgboost", "lightgbm", "shap"):
    try:
        m = __import__(lib)
        print(lib, m.__version__)
    except ImportError:
        print(lib, "未安装")
""")

code("""from pathlib import Path
import pandas as pd, numpy as np
from IPython.display import display, Image, Markdown

# 自动定位项目根目录（含 data/ 的上级目录）
BASE = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data").is_dir()), Path.cwd())
DATA = BASE / "data"
OUT = BASE / "outputs"
print("项目根目录:", BASE)

# 项目结构（核心目录）
for name in ("scripts", "data", "outputs", "notebooks", "reports"):
    p = BASE / name
    print(f"{'✓' if p.exists() else '✗'} {name}/")
""")

md("""## 1. 数据探索（W1）

**业务问题**：给定申请人的历史与行为特征，估计其**违约概率**，据此决定审批/利率。
**目标变量**：`isDefault`（1=违约，0=正常）。**评估指标**：违约率仅 20%，类别不均衡，因此以 精确率 / 召回率 / F1 / AUC 为主，准确率为辅。""")

code("""# 原始数据概览（train.csv 约 80 万行；首次读取需约 10-20 秒）
train_raw = pd.read_csv(DATA / "train.csv")
print("原始 train.csv:", train_raw.shape, "| testA.csv: 20 万行（无标签）")
print("\\n目标变量分布（违约率 %.1f%%）:" % (train_raw["isDefault"].mean() * 100))
print(train_raw["isDefault"].value_counts(normalize=True).round(3))

miss = train_raw.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
print("\\n缺失最多的 6 列:\\n", miss.head(6))
""")

code("""# W1 EDA 图（数据分布 / 相关性 / 等级违约率）
for f in ["eda_target_distribution.png", "eda_grade_default_rate.png",
          "eda_numeric_distributions.png", "eda_corr_heatmap.png"]:
    display(Image(str(OUT / "W1" / f), width=720))
""")

code("""# 数据质量报告（W1 输出）
print((OUT / "W1" / "data_quality_report.txt").read_text()[:1500])
""")

md("""## 2. 数据预处理（W1）

**原则：修格子，不砍样本。** 缺失值优先填充（中位数/众数/业务占位），异常值按业务规则封顶或删除，仅删 1,502 行（0.19%），样本保留率 99.8%。

> 完整重跑：`.venv/bin/python scripts/data_preprocessing.py`（含检测报告与处理，约 1-2 分钟）""")

code("""# 清洗前后对比
n_before = len(train_raw)
n_after = len(pd.read_csv(DATA / "train_clean.csv", usecols=["id"]))
print(f"清洗前 {n_before:,} 行 → 清洗后 {n_after:,} 行（删除 {n_before - n_after:,} 行 = 0.19%）")

clean_summary = pd.DataFrame({
    "处理项": ["annualIncome=0 视为缺失", "revolUtil>100 封顶", "dti 越界删行",
              "缺失极少列删行", "n0-n14 缺失", "employmentLength 缺失"],
    "处理方式": ["视为缺失并填充", "封顶到 100", "删除行", "删除行", "中位数填充", "填充'未知'"],
    "影响量": [229, 2719, 325, 1177, "—", 46375],
})
clean_summary
""")

code("""# 异常检测要点（先检测后处理：业务规则 + 3σ + IQR）
display(Image(str(OUT / "W1" / "outlier_boxplot.png"), width=720))
print((OUT / "W1" / "outlier_report.txt").read_text()[1500:3200])
""")

md("""## 3. 特征工程（W2）

四步（对应任务 3.1–3.4）：
1. **类别编码**：序数（grade/subGrade/employmentLength）、独热（homeOwnership/verificationStatus/purpose）、目标编码（regionCode，仅用训练集违约率防泄漏）
2. **数值变换**：右偏列 `log1p`（含年收入）；连续列 z-score 标准化
3. **特征创建**：`fico_avg`（上下限合并）、`credit_history_years`（信用历史长度）、`installment_income_ratio`（月供收入比）、2 个交互项（grade×dti、利率×金额）
4. **特征选择**：过滤法（相关系数）+ 嵌入法（随机森林重要性），60 列 → 保留 38 个

> 完整重跑：`.venv/bin/python scripts/W2/feature_engineering.py`（含随机森林重要性计算，约 1 分钟）""")

code("""# 特征工程最终产物
df = pd.read_csv(DATA / "train_final.csv")
print("train_final.csv:", df.shape, "（38 个特征 + 目标 isDefault）")
print("违约率: %.1f%%" % (df["isDefault"].mean() * 100))

# 新增特征清单（W2 创建）
new_feats = ["fico_avg", "credit_history_years", "installment_income_ratio",
             "grade_dti_inter", "interest_loan_inter"]
print("\\nW2 新增特征:", new_feats)
print("删除列（无信息/高基数）: id, policyCode, employmentTitle, postCode, title")
""")

code("""# 特征重要性（W2：随机森林嵌入法）
imp = pd.read_csv(OUT / "W2" / "feature_importance.csv", index_col=0)
imp.head(10)
""")

code("""display(Image(str(OUT / "W2" / "feature_importance.png"), width=720))
""")

md("""## 4. 模型构建与评估（W2）

**数据集划分（4.1）**：训练 70% / 验证 20% / 测试 10%，按目标变量分层，保证三集违约率都约 20%。
**基础模型（4.2）**：逻辑回归（线性基线）、随机森林、XGBoost。类别不均衡处理：LR/RF 用 `class_weight="balanced"`，XGB 用 `scale_pos_weight`。
**评估（4.3）**：准确率 / 精确率 / 召回率 / F1 / AUC。

> 完整重跑：`.venv/bin/python scripts/W2/model_training.py`（80 万行三模型，约 3-5 分钟）""")

code("""# W2 验证集三模型对比（index 为模型名）
comp_w2 = pd.read_csv(OUT / "W2" / "model_comparison.csv", index_col=0)
comp_w2.round(4)
""")

code("""# ROC / PR 曲线（三模型对比）
display(Image(str(OUT / "W2" / "roc_curves.png"), width=720))
display(Image(str(OUT / "W2" / "pr_curves.png"), width=720))
""")

md("""**怎么看这张对比表**：违约率 20% 下，准确率参考意义有限（全猜"正常"也有 80%）。更看 AUC（排序能力）与召回率（能不能抓住违约者）。树模型（RF/XGB）明显优于线性 LR，说明特征与目标之间以非线性关系为主。""")

md("""## 5. 模型优化（W3）

1. **随机搜索调参**（5 折交叉验证）：XGBoost 与 LightGBM 各 8 组参数组合，按验证集 AUC 取最优
2. **模型集成**：投票法 + Stacking（元模型处理类别不均衡），与单模型对比
3. **可解释性**：SHAP 全局特征归因

> 完整重跑：`.venv/bin/python scripts/W3/model_optimization.py`（含随机搜索与 SHAP，约 10-15 分钟）""")

code("""# W3 优化后对比（7 个候选）
comp_w3 = pd.read_csv(OUT / "W3" / "model_comparison_optimized.csv")
comp_w3.round(4)
""")

code("""# 随机搜索记录（前 6 组）
rs = pd.read_csv(OUT / "W3" / "random_search_results.csv")
rs.head(6)
""")

code("""# SHAP 全局归因
display(Image(str(OUT / "W3" / "shap_summary.png"), width=760))
shap_imp = pd.read_csv(OUT / "W3" / "shap_feature_importance.csv", index_col=0)
shap_imp.head(10)
""")

md("""**结论**：Stacking 验证集 AUC 0.7298 与 LightGBM（优化）0.7294 几乎持平，但 LGB 更可解释、部署简单，故选 **LightGBM(优化)** 为最终模型。

**最终参数**：`subsample=0.8, num_leaves=63, n_estimators=200, max_depth=8, learning_rate=0.05, colsample_bytree=0.7`""")

md("""## 6. 最终评估（W4）

**纪律**：测试集（79,850 人）从 W1 起全程未参与任何建模决策；最终模型训练完才碰它，保证成绩单可信。
**验证方法**：训练/验证/测试三集 AUC 对比（泛化分析）+ 学习曲线（是否过拟合、加数据是否还有用）。

> 完整重跑：`.venv/bin/python scripts/W4/model_final_eval.py`、`scripts/W4/learning_curve.py`""")

code("""# 测试集最终成绩单（阈值 0.5）
pd.read_csv(OUT / "W4" / "model_final_metrics.csv").round(4)
""")

code("""# 泛化对比 + 阈值对照
gen = pd.read_csv(OUT / "W4" / "generalization_compare.csv", index_col=0)
print("训练 / 验证 / 测试 AUC 差距小 → 无明显过拟合")
display(gen.round(4))
print("\\n阈值对照（0.3-0.7）:")
pd.read_csv(OUT / "W4" / "threshold_table.csv").round(3)
""")

code("""# 测试集可视化：混淆矩阵 / ROC / PR / 学习曲线
for f in ["confusion_matrix_test.png", "roc_curve_test.png",
          "pr_curve_test.png", "learning_curve.png"]:
    display(Image(str(OUT / "W4" / f), width=720))
""")

code("""# 活代码复现：用最终参数重训 LightGBM，验证测试集指标与成绩单一致（约 30 秒）
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from lightgbm import LGBMClassifier

X = df.drop(columns=["isDefault"]); y = df["isDefault"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
best_params = {"subsample": 0.8, "num_leaves": 63, "n_estimators": 200,
               "max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.7}
model = LGBMClassifier(**best_params, scale_pos_weight=scale_pos,
                       verbosity=-1, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]
pred = (p >= 0.5).astype(int)
print("复现: acc=%.4f prec=%.4f rec=%.4f f1=%.4f auc=%.4f" % (
    accuracy_score(y_test, pred), precision_score(y_test, pred),
    recall_score(y_test, pred), f1_score(y_test, pred), roc_auc_score(y_test, p)))
print("官方: acc=0.6560 prec=0.3254 rec=0.6749 f1=0.4390 auc=0.7249")
""")

md("""## 7. 业务结论与交付

### 阈值怎么选（6.2 风险阈值选择）
- 阈值 0.5（默认平衡档）：拒绝约 41% 的申请，拦下 67.5% 的违约者，误伤约 35% 的正常客户
- 阈值 0.3（宽进）：拦下 92.3% 违约者，但误伤翻倍，适合获客扩张期
- 阈值 0.7（严审）：精确率 47.4%（误伤少），但漏掉约 74% 违约者，适合风险收缩期
- **最终选择**：需结合坏账损失与放贷收益测算（当前数据无法精确测算，报告中给出框架与演示参数）

### 给业务方的话
1. 模型在从未见过的测试集上 AUC 0.7249，训练/验证/测试差距小，**没有过拟合**，可稳定推广到新客户
2. 与随机拒绝相比，同等拒绝比例下多拦下约 26 个百分点的违约者，同时少误伤约 6 个百分点
3. 最重要特征：subGrade（等级）、term（期限）、homeOwnership、fico（信用分）——**放贷时可优先核对这几项**
4. 局限：数据集字段有限（部分匿名/脱敏），AUC 0.72 属同数据规模下的合理水平；提升需接入征信局数据、还款行为序列特征
""")

code("""# 业务仪表板（阈值模拟器所在页面）
print("运行仪表板：streamlit run scripts/W4/app_dashboard.py")
print("页面：① 数据概览（EDA） ② 模型表现 ③ 阈值模拟器（滑块实时看精确率/召回率/拒绝占比）")
""")

md("""### 交付清单对照
- **代码**：`data_preprocessing.py`（W1）｜`feature_engineering.py`、`model_training.py`（W2）｜`model_optimization.py`（W3）｜`model_final_eval.py`、`learning_curve.py`、`app_dashboard.py`（W4）｜本 Notebook（main_pipeline）
- **报告**：数据探索报告 / 特征工程报告 / 模型评估报告 / 模型优化报告 / 最终项目报告（`reports/`）
- **可视化**：EDA 图（W1）、特征重要性（W2）、ROC/PR/混淆矩阵/学习曲线（W2/W4）、SHAP（W3）、Streamlit 仪表板（W4）
- **文档**：`README.md`、`requirements.txt`、`presentation.pptx`（待生成）
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

out = NB_DIR / "main_pipeline.ipynb"
nbf.write(nb, out)
print("已生成:", out, "| 单元格数:", len(cells))
