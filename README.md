# 信贷违约预测模型

基于用户历史数据与行为特征，构建机器学习模型预测借款人违约概率，为金融机构风控决策（放款审批、风险定价、额度管理）提供数据支持。

最终模型：**LightGBM（随机搜索优化）**，测试集 AUC **0.7249**（阈值 0.5 下准确率 0.656 / 精确率 0.325 / 召回率 0.675 / F1 0.439）。

## 获取代码

```bash
git clone https://github.com/1497071625-dev/credit-default-risk-model.git
cd credit-default-risk-model
```

仓库里没有原始数据（约 1 GB），要自己下载，见下面「数据集」一节。

## 项目结构

```
credit-default-risk-model/
├── data/         原始数据与清洗、特征工程产物（体积大，未入库，按「数据集」一节自行下载）
├── notebooks/    main_pipeline.ipynb：一键复现全流程（W1–W4，读出产物，几分钟跑完）
├── outputs/      脚本产物（W1 清洗与 EDA / W2 特征与基线 / W3 优化与 SHAP / W4 最终评估、决策分析、仪表板）
├── reports/      项目文档（环境配置、数据探索、特征工程、模型评估、模型优化、最终报告，以及 W1–W4 周报；仓库留 md 源稿，docx 交付版在交付包里）
├── scripts/      可执行脚本（scripts/W1–W4，按下文顺序运行）
├── README.md     本文件
└── requirements.txt
```

仓库只放代码、报告和图表产物。原始数据、虚拟环境 `.venv/`、运行缓存 `.cache/`、旧版留档，以及逐笔预测明细 `outputs/W4/test_predictions*.csv`（约 7 MB，脚本可重新生成）都不入库。

## 数据集

- 来源：阿里天池信贷违约数据集（[链接](https://tianchi.aliyun.com/dataset/140861)）
- 训练集 `data/train.csv`：80 万条 × 47 字段，含目标变量 `isDefault`（0=正常，1=违约），违约率 20%
- 测试集 `data/testA.csv`：20 万条 × 46 字段，不含目标变量
- 原始文件较大（train 约 167MB），未纳入版本控制；下载后放回 `data/` 即可

## 环境搭建

要求：Python 3.13（项目在 macOS arm64 上开发验证）。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS 下若 `import xgboost` 报错，需先安装 OpenMP 运行时：`brew install libomp`（详见 `reports/环境配置.md`）。

## 快速复现（推荐）

一条命令跑完 W1–W4 全流程（数据加载 → 清洗 → EDA → 特征工程 → 基线建模 → 优化 → 最终评估），在项目根目录执行：

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/main_pipeline.ipynb
```

Notebook 优先读取各阶段已生成的中间产物（快路径，约 1 分钟）；如需从原始数据完整重跑，按单元格内标注的脚本命令逐阶段执行。

## 分步运行指南

```bash
# ① 数据加载与质量检查（W1）→ outputs/W1/data_quality_report.txt
.venv/bin/python scripts/W1/01_data_exploration.py

# ② 异常值检测 + 数据清洗（W1）→ outputs/W1/outlier_report.txt、data/train_clean.csv、data/testA_clean.csv
.venv/bin/python scripts/W1/data_preprocessing.py

# ③ EDA 可视化（W1）→ outputs/W1/eda_*.png、outlier_boxplot.png
.venv/bin/python scripts/W1/eda_visualization.py
.venv/bin/python scripts/W1/eda_income_log_compare.py

# ④ 特征工程（W2）→ data/train_final.csv、data/testA_final.csv、outputs/W2/feature_importance.png
.venv/bin/python scripts/W2/feature_engineering.py

# ⑤ 三模型基线训练与评估（W2）→ outputs/W2/model_comparison.csv、roc_curves.png、pr_curves.png、confusion_matrix_xgb.png
.venv/bin/python scripts/W2/model_training.py

# ⑥ 特征相关性与共线性分析（W2，对应《特征工程报告》第 7 节）→ outputs/W2/correlation_heatmap.png、correlation_top15.png
.venv/bin/python scripts/W2/correlation_analysis.py

# ⑦ 模型优化（W3：随机搜索调参、K 折交叉验证、投票/堆叠集成、SHAP）→ outputs/W3/
.venv/bin/python scripts/W3/model_optimization.py

# ⑦-补 优化结果配图（W3，不训练模型，秒出图）→ outputs/W3/random_search_auc.png、model_comparison_optimized.png
.venv/bin/python scripts/W3/optimization_plots.py

# ⑦-补2 调参方法对比（W3，网格/随机/贝叶斯 TPE 等预算对照，约 22 分钟；末尾加 --from-cache 可跳过训练直接重画图）
.venv/bin/python scripts/W3/tuning_method_compare.py

# ⑦-补2b 调参方法全组合对照（W3，XGBoost + LightGBM × 网格/随机/贝叶斯共 6 个组合，约 30 分钟；只重画图加 --from-cache --big-cache）
.venv/bin/python scripts/W3/tuning_full_combo.py

# ⑦-补3 模型差异显著性检验（W3，DeLong + Bootstrap 1000 次；首次运行需重新训练七个模型并缓存）
.venv/bin/python scripts/W3/model_significance_test.py

# ⑦-补4 训练/预测耗时实测（W3，回答“集成模型贵多少”——训练一次约 6 分钟 vs 单模型约 6 秒）→ outputs/W3/prediction_cost.csv
.venv/bin/python scripts/W3/prediction_cost.py

# ⑧ 最终模型测试集评估（W4）→ outputs/W4/model_final_metrics.csv、generalization_compare.csv、threshold_table.csv、ROC/PR/混淆矩阵图
.venv/bin/python scripts/W4/model_final_eval.py

# ⑨ 学习曲线（W4）→ outputs/W4/learning_curve.png
.venv/bin/python scripts/W4/learning_curve.py

# ⑩ 生成测试集预测概率（W4，仪表板数据源）→ outputs/W4/test_predictions.csv
.venv/bin/python scripts/W4/prepare_test_predictions.py

# ⑪ 概率校准（W4）→ outputs/W4/calibration_compare.csv、calibration_bins.csv、calibration_curve.png、test_predictions_calibrated.csv
.venv/bin/python scripts/W4/probability_calibration.py

# ⑫ 特征数量 vs AUC（W4）→ outputs/W4/feature_count_auc.csv、feature_count_auc.png
.venv/bin/python scripts/W4/feature_count_auc.py

# ⑬ 期望损失决策框架与敏感性（W4）→ outputs/W4/expected_loss_summary.csv、expected_loss_sensitivity.csv、expected_loss_sensitivity.png
.venv/bin/python scripts/W4/expected_loss_framework.py

# ⑭ 阈值金额扫描（W4）→ outputs/W4/threshold_sweep.csv、threshold_equiv.csv、threshold_sweep.png
.venv/bin/python scripts/W4/threshold_sweep.py

# ⑮ 阈值改在验证集上选（W4，约 30 秒）→ outputs/W4/threshold_select_on_val.csv
.venv/bin/python scripts/W4/threshold_select_on_val.py

# ⑯ KS 与头部捕获率（W4，秒级）→ outputs/W4/ks_summary.csv、head_capture.csv
.venv/bin/python scripts/W4/head_capture.py

# ⑰ 时间外验证（W4，按放款日期切分重训重评，几分钟）→ outputs/W4/oot_validation.csv、oot_calibration.csv
.venv/bin/python scripts/W4/oot_validation.py

# ⑱ 人工复审带（W4，秒级）→ outputs/W4/review_band.csv、review_band_key.csv
.venv/bin/python scripts/W4/review_band.py

# ⑲ 差异化定价：低风险客户的让价空间（W4，秒级）→ outputs/W4/pricing_headroom.csv
.venv/bin/python scripts/W4/pricing_headroom.py

# ⑳ 生成离线仪表板（W4）→ outputs/W4/dashboard.html（单文件，双击即开）
.venv/bin/python scripts/W4/dashboard_chart_data.py   # 先从 outputs 抽绘图数据 → dashboard_charts.json
.venv/bin/python scripts/W4/build_dashboard_html.py   # 再生成 HTML（内联 SVG，不依赖图表库）

# ㉑ 报告与周报配图（W4，秒级）→ outputs/W4/fig_perf_row.png（混淆矩阵/ROC/PR 并排）
.venv/bin/python scripts/W4/report_figures.py

# ㉒ 重新生成主流程 Notebook（W4）→ notebooks/main_pipeline.ipynb（生成后需执行一遍）
.venv/bin/python scripts/W4/build_main_pipeline.py
```

> 报告与周报的 docx 由同名 md 生成（周报 `--profile weekly`，报告 `--profile final8`，依赖 python-docx），演示 PPT 由 SVG 源导出（依赖 python-pptx）；生成脚本与 PPT 工程留在开发环境。仓库只留 md 源稿与能直接预览的 `outputs/W4/*.pdf`，docx 与 pptx 是发出去的交付版，不入库。

也可以直接跑整合版 Notebook：`notebooks/main_pipeline.ipynb`（快速路径读 `data/` 与 `outputs/` 已有产物，几分钟出结果）。

## 交互仪表板

```bash
.venv/bin/python -m streamlit run scripts/W4/app_dashboard.py
```

四个页签：① 数据概览（W1 数据质量与 EDA）② 模型表现（测试集成绩单与 KS、头部捕获率、泛化、性能图、时间外验证、模型对比、特征重要性、调参、特征数量）③ 放贷方案① 单一线：一条线管所有人（阈值滑块，人数账 + 金额账）④ 放贷方案② 拆线：按期限设两条线（概率校准、分箱对照、逐笔算账、和 ③ 的区别、敏感性滑块与对照表、两条线对比、人工复审带、上线清单）。

金额有两个口径，别混用。**净增益**（页签③，对比基准是不用模型、全批放款）：`避免坏账`＝Σ(被拒且真违约) × 贷款金额 × (1−回收率)；`放弃收入`＝Σ(被拒且正常) × 贷款金额 × 年净息差 × 期限年；`净增益`＝避免坏账 − 放弃收入。**期望利润**（页签④）：给每笔被批准的贷款算 (1−p) × 净息差 × 期限 − p × (1−回收率) 后求和。回收率与净息差是模拟值（默认 30% / 6%，数据集中没有这两列，页签④ 的敏感性表就是在换这两档），上线前要换成机构财务口径；其余金额全部来自测试集真实数据。

不想装环境的话，用离线单文件版：`outputs/W4/dashboard.html` 由 `scripts/W4/dashboard_chart_data.py`（抽数）+ `scripts/W4/build_dashboard_html.py` + `scripts/W4/svg_charts.py`（画图）生成，四个页签内容与 Streamlit 版一致，③ 的阈值、回收率、净息差三个滑块和 ④ 的回收率、净息差滑块都能拖，双击文件即可打开，Ctrl/⌘+P 还能存成 PDF。①② 的图是内联 SVG，鼠标悬浮即出读数；③④ 的老位图点一下放大。

## 交付物（W4）

| 交付物 | 文件 | 要求 |
|---|---|---|
| 最终项目报告 | `reports/最终项目报告.md`（交付版 docx 同名） | 10–15 页，含业务建议（放贷线与差异化定价）与 4 张图（实测 14 页） |
| W4 周报 | `reports/W4_周报_2026-09-27.md`（交付版 docx 同名） | 详细总结版，含 4 张图（放贷方案①② 各一节），单独发 |
| 业务仪表板 | `outputs/W4/dashboard.html`（离线单文件）｜`scripts/W4/app_dashboard.py`（Streamlit） | Plotly/Dash 或 Streamlit |
| 项目汇报演示 | `outputs/W4/信贷违约模型与审批线_W4汇报.pdf`（可编辑 pptx 在交付包） | presentation.pptx，10–15 页 |
| 代码 | `scripts/W1–W4/` + `notebooks/main_pipeline.ipynb` | 五个规定脚本 + 全流程 Notebook |
| 图表产物 | `outputs/W4/*.png`、`*.csv` | EDA / ROC / PR / 混淆矩阵 / 学习曲线 / 校准 / 敏感性 |

## 文档

- `reports/环境配置.md`：环境与依赖配置说明
- `reports/数据探索+EDA可视化报告.md`：数据质量检查、清洗策略、EDA 发现
- `reports/特征工程报告.md`：特征编码 / 变换 / 创建 / 选择决策及防泄漏说明
- `reports/模型评估报告.md`：三模型基线对比与业务解读
- `reports/模型优化报告.md`：调参、K 折交叉验证、投票/堆叠集成与 SHAP 解释
- `reports/最终项目报告.md`：完整项目总结（含头部区分度、时间外验证、风险阈值与定价、人工复审带、局限性；含 4 张图，导出 15 页）
- 周报：`reports/W1_周报_2026-09-07.md`、`W3_周报_2026-09-21.md`、`W4_周报_2026-09-27.md`，以及 `W2_周报_2026-09-14.docx`（W2 当时只出了 docx，没有 md 源稿）
- 演示 PPT：`outputs/W4/信贷违约模型与审批线_W4汇报.pdf`（15 页，可编辑的 pptx 在交付包里）
