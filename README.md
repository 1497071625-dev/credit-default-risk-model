# 信贷违约预测模型

基于用户历史数据与行为特征，构建机器学习模型预测借款人违约概率，为金融机构风控决策（放款审批、风险定价、额度管理）提供数据支持。

最终模型：**LightGBM（随机搜索优化）**，测试集 AUC **0.7249**（阈值 0.5 下准确率 0.656 / 精确率 0.325 / 召回率 0.675 / F1 0.439）。

## 项目结构

```
Ali-DA/
├── data/         原始数据与各阶段产物（train_clean → encoded → scaled → final）
├── notebooks/    main_pipeline.ipynb：一键复现全流程（W1–W4）
├── outputs/      脚本产物（W1 清洗与 EDA / W2 特征与基线 / W3 优化与 SHAP / W4 最终评估）
├── reports/      项目文档（环境配置、数据探索、特征工程、模型评估、模型优化、最终报告、周报）
├── scripts/      可执行脚本（scripts/W1–W4，按下文顺序运行）
├── README.md     本文件
└── requirements.txt
```

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

# ⑥ 模型优化（W3：随机搜索调参、K 折交叉验证、投票/堆叠集成、SHAP）→ outputs/W3/
.venv/bin/python scripts/W3/model_optimization.py

# ⑦ 最终模型测试集评估（W4）→ outputs/W4/model_final_metrics.csv、generalization_compare.csv、threshold_table.csv、ROC/PR/混淆矩阵图
.venv/bin/python scripts/W4/model_final_eval.py

# ⑧ 学习曲线（W4）→ outputs/W4/learning_curve.png
.venv/bin/python scripts/W4/learning_curve.py

# ⑨ 生成测试集预测概率（W4，仪表板数据源）→ outputs/W4/test_predictions.csv
.venv/bin/python scripts/W4/prepare_test_predictions.py
```

## 交互仪表板

```bash
.venv/bin/python -m streamlit run scripts/W4/app_dashboard.py
```

三个页签：① 数据概览（W1 数据质量与 EDA）② 模型表现（W2 特征重要性 vs W3 SHAP、测试集评估）③ 阈值模拟器（人数账 + 金额账演示测算）。

页签③金额账口径（对比基准：不用模型、全批放款）：`避免坏账`＝Σ(被拒且真违约) × 贷款金额 × (1−回收率)；`放弃收入`＝Σ(被拒且正常) × 贷款金额 × 年净息差 × 期限年；`净增益`＝避免坏账 − 放弃收入。回收率与净息差为演示参数（默认 50% / 5%，数据集中无此字段），需由机构提供真实值替换；其余金额全部来自测试集真实数据。

## 文档

- `reports/环境配置.md`：环境与依赖配置说明
- `reports/数据探索报告.md`：数据质量检查、清洗策略、EDA 发现
- `reports/特征工程报告.md`：特征编码 / 变换 / 创建 / 选择决策及防泄漏说明
- `reports/模型评估报告.md`：三模型基线对比与业务解读
- `reports/模型优化报告.md`：调参、K 折交叉验证、投票/堆叠集成与 SHAP 解释
- `reports/最终项目报告.md`：完整项目总结（含业务建议、风险阈值选择、局限性）
