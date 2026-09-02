"""
prepare_test_predictions.py - 生成测试集预测概率（供 Streamlit 阈值模拟器使用）
项目：信贷违约预测模型
输入：data/train_final.csv
输出：outputs/W4/test_predictions.csv（测试集 79850 人的真实标签 + 违约概率）

说明：最终模型在测试集上的完整概率分布只存了 0.3/0.5/0.7 等几个离散点；
仪表盘的阈值滑块需要任意阈值都能即时算指标，所以把概率一次性算好存下来。
切分逻辑与 model_final_eval.py 完全一致（同一个随机种子），保证是同一份测试集。
同时按行位置带出贷款金额与期限（来自 train_clean.csv，供阈值模拟器做"金额账"演示）。
"""
from pathlib import Path
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W4"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "train_final.csv")
X = df.drop(columns=["isDefault"])
y = df["isDefault"]

# 贷款金额/期限来自清洗后、特征工程前的表（train_final 中 loanAmnt 已 log+标准化，
# 无法还原原始金额；两表行序全链路一致，可按位置对齐）
clean_info = pd.read_csv(DATA / "train_clean.csv", usecols=["loanAmnt", "term"])

# 与 W2/W4 完全一致的切分（保证是同一份测试集）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
print(f"训练 {len(X_train):,} / 验证 {len(X_val):,} / 测试 {len(X_test):,}")

# 最终模型参数（W3 随机搜索最优）
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
best_params = {"subsample": 0.8, "num_leaves": 63, "n_estimators": 200,
               "max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.7}
model = LGBMClassifier(**best_params, scale_pos_weight=scale_pos,
                       verbosity=-1, random_state=42, n_jobs=-1)
t0 = time.time()
model.fit(X_train, y_train)
print(f"模型训练完成，用时 {time.time()-t0:.0f}s")

p_test = model.predict_proba(X_test)[:, 1]
pred = pd.DataFrame({"y_true": y_test.reset_index(drop=True),
                     "prob_default": p_test,
                     "loanAmnt": clean_info.loc[X_test.index, "loanAmnt"].values,
                     "term": clean_info.loc[X_test.index, "term"].values})
pred.to_csv(OUT / "test_predictions.csv", index=False)
print(f"已保存: {OUT / 'test_predictions.csv'} ({len(pred):,} 行)")
print(f"违约率 {pred['y_true'].mean()*100:.1f}% | 概率均值 {pred['prob_default'].mean():.4f}")
print(f"贷款金额范围 {pred['loanAmnt'].min():,.0f}-{pred['loanAmnt'].max():,.0f} 元")
