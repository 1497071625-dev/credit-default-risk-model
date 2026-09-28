"""
prediction_cost.py - 各模型的“训练耗时 / 预测耗时”实测（阶段 5.4 补充）
项目：信贷违约预测模型
输入：data/train_final.csv
输出：outputs/W3/prediction_cost.csv

要回答的问题：报告里“多花约 40 倍算力换 0.0004 的 AUC，不划算”这句话，
      必须说清“算力”指的是什么——
        · 训练耗时：一次性成本，调参/换模型时才付；
        · 预测耗时：每次给客户打分都要付，才是上线后的持续成本。
      原记录只有“训练+预测合计”，这里把两者分开实测。

口径：与 W2/W3 完全一致的数据划分（7:2:1，stratify，random_state=42）、
      与 model_optimization.py 完全一致的模型参数。预测耗时取 3 次中的最好值
      （排除系统抖动，测的是“这套模型最快能多快打完 15.97 万人”）。
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W3"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "train_final.csv")
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, _ = train_test_split(X_val, test_size=1 / 3, stratify=y_val, random_state=42)
y_val = y.loc[X_val.index]
print(f"训练集 {len(X_train):,} 行 / 验证集 {len(X_val):,} 行")

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=12,
                            class_weight="balanced", random_state=42, n_jobs=-1)
xgb_base = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                         scale_pos_weight=scale_pos, eval_metric="logloss",
                         random_state=42, n_jobs=-1, verbosity=0)
xgb_opt = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.7, min_child_weight=3,
                        scale_pos_weight=scale_pos, eval_metric="logloss",
                        random_state=42, n_jobs=-1, verbosity=0)
lgb_opt = LGBMClassifier(n_estimators=200, max_depth=8, num_leaves=63, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7,
                         scale_pos_weight=scale_pos, verbosity=-1,
                         random_state=42, n_jobs=-1)

MODELS = {
    "XGB(基线)": xgb_base,
    "XGB(优化)": xgb_opt,
    "LGB(优化)": lgb_opt,
    "Voting(集成)": VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_opt), ("lgb", lgb_opt)],
        voting="soft", n_jobs=1),
    "Stacking(集成)": StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("xgb", xgb_opt), ("lgb", lgb_opt)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        cv=5, n_jobs=1),
}

rows = []
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_s = time.time() - t0

    times = []
    for _ in range(3):
        t1 = time.time()
        p = model.predict_proba(X_val)[:, 1]
        times.append(time.time() - t1)
    predict_s = min(times)

    auc = roc_auc_score(y_val, p)
    rows.append({"model": name, "train_seconds": round(train_s, 1),
                 "predict_seconds": round(predict_s, 3),
                 "predict_runs_seconds": " / ".join(f"{t:.3f}" for t in times),
                 "rows": len(X_val), "auc_check": round(auc, 4)})
    print(f"  {name:16s} 训练 {train_s:6.1f}s | 预测 {predict_s:5.2f}s（3 次：{rows[-1]['predict_runs_seconds']}）| AUC {auc:.4f}")

out = pd.DataFrame(rows)
out.to_csv(OUT / "prediction_cost.csv", index=False)
print("\n已保存:", OUT / "prediction_cost.csv")

base_p = out.loc[out.model == "LGB(优化)", "predict_seconds"].iloc[0]
stk_p = out.loc[out.model == "Stacking(集成)", "predict_seconds"].iloc[0]
print(f"预测耗时：LGB(优化) {base_p:.2f}s vs Stacking {stk_p:.2f}s = {stk_p / base_p:.1f} 倍")
