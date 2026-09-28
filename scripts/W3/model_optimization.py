# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
model_optimization.py - 模型优化（W3 交付物，阶段 5）
项目：信贷违约预测模型
输入：data/train_final.csv（特征工程最终产物）
输出：outputs/W3/（调优参数记录、优化前后对比表、SHAP 解释）

流程（对应项目任务 5.1-5.4 + 模型解释）：
  5.1 超参数调优：随机搜索（RandomizedSearchCV，配 5 折交叉验证）
      说明：网格搜索在 80 万行数据下组合爆炸、贝叶斯实现复杂收益有限，
      随机搜索能快速覆盖重要参数，实践中效果足够（报告中会说明选择理由）。
      调参对象：XGBoost 与 LightGBM（树模型参数多、提升空间大）；
      逻辑回归/随机森林保持 W2 基线（LR 参数少、RF 调参代价太高）。
  5.2 交叉验证：调参时用 StratifiedKFold(5)，每折保持类别比例；
      每个候选参数组合取 5 折平均 AUC 作为评分。
  5.3 模型集成：投票法（soft voting 概率平均）+ 堆叠法（base 预测作新特征，
      逻辑回归作 meta 学习器）；LightGBM 加入对比（梯度提升同族，更快）。
  5.4 调优后再训练：全部模型在验证集复测准确率/精确率/召回率/F1/AUC，
      输出"基线 vs 优化后"对比表。
  模型解释：SHAP TreeExplainer（最优树模型），全局特征归因图（summary + 均值表）。

防泄漏纪律与 W2 一致：K 折调参只在训练集内进行，验证集只用于最终复测，
10% 测试集仍全程不碰（留阶段 6）。
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 向上查找含 data/ 的目录作为项目根（脚本放 scripts/W3/ 下可正常运行）
BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W3"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "train_final.csv")
print(f"加载: train_final.csv {df.shape}，违约率 {df['isDefault'].mean()*100:.1f}%")

# ---------- 数据集划分（与 W2 model_training.py 完全一致，保证可比） ----------
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
print(f"划分: 训练 {len(X_train):,} / 验证 {len(X_val):,} / 测试 {len(X_test):,}（未使用）")

# 类别不平衡权重（与 W2 相同）：负样本数 / 正样本数
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {scale_pos:.2f}\n")

# ---------- 1) 随机搜索调参（XGBoost / LightGBM，5 折交叉验证） ----------
print("=" * 60)
print("5.1+5.2 随机搜索调参（5 折交叉验证，评分=平均 AUC）")
print("=" * 60)

def run_random_search(name, model, param_grid, X_tr, y_tr, n_iter=8, cv=5):
    t0 = time.time()
    rs = RandomizedSearchCV(
        model, param_grid, n_iter=n_iter, cv=StratifiedKFold(cv),
        scoring="roc_auc", random_state=42, n_jobs=1, verbose=0)
    rs.fit(X_tr, y_tr)
    # 调优参数记录：每个候选组合 + 5 折平均分
    rec = pd.DataFrame({
        "model": name,
        "mean_cv_auc": rs.cv_results_["mean_test_score"],
        "std_cv_auc": rs.cv_results_["std_test_score"],
        "params": [str(p) for p in rs.cv_results_["params"]],
    }).sort_values("mean_cv_auc", ascending=False)
    print(f"[{name}] 用时 {time.time()-t0:.0f}s | 最优 K 折平均 AUC = "
          f"{rs.best_score_:.4f}\n    最优参数: {rs.best_params_}")
    return rs, rec

xgb_params = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}
lgb_params = {
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

xgb_rs, xgb_rec = run_random_search(
    "XGBoost", XGBClassifier(scale_pos_weight=scale_pos, eval_metric="logloss",
                             random_state=42, n_jobs=-1, verbosity=0),
    xgb_params, X_train, y_train)
lgb_rs, lgb_rec = run_random_search(
    "LightGBM", LGBMClassifier(scale_pos_weight=scale_pos, verbosity=-1,
                               random_state=42, n_jobs=-1),
    lgb_params, X_train, y_train)

rec_all = pd.concat([xgb_rec, lgb_rec], ignore_index=True)
rec_all.to_csv(OUT / "random_search_results.csv", index=False)
print(f"调优参数记录已保存: {OUT / 'random_search_results.csv'}")

# ---------- 2) 训练全部模型：基线（W2 同参数）+ 优化后 + 集成 ----------
print("\n" + "=" * 60)
print("5.3+5.4 训练全部模型并在验证集复测（基线 vs 优化后 vs 集成）")
print("=" * 60)

# 先建各模型实例（Voting/Stacking 内部会 clone，共享实例安全）
lr_base = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=42)
rf_base = RandomForestClassifier(n_estimators=100, max_depth=12,
                                 class_weight="balanced",
                                 random_state=42, n_jobs=-1)
xgb_base = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                         scale_pos_weight=scale_pos, eval_metric="logloss",
                         random_state=42, n_jobs=-1, verbosity=0)
xgb_opt = XGBClassifier(**xgb_rs.best_params_, scale_pos_weight=scale_pos,
                        eval_metric="logloss", random_state=42,
                        n_jobs=-1, verbosity=0)
lgb_opt = LGBMClassifier(**lgb_rs.best_params_, scale_pos_weight=scale_pos,
                         verbosity=-1, random_state=42, n_jobs=-1)

models = {
    # 基线（与 W2 model_training.py 参数完全一致，用于对比）
    "LR(基线)": lr_base,
    "RF(基线)": rf_base,
    "XGB(基线)": xgb_base,
    # 优化后（随机搜索选出的最优参数）
    "XGB(优化)": xgb_opt,
    "LGB(优化)": lgb_opt,
    # 集成：投票法（4 模型概率平均）
    "Voting(集成)": VotingClassifier(
        estimators=[("lr", lr_base), ("rf", rf_base),
                    ("xgb", xgb_opt), ("lgb", lgb_opt)],
        voting="soft", n_jobs=1),
    # 集成：堆叠法（4 个 base 模型预测作特征，逻辑回归综合）
    "Stacking(集成)": StackingClassifier(
        estimators=[("lr", lr_base), ("rf", rf_base),
                    ("xgb", xgb_opt), ("lgb", lgb_opt)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
                         cv=5, n_jobs=1),
}

probs, rows = {}, []
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    p = model.predict_proba(X_val)[:, 1]
    probs[name] = p
    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_val, (p >= 0.5).astype(int)),
        "precision": precision_score(y_val, (p >= 0.5).astype(int)),
        "recall": recall_score(y_val, (p >= 0.5).astype(int)),
        "f1": f1_score(y_val, (p >= 0.5).astype(int)),
        "auc": roc_auc_score(y_val, p),
    })
    r = rows[-1]
    print(f"  {name:16s} 用时 {time.time()-t0:4.0f}s | acc={r['accuracy']:.4f} "
          f"prec={r['precision']:.4f} rec={r['recall']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}")

comp = pd.DataFrame(rows).sort_values("auc", ascending=False)
comp.to_csv(OUT / "model_comparison_optimized.csv", index=False)
print(f"\n优化前后对比表已保存: {OUT / 'model_comparison_optimized.csv'}")

# ---------- 3) SHAP 特征归因（最优树模型 + 训练集抽样 5 万行） ----------
print("\n" + "=" * 60)
print("模型解释：SHAP（全局特征归因）")
print("=" * 60)
best_tree = max(["XGB(优化)", "LGB(优化)"],
                key=lambda n: roc_auc_score(y_val, probs[n]))
print(f"选最优树模型: {best_tree}（验证集 AUC {roc_auc_score(y_val, probs[best_tree]):.4f}）")

try:
    import shap
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_train), size=min(50000, len(X_train)), replace=False)
    X_sample = X_train.iloc[sample_idx]
    explainer = shap.TreeExplainer(models[best_tree])
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # 二分类：取正类（违约）的归因

    # 全局：特征归因 summary 图 + 均值表
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    shap_imp = pd.Series(np.abs(shap_values).mean(axis=0),
                         index=X_sample.columns).sort_values(ascending=False)
    shap_imp.to_csv(OUT / "shap_feature_importance.csv", header=["mean_abs_shap"])
    print(f"全局归因已保存: shap_summary.png + shap_feature_importance.csv")
    print(f"SHAP Top 5: {[(c, round(v, 4)) for c, v in shap_imp.head(5).items()]}")

except ImportError:
    print("未安装 shap，跳过 SHAP 解释（pip install shap）")

print("\nW3 模型优化全部完成")
print(f"产出: {OUT}")
