"""
model_final_eval.py - 最终模型评估（W4 交付物，阶段 6）
项目：信贷违约预测模型
输入：data/train_final.csv（特征工程最终产物）
输出：outputs/W4/（测试集最终成绩单、混淆矩阵、ROC/PR 曲线、阈值对照表）

流程（对应项目任务 6.1-6.2）：
  6.1 测试集最终评估：用最终模型（LightGBM 优化）预测从未碰过的 10% 测试集，
      输出准确率/精确率/召回率/F1/AUC + 混淆矩阵 + ROC/PR 曲线。
  6.2 阈值选择：测试集上给出 0.3-0.7 各阈值下的精确率/召回率/F1 对照表，
      供业务按"严审 vs 宽进"目标选择。

关键纪律：
  - 数据划分与 W2 完全一致（同样的随机种子与参数），保证测试集就是
    从 W1 起从未参与任何建模决策的那 7.985 万人。
  - 脚本先在验证集上复测一遍：若验证集数字与 W3 的 LGB(优化) 行一致
    （AUC≈0.729），即证明切分与参数无偏差，测试集结果可信。

最终模型：LightGBM（W3 随机搜索最优参数，5 折平均 AUC=0.7249）
  参数：subsample=0.8, num_leaves=63, n_estimators=200, max_depth=8,
        learning_rate=0.05, colsample_bytree=0.7
  选择理由：与 Stacking(集成) 的验证集 AUC（0.7298 vs 0.7294）几乎无差，
  但 LGB 可解释（SHAP）、部署简单，业务交付更合适。
"""
from pathlib import Path
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix)
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
print(f"加载: train_final.csv {df.shape}，违约率 {df['isDefault'].mean()*100:.1f}%")

# ---------- 6.1a 数据集划分：与 W2 完全一致（保证测试集是同一份未使用的数据） ----------
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
print(f"\n[6.1] 划分: 训练 {len(X_train):,} / 验证 {len(X_val):,} / 测试 {len(X_test):,}")
print(f"      各集违约率: "
      f"训练 {y_train.mean()*100:.1f}% / 验证 {y_val.mean()*100:.1f}% / 测试 {y_test.mean()*100:.1f}%")

# ---------- 最终模型：LightGBM（W3 随机搜索最优参数） ----------
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
best_params = {"subsample": 0.8, "num_leaves": 63, "n_estimators": 200,
               "max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.7}
final_model = LGBMClassifier(**best_params, scale_pos_weight=scale_pos,
                             verbosity=-1, random_state=42, n_jobs=-1)

t0 = time.time()
print("\n训练最终模型 LightGBM(优化) ...")
final_model.fit(X_train, y_train)
print(f"  用时 {time.time()-t0:.0f}s")

# ---------- 6.1b 验证集复测（sanity check：应与 W3 的 LGB(优化) 行一致） ----------
p_val = final_model.predict_proba(X_val)[:, 1]
val_metrics = {
    "accuracy": accuracy_score(y_val, (p_val >= 0.5).astype(int)),
    "precision": precision_score(y_val, (p_val >= 0.5).astype(int)),
    "recall": recall_score(y_val, (p_val >= 0.5).astype(int)),
    "f1": f1_score(y_val, (p_val >= 0.5).astype(int)),
    "auc": roc_auc_score(y_val, p_val),
}
print("\n[校验] 验证集复测（应与 W3 的 LGB(优化) 行一致）:")
print(f"  acc={val_metrics['accuracy']:.4f} prec={val_metrics['precision']:.4f} "
      f"rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} "
      f"auc={val_metrics['auc']:.4f}")
print("  W3 参考: acc=0.6566 prec=0.3271 rec=0.6825 f1=0.4422 auc=0.7294")

# ---------- 6.1c 测试集最终评估（从未碰过的 7.985 万人） ----------
print("\n" + "=" * 60)
print("6.1 测试集最终评估（最终成绩单）")
print("=" * 60)
p_test = final_model.predict_proba(X_test)[:, 1]
y_pred = (p_test >= 0.5).astype(int)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, p_test),
}
pd.DataFrame([metrics]).to_csv(OUT / "model_final_metrics.csv", index=False)
for k, v in metrics.items():
    print(f"  {k:9s}: {v:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted: Not Default", "Predicted: Default"])
ax.set_yticklabels(["Actual: Not Default", "Actual: Default"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14)
ax.set_title("Test Confusion Matrix (threshold 0.5)")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix_test.png", dpi=120)
plt.close()
print(f"混淆矩阵已保存: {OUT / 'confusion_matrix_test.png'}")

# ROC 曲线（测试集）
fpr, tpr, _ = roc_curve(y_test, p_test)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"LGB (final) AUC={metrics['auc']:.3f}", lw=2)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random (0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Test ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "roc_curve_test.png", dpi=120)
plt.close()
print(f"ROC 曲线已保存: {OUT / 'roc_curve_test.png'}")

# PR 曲线（测试集）
prec, rec, _ = precision_recall_curve(y_test, p_test)
baseline = y_test.mean()
# 与 W2 相同的两点处理，去掉左端竖线：
# 1) rec[:-1]/prec[:-1]: 去掉 sklearn 追加的 (recall=0, precision=1) 末点；
# 2) 只画 recall>=0.01 的点：更左端只有寥寥几个人被预测为正，
#    precision 是纯统计噪声（如 1/3、2/3 乱跳），对业务无意义且难看。
mask = rec[:-1] >= 0.01
plt.figure(figsize=(6, 5))
plt.plot(rec[:-1][mask], prec[:-1][mask], label="LGB (final)", lw=2)
plt.axhline(baseline, color="gray", ls="--",
            label=f"All-default baseline ({baseline:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Test PR Curve")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT / "pr_curve_test.png", dpi=120)
plt.close()
print(f"PR 曲线已保存: {OUT / 'pr_curve_test.png'}")

# ---------- 6.1d 泛化能力分析：训练/验证/测试三集对比 ----------
p_train = final_model.predict_proba(X_train)[:, 1]
train_metrics = {
    "accuracy": accuracy_score(y_train, (p_train >= 0.5).astype(int)),
    "precision": precision_score(y_train, (p_train >= 0.5).astype(int)),
    "recall": recall_score(y_train, (p_train >= 0.5).astype(int)),
    "f1": f1_score(y_train, (p_train >= 0.5).astype(int)),
    "auc": roc_auc_score(y_train, p_train),
}
gen = pd.DataFrame([train_metrics, val_metrics, metrics],
                   index=["train", "val", "test"])
gen.to_csv(OUT / "generalization_compare.csv")
print("\n" + "=" * 60)
print("6.1d 泛化能力分析（训练/验证/测试三集对比，阈值 0.5）")
print("=" * 60)
print(gen.round(4).to_string())
print(f"泛化对比表已保存: {OUT / 'generalization_compare.csv'}")
print("  (训练明显高于验证/测试 = 过拟合；三集接近 = 泛化正常)")

# ---------- 6.2 阈值对照表（业务选择用） ----------
print("\n" + "=" * 60)
print("6.2 阈值对照表（业务按风控目标选择）")
print("=" * 60)
rows = []
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (p_test >= th).astype(int)
    rows.append({
        "threshold": th,
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "default_rate": pred.mean(),
    })
th_tab = pd.DataFrame(rows)
th_tab.to_csv(OUT / "threshold_table.csv", index=False)
print(th_tab.round(4).to_string(index=False))
print(f"阈值对照表已保存: {OUT / 'threshold_table.csv'}")

print("\nW4 测试集最终评估完成")
print(f"产出: {OUT}")
