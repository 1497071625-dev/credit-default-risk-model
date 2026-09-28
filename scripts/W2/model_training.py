# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
model_training.py - 模型训练与评估（W2 交付物，阶段 4）
项目：信贷违约预测模型
输入：data/train_final.csv（特征工程产物）
输出：outputs/W2/model_comparison.csv（验证集性能对比表）

流程（对应项目任务 4.1-4.4）：
  4.1 数据集划分：训练 70% / 验证 20% / 测试 10%（按目标变量分层，保持类别比例）
  4.2 基础模型：逻辑回归（线性基准）、随机森林、XGBoost（树模型）
  4.3 评估指标：准确率、精确率、召回率、F1、AUC-ROC（类别不平衡，不全看准确率）
  4.4 模型对比：三模型验证集性能对比表

类别不平衡处理：三个模型都做同等处理
  - 逻辑回归 / 随机森林：class_weight="balanced"
  - XGBoost：scale_pos_weight = 负样本数 / 正样本数
"""
from pathlib import Path
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix, auc)
from xgboost import XGBClassifier

# 向上查找含 data/ 的目录作为项目根（脚本放 scripts/W2/ 下可正常运行）
BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W2"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "train_final.csv")
print(f"加载: train_final.csv {df.shape}，违约率 {df['isDefault'].mean()*100:.1f}%")

# ---------- 4.1 数据集划分（70/20/10，分层保持类别比例） ----------
X = df.drop(columns=["isDefault"])
y = df["isDefault"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=42)
print(f"\n[4.1] 划分: 训练 {len(X_train):,} / 验证 {len(X_val):,} / 测试 {len(X_test):,}")
print(f"      各集违约率: "
      f"训练 {y_train.mean()*100:.1f}% / 验证 {y_val.mean()*100:.1f}% / 测试 {y_test.mean()*100:.1f}%")

# ---------- 4.2 + 4.3 训练三模型并评估（验证集） ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                              random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12,
                                            class_weight="balanced",
                                            n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                             eval_metric="logloss", n_jobs=-1, random_state=42),
}

METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
results = {}
probs = {}

for name, model in models.items():
    t0 = time.time()
    print(f"\n训练 {name} ...")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    probs[name] = y_prob
    results[name] = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_prob),
    }
    print(f"  用时 {time.time()-t0:.0f}s | "
          f"acc={results[name]['accuracy']:.3f} prec={results[name]['precision']:.3f} "
          f"rec={results[name]['recall']:.3f} f1={results[name]['f1']:.3f} auc={results[name]['auc']:.3f}")

# ---------- 4.4 对比表 ----------
comp = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "auc"]]
comp = comp.round(4)
print("\n===== 验证集性能对比（0-1，越高越好） =====")
print(comp.to_string())
comp.to_csv(OUT / "model_comparison.csv")
print(f"\n已保存: {OUT / 'model_comparison.csv'}")

# ---------- 模型性能可视化（交付物：ROC/PR 曲线、混淆矩阵） ----------
# 1) ROC 曲线：三模型一条图，AUC 越大越靠左上
fig, ax = plt.subplots(figsize=(7, 6))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_val, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (Validation set)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "roc_curves.png", dpi=120)
plt.close(fig)
print("已保存: roc_curves.png（三模型 ROC 对比）")

# 2) PR 曲线：类别不平衡下比 ROC 更敏感；灰色虚线 = 全猜违约的基准
fig, ax = plt.subplots(figsize=(7, 6))
for name, p in probs.items():
    prec, rec, _ = precision_recall_curve(y_val, p)
    # 两点处理，让曲线真实且干净：
    # 1) rec[:-1]/prec[:-1]: sklearn 会在末尾追加 (recall=0, precision=1)，
    #    直接全画会把最后一段连成 x=0 处的一条贯穿竖线，去掉；
    # 2) 只画 recall>=0.01 的点：更左端只有寥寥几个人被预测为正，
    #    precision 是纯统计噪声（如 1/3、2/3 乱跳），对业务无意义且难看。
    mask = rec[:-1] >= 0.01
    ax.plot(rec[:-1][mask], prec[:-1][mask], label=name)
ax.axhline(y=y_val.mean(), color="gray", linestyle="--",
           label=f"Baseline ({y_val.mean():.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("PR Curves (Validation set)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "pr_curves.png", dpi=120)
plt.close(fig)
print("已保存: pr_curves.png（三模型 PR 对比）")

# 3) 混淆矩阵：只画最优模型 XGBoost（阈值 0.5）
best = "XGBoost"
y_pred_best = (probs[best] >= 0.5).astype(int)
cm = confusion_matrix(y_val, y_pred_best)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred Normal", "Pred Default"])
ax.set_yticklabels(["True Normal", "True Default"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14)
ax.set_title(f"{best} Confusion Matrix (threshold=0.5)")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix_xgb.png", dpi=120)
plt.close(fig)
print(f"已保存: confusion_matrix_xgb.png（{best} 混淆矩阵，阈值 0.5）")

# ---------- 备注：10% 测试集保留给阶段 6 最终评估，此处不碰 ----------
print("说明: 10% 测试集已切出但未使用，留待阶段 6 最终评估（防多次调参污染）")
