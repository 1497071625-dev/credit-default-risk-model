# -*- coding: utf-8 -*-
# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
oot_validation.py —— 时间外验证（W4 补充分析）

为什么要有这个脚本
    主报告用随机 7:2:1 切分：同一批放款随机分到训练和测试，模型"见过"同时段的行情，
    成绩会偏乐观。真实上线是拿过去训练、给未来放贷打分，中间隔着行情变化。
    本脚本按放款时间（issueDate）切一次：早期放款训练、后期放款测试，
    再与同规模的随机切分对照，回答"换到未来，成绩掉多少、概率偏多少"。

切分口径
    训练段：issueDate ≤ 2016-12-31（664,968 笔，违约率 19.7%）
    时间外测试段：2017-01-01 ~ 2018-06-30（124,550 笔，违约率 22.3%）
    2018-07 之后的 8,980 笔不参与：离数据截止太近，违约还没到暴露期，
    违约率只有 7.5%，是"真变好"还是"没暴露"分不清，不干净。

模型与 W4 最终模型完全同参（LightGBM 优化参数），只换切分方式。

输出
    outputs/W4/oot_validation.csv     随机切分 vs 时间外切分的成绩对照
    outputs/W4/oot_calibration.csv    时间外测试集上预测概率与实际违约率的分箱对照
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from lightgbm import LGBMClassifier

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W4"
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = {"subsample": 0.8, "num_leaves": 63, "n_estimators": 200,
          "max_depth": 8, "learning_rate": 0.05, "colsample_bytree": 0.7}


def ks_stat(y_true: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ys = y_true[order]
    pos = np.cumsum(ys) / ys.sum()
    neg = np.cumsum(1 - ys) / (1 - ys).sum()
    return float(np.max(np.abs(pos - neg)))


def head_capture(y_true: np.ndarray, p: np.ndarray, q: float) -> float:
    k = int(round(len(p) * q))
    idx = np.argsort(-p)[:k]
    return float(y_true[idx].sum() / y_true.sum())


def evaluate(y_true, p, tag, n_train):
    reject = p >= 0.5
    tp = int((reject & (y_true == 1)).sum())
    fp = int((reject & (y_true == 0)).sum())
    fn = int((~reject & (y_true == 1)).sum())
    return {
        "口径": tag,
        "训练样本数": int(n_train),
        "样本数": int(len(y_true)),
        "违约率": float(y_true.mean()),
        "AUC": float(roc_auc_score(y_true, p)),
        "KS": ks_stat(y_true, p),
        "前5%捕获率": head_capture(y_true, p, 0.05),
        "前10%捕获率": head_capture(y_true, p, 0.10),
        "前20%捕获率": head_capture(y_true, p, 0.20),
        "预测概率均值": float(p.mean()),
        "Brier": float(brier_score_loss(y_true, p)),
        "阈值0.5拒绝率": float(reject.mean()),
        "阈值0.5精确率": float(tp / (tp + fp)) if tp + fp else 0.0,
        "阈值0.5召回率": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


print("加载 train_final.csv 与 issueDate ...")
df = pd.read_csv(DATA / "train_final.csv")
tcol = pd.read_csv(DATA / "train_clean.csv", usecols=["issueDate"])
assert len(tcol) == len(df), "两个文件行数不一致"
df["issueDate"] = pd.to_datetime(tcol["issueDate"]).values

FEATURES = [c for c in df.columns if c not in ("isDefault", "issueDate")]
X = df[FEATURES].to_numpy(dtype=np.float32)
y = df["isDefault"].to_numpy(int)
d = df["issueDate"]
print(f"  {len(df):,} 行 × {len(FEATURES)} 特征")

tr_time = (d <= "2016-12-31").to_numpy()
te_time = ((d >= "2017-01-01") & (d <= "2018-06-30")).to_numpy()
n_tr, n_te = int(tr_time.sum()), int(te_time.sum())
print(f"时间切分: 训练 {n_tr:,} / 时间外测试 {n_te:,}")

# 随机对照：训练/测试与时间切分**同样大小**，分层抽样（剩下的 8,980 笔丢弃）
idx_all = np.arange(len(y))
tr_rand, te_rand = train_test_split(idx_all, train_size=n_tr, test_size=n_te,
                                    stratify=y, random_state=42)
tr_rand_mask = np.zeros(len(y), bool); tr_rand_mask[tr_rand] = True
te_rand_mask = np.zeros(len(y), bool); te_rand_mask[te_rand] = True

rows = []
for tag, tr_mask, te_mask in [("随机切分（对照）", tr_rand_mask, te_rand_mask),
                              ("时间外（2017-01 起）", tr_time, te_time)]:
    spw = (y[tr_mask] == 0).sum() / (y[tr_mask] == 1).sum()
    model = LGBMClassifier(**PARAMS, scale_pos_weight=spw, verbosity=-1,
                           random_state=42, n_jobs=-1)
    model.fit(X[tr_mask], y[tr_mask])
    p = model.predict_proba(X[te_mask])[:, 1]
    row = evaluate(y[te_mask], p, tag, int(tr_mask.sum()))
    rows.append(row)
    print(f"[{tag}] 训练 {int(tr_mask.sum()):,} | AUC {row['AUC']:.4f} "
          f"KS {row['KS']:.4f} | 前10%捕获 {row['前10%捕获率']:.4f} | "
          f"预测均值 {row['预测概率均值']:.4f} vs 实际 {row['违约率']:.4f}")
    if tag.startswith("时间外"):
        bins = pd.qcut(p, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"bin": bins, "y": y[te_mask], "p": p}) \
            .groupby("bin").agg(平均预测概率=("p", "mean"), 实际违约率=("y", "mean"),
                                人数=("y", "size")).reset_index(drop=True)
        cal["平均预测概率"] = cal["平均预测概率"].round(4)
        cal["实际违约率"] = cal["实际违约率"].round(4)
        cal.to_csv(OUT / "oot_calibration.csv", index=False)
        print("  已保存 oot_calibration.csv")

res = pd.DataFrame(rows)
res.to_csv(OUT / "oot_validation.csv", index=False)
print("已保存 oot_validation.csv")
print(res.T.to_string())
