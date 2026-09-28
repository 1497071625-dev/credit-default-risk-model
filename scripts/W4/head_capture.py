# -*- coding: utf-8 -*-
# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
head_capture.py —— 区分度指标：KS 与头部捕获率（W4 补充分析）

为什么要有这个脚本
    AUC / 精确率 / 召回率都在整批人上算平均，业务看不到"最坏的那一小撮有没有被抓住"。
    实际审批里，分数最高的那 5%、10% 是重点盯防对象：抓得住，拒贷名单就短而准。
    本脚本补两个排序指标：
      KS（Kolmogorov-Smirnov）：违约者与正常人的分数分布拉开的最大距离。
     头部捕获率：按模型分数从高到低取前 N%，这批人里抓到多少比例的违约者；
                 lift = 捕获率 ÷ N%（1.0 表示与随机挑没区别）。

输入
    outputs/W4/test_predictions.csv（y_true、prob_default；只看排序，与概率刻度无关）

输出
    outputs/W4/head_capture.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "outputs").is_dir()),
    Path(__file__).resolve().parents[2],
)
W4 = BASE_DIR / "outputs" / "W4"

df = pd.read_csv(W4 / "test_predictions.csv")
y = df["y_true"].to_numpy(int)
p = df["prob_default"].to_numpy(float)
n, n_pos = len(y), int(y.sum())

# KS：把所有人按分数升序排，两条累计分布曲线的最大间距
order = np.argsort(p)
ys = y[order]
pos_cum = np.cumsum(ys) / ys.sum()
neg_cum = np.cumsum(1 - ys) / (1 - ys).sum()
ks = float(np.max(np.abs(pos_cum - neg_cum)))

# 每个头部位置的 KS（KS 定义在整条曲线上，这里给出最大值出现的位置，便于解释）
ks_at = float(p[order][int(np.argmax(np.abs(pos_cum - neg_cum)))])

rows = []
for q in [0.05, 0.10, 0.20, 0.30]:
    k = int(round(n * q))
    idx = np.argsort(-p)[:k]
    cap = float(y[idx].sum() / n_pos)
    rows.append({
        "头部占比": q,
        "头部人数": k,
        "命中违约人数": int(y[idx].sum()),
        "捕获率": cap,
        "lift": cap / q,
        "头部内违约率": float(y[idx].mean()),
    })
res = pd.DataFrame(rows)
res.to_csv(W4 / "head_capture.csv", index=False)

summary = pd.DataFrame([{"指标": "KS", "值": ks, "说明": f"最大间距出现在分数 {ks_at:.3f} 附近"}])
summary.to_csv(W4 / "ks_summary.csv", index=False)

print(f"KS = {ks:.4f}（最大间距在分数 {ks_at:.3f} 附近）")
print(res.to_string(index=False))
print("已保存 head_capture.csv、ks_summary.csv")
