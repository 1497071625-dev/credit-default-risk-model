"""
dashboard_chart_data.py - 为仪表盘内联 SVG 图准备数据
项目：信贷违约预测模型（W4 交付物）
输入：data/train_clean.csv、outputs/W4/*.csv、outputs/W2/feature_importance.csv、outputs/W3/*.csv
输出：outputs/W4/dashboard_charts.json（供 scripts/W4/svg_charts.py 渲染）
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (p / "data").is_dir())
OUT = BASE / "outputs"
W1, W2, W3, W4 = OUT / "W1", OUT / "W2", OUT / "W3", OUT / "W4"

data: dict = {}

# ---------- ① W1：四张图 ----------
df = pd.read_csv(BASE / "data" / "train_clean.csv")
n = len(df)
d = int(df["isDefault"].sum())
data["target"] = {"total": n, "normal": int(n - d), "default": d}

def hist(series: pd.Series, bins: int = 50, transform=None):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if transform:
        s = transform(s)
    cnt, edges = np.histogram(s.to_numpy(), bins=bins)
    return [[float(edges[i]), float(edges[i + 1]), int(cnt[i])] for i in range(len(cnt))]

KEY_NUM = ["loanAmnt", "interestRate", "installment", "annualIncome", "dti", "revolUtil"]
data["dist"] = {c: hist(df[c]) for c in KEY_NUM}
data["income"] = {
    "raw": hist(df["annualIncome"]),
    "log": hist(df["annualIncome"], transform=np.log10),
}

def box(col: str):
    out = {}
    for flag, key in ((0, "normal"), (1, "default")):
        s = pd.to_numeric(df.loc[df["isDefault"] == flag, col], errors="coerce").dropna()
        q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
        out[key] = {"n": int(s.size), "min": float(s.min()), "q1": float(q1), "med": float(med),
                    "q3": float(q3), "max": float(s.max()), "mean": float(s.mean())}
    return out

data["cmp"] = {c: box(c) for c in KEY_NUM}

g = df.groupby("grade")["isDefault"].agg(["mean", "size"])
data["grade"] = [[str(i), float(r["mean"] * 100), int(r["size"])] for i, r in g.iterrows()]

corr_cols = KEY_NUM + ["ficoRangeLow", "isDefault"]
cm = df[corr_cols].corr()
data["corr"] = {"labels": list(corr_cols),
                "matrix": [[round(float(v), 4) for v in row] for row in cm.to_numpy()]}

# ---------- ② W4/W2/W3：八张图 ----------
pred = pd.read_csv(W4 / "test_predictions.csv")
y, p = pred["y_true"].to_numpy(), pred["prob_default"].to_numpy()

thr = np.linspace(0.001, 0.999, 400)
order = np.argsort(-p)
ys = y[order]
tp = np.cumsum(ys)
fp = np.cumsum(1 - ys)
tot_p, tot_n = int(ys.sum()), int(len(ys) - ys.sum())
for t in (0.5,):
    k = int((p >= t).sum())
    data["cm"] = {"tp": int(tp[k - 1]), "fp": int(fp[k - 1]),
                  "fn": int(tot_p - tp[k - 1]), "tn": int(tot_n - fp[k - 1])}

tpr = tp / tot_p
fpr = fp / tot_n
prec = tp / (tp + fp)
data["roc"] = {"auc": float(np.trapezoid(tpr, fpr)),
               "pts": [[round(float(fpr[i]), 4), round(float(tpr[i]), 4)]
                       for i in range(0, len(fpr), max(1, len(fpr) // 160))]}
data["pr"] = {"ap": float(np.sum((tpr - np.concatenate([[0], tpr[:-1]])) * prec)),
              "base": float(tot_p / len(y)),
              "pts": [[round(float(tpr[i]), 4), round(float(prec[i]), 4)]
                      for i in range(0, len(fpr), max(1, len(fpr) // 160))]}

lc = pd.read_csv(W4 / "learning_curve.csv")
data["learn"] = [[int(r["train_size"]), float(r["train_auc"]), float(r["val_auc"])]
                 for _, r in lc.iterrows()]

imp = pd.read_csv(W2 / "feature_importance.csv", index_col=0)["importance"].head(15)
data["imp"] = [[str(i), float(v)] for i, v in imp.items()]
shap = pd.read_csv(W3 / "shap_feature_importance.csv", index_col=0)["mean_abs_shap"].head(15)
data["shap"] = [[str(i), float(v)] for i, v in shap.items()]

tc = pd.read_csv(W3 / "tuning_full_combo.csv")
series = []
for (model, method), grp in tc.groupby(["model", "method"], sort=False):
    pts = [[int(r["trial"]), round(float(r["best_so_far"]), 5)] for _, r in grp.iterrows()]
    series.append({"model": str(model), "method": str(method), "pts": pts})
data["tuning"] = series

fc = pd.read_csv(W4 / "feature_count_auc.csv")
data["fc"] = [[int(r["n_features"]), float(r["train_auc"]), float(r["val_auc"])] for _, r in fc.iterrows()]

sens = pd.read_csv(W4 / "expected_loss_sensitivity.csv")
data["sens"] = {
    "probes": [{"name": str(r["收益口径"]), "rec": float(r["回收率"]), "reject": float(r["拒绝率"]),
                "catch": float(r["拦截率"]), "prec": float(r["被拒者精确率"]),
                "p3": float(r["隐含阈值 3 年"]), "p5": float(r["隐含阈值 5 年"])}
               for _, r in sens.iterrows()]
}

(W4 / "dashboard_charts.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print("已写出", W4 / "dashboard_charts.json", "| keys:", ", ".join(data))
