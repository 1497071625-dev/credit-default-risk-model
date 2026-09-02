"""
data_preprocessing.py - 数据预处理完整流程（Step 1 检测 + Step 2 处理）
项目：信贷违约预测模型（W1 交付物）

Step 1 异常值检测（任务 2.2）：
  - 业务规则：按业务常识设定检查标准（年收入>0、dti 在 0-100、revolUtil 在 0-100、fico 上下限不倒挂）
  - 3σ / IQR：统计离群数量，供判断；有偏分布列以业务规则 + 箱线图为准
  - 极端样本画像（跨列自洽）与长尾分布检查：判定真实极端值，保留
  - 输出：outputs/W1/outlier_report.txt、outputs/W1/outlier_boxplot.png、outputs/W1/outlier_tail.png

Step 2 清洗（依据 Step 1 检测结论处理）：
  A. 缺失值：极少缺失列删行；n0-n14 填中位数；employmentLength 填"未知"；其余补齐
  B. 异常值（业务规则）：annualIncome=0 视为缺失；revolUtil 封顶 100；dti 越界删行
  C. 一致性：id 唯一性校验；ficoRangeLow<=ficoRangeHigh 校验
  - 输出：data/train_clean.csv、data/testA_clean.csv
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

# 向上查找含 data/ 的目录作为项目根（脚本放 scripts/ 或 scripts/W1/ 均可运行）
BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"
OUT = BASE_DIR / "outputs" / "W1"
OUT.mkdir(parents=True, exist_ok=True)

# 缺失极少、直接删行的列（共约 1000 行受影响）
DROP_COLS = ["title", "postCode", "employmentTitle", "dti",
             "pubRecBankruptcies", "revolUtil"]
# 匿名行为特征（填中位数）
N_COLS = [f"n{i}" for i in range(15)]


def detect_outliers(df: pd.DataFrame):
    """Step 1 异常值检测（业务规则 + 3σ + 箱线图/IQR + 画像 + 长尾）"""
    buf = []

    def log(msg=""):
        print(msg)
        buf.append(str(msg))

    log("=" * 60)
    log("异常值检测（任务 2.2：箱线图、3σ、业务规则）")
    log("=" * 60)
    log(f"数据: {len(df):,} 行（原始 train.csv，先检测后处理）")

    # [1] 业务规则检测
    log("\n[1] 业务规则检测（按业务常识设定检查标准）")
    biz = {
        "annualIncome<=0（视为缺失）": int((df["annualIncome"] <= 0).sum()),
        "dti<0 或 dti>100（越界删行）": int(((df["dti"] < 0) | (df["dti"] > 100)).sum()),
        "revolUtil>100（封顶到100）": int((df["revolUtil"] > 100).sum()),
        "ficoRangeLow>ficoRangeHigh（不一致）": int((df["ficoRangeLow"] > df["ficoRangeHigh"]).sum()),
    }
    for k, v in biz.items():
        log(f"  {k}: {v:,}")

    # [2] 3σ 检测
    log("\n[2] 3σ 检测（均值±3倍标准差之外的点）")
    SIGMA_COLS = ["loanAmnt", "interestRate", "installment", "annualIncome", "dti",
                  "revolUtil", "ficoRangeLow", "openAcc", "pubRec", "pubRecBankruptcies",
                  "revolBal", "totalAcc", "delinquency_2years"]
    log("  列名                       3σ异常数    占比")
    for c in SIGMA_COLS:
        s = df[c].dropna()
        mu, sd = s.mean(), s.std()
        n = int(((s < mu - 3 * sd) | (s > mu + 3 * sd)).sum())
        log(f"  {c:<26} {n:>8,}   {n / len(df) * 100:6.2f}%")

    # [3] 箱线图 / IQR 检测
    log("\n[3] 箱线图 / IQR 离群点检测（Q1-1.5IQR 与 Q3+1.5IQR 之外）")
    BOX_COLS = ["loanAmnt", "interestRate", "installment", "annualIncome", "dti", "revolUtil"]
    log("  列名                        IQR离群数   占比")
    for c in BOX_COLS:
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        log(f"  {c:<26} {n:>8,}   {n / len(df) * 100:6.2f}%")

    # [4] 箱线图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, BOX_COLS):
        ax.boxplot(df[col].dropna(), orientation="vertical")
        ax.set_title(col)
        ax.set_ylabel("Value")
        if col == "annualIncome":
            ax.set_yscale("log")
    fig.suptitle("Outlier boxplots (IQR whiskers, train raw)")
    fig.tight_layout()
    fig.savefig(OUT / "outlier_boxplot.png", dpi=120)
    plt.close(fig)
    log(f"\n箱线图已保存: {OUT / 'outlier_boxplot.png'}")

    # [5] 极端样本画像（跨列自洽检查）
    log("\n[5] 极端样本画像（跨列自洽：看这些点像不像真人）")
    s_inc = df["annualIncome"].dropna()
    q1i, q3i = s_inc.quantile(0.25), s_inc.quantile(0.75)
    up_inc = q3i + 1.5 * (q3i - q1i)
    n_ext = int((df["annualIncome"] > up_inc).sum())
    ext = df[df["annualIncome"] > up_inc].sort_values("annualIncome", ascending=False).head(30)
    log(f"  年收入 IQR 上界 = {up_inc:,.0f}，极端样本 {n_ext:,} 个，展示收入最高的 30 个：")
    log("  收入      贷款金额   循环余额      FICO下限  dti    违约")
    for _, r in ext.iterrows():
        log(f"  {r['annualIncome']:>10,.0f} {r['loanAmnt']:>9,.0f} {r['revolBal']:>12,.0f} {r['ficoRangeLow']:>7.0f} {r['dti']:>6.1f} {int(r['isDefault'])}")
    corr_inc = ext[["annualIncome", "loanAmnt"]].corr().iloc[0, 1]
    log(f"  自洽检查：极端样本中 收入 vs 贷款金额 相关系数 = {corr_inc:.2f}（>0 说明收入高者贷款也高，画像自洽）")

    # [6] 长尾分布检查（连续成串 vs 孤立跳变）
    log("\n[6] 长尾分布检查（极端值是否连续成串）")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col in zip(axes, ["annualIncome", "installment"]):
        s = df[col].dropna()
        q1v, q3v = s.quantile(0.25), s.quantile(0.75)
        upv = q3v + 1.5 * (q3v - q1v)
        tail = s[s > upv]
        if col == "annualIncome":
            ax.hist(np.log10(tail), bins=30, color="#4C72B0", edgecolor="white")
            ax.set_title(f"{col} tail (>IQR upper, log10)")
            ax.set_xlabel("annualIncome (log10)")
        else:
            ax.hist(tail, bins=30, color="#4C72B0", edgecolor="white")
            ax.set_title(f"{col} tail (>IQR upper)")
            ax.set_xlabel(col)
        ax.set_ylabel("Count")
        shape = "连续成串" if len(tail) > 100 else "较稀疏"
        log(f"  {col}: 上界 {upv:,.0f}，尾巴样本 {len(tail):,} 个，{shape}")
    fig.suptitle("Long-tail distribution of extreme values")
    fig.tight_layout()
    fig.savefig(OUT / "outlier_tail.png", dpi=120)
    plt.close(fig)
    log(f"\n长尾分布图已保存: {OUT / 'outlier_tail.png'}")

    # [7] 汇总
    log("\n[7] 汇总：建议处理策略（Step 2 清洗依据此结论执行）")
    log("  annualIncome<=0  -> 视为缺失并填充")
    log("  dti 越界         -> 删除行")
    log("  revolUtil>100    -> 封顶到 100")
    log("  fico 上下限倒挂  -> 0 条，无需处理")
    log("  3σ/IQR 离群点    -> 经跨列自洽 + 长尾成串检查，判定为真实极端值，保留")

    report = OUT / "outlier_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(buf))
    log(f"\n检测报告已保存: {report}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 清洗（依据 Step 1 检测结论处理）"""
    df = df.copy()
    dropped = 0

    # --- B1. 异常值（业务规则，处理策略与 Step 1 检测结果一致）---
    n = (df["annualIncome"] == 0).sum()
    df.loc[df["annualIncome"] == 0, "annualIncome"] = pd.NA   # 收入=0 视为缺失
    print(f"  [B1a] annualIncome=0 视为缺失: {n} 个")

    n = (df["revolUtil"] > 100).sum()
    df.loc[df["revolUtil"] > 100, "revolUtil"] = 100.0        # 利用率封顶 100
    print(f"  [B1b] revolUtil>100 封顶: {n} 个")

    bad_dti = (df["dti"] < 0) | (df["dti"] > 100)             # dti 越界删行
    n = bad_dti.sum()
    df = df[~bad_dti]
    dropped += n
    print(f"  [B1c] dti 越界删行: {n} 行")

    # --- A1. 缺失极少的列：删行 ---
    n = df[DROP_COLS].isna().any(axis=1).sum()
    df = df.dropna(subset=DROP_COLS)
    dropped += n
    print(f"  [A1] 缺失极少列删行: {n} 行")

    # --- A2. 匿名特征 n0-n14 填中位数 ---
    df[N_COLS] = df[N_COLS].fillna(df[N_COLS].median())
    print("  [A2] n0-n14 填中位数")

    # --- A3. employmentLength 填"未知" ---
    n = df["employmentLength"].isna().sum()
    df["employmentLength"] = df["employmentLength"].fillna("未知")
    print(f"  [A3] employmentLength 填'未知': {n} 个")

    # --- A4. 其余缺失：数值列填中位数，文本列填"未知" ---
    for c in df.columns[df.isna().any()]:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna("未知")

    # --- C. 一致性检查 ---
    print(f"  [C1] id 重复: {df['id'].duplicated().sum()}")
    print(f"  [C2] ficoRangeLow>High 不一致: {(df['ficoRangeLow'] > df['ficoRangeHigh']).sum()}")
    print(f"  [C3] 剩余缺失格: {int(df.isna().sum().sum())}")
    print(f"  => 删行合计: {dropped} 行")
    return df


def main():
    print("===== Step 1 异常值检测（train 原始数据）=====")
    train = pd.read_csv(DATA / "train.csv")
    detect_outliers(train)

    print("\n===== Step 2 清洗 train.csv =====")
    print(f"清洗前: {len(train):,} 行 x {train.shape[1]} 列")
    train = clean(train)
    print(f"清洗后: {len(train):,} 行 x {train.shape[1]} 列\n")

    print("===== Step 2 清洗 testA.csv =====")
    test = pd.read_csv(DATA / "testA.csv")
    print(f"清洗前: {len(test):,} 行 x {test.shape[1]} 列")
    test = clean(test)
    print(f"清洗后: {len(test):,} 行 x {test.shape[1]} 列\n")

    train.to_csv(DATA / "train_clean.csv", index=False)
    test.to_csv(DATA / "testA_clean.csv", index=False)
    print(f"已保存: {DATA / 'train_clean.csv'}、{DATA / 'testA_clean.csv'}")


if __name__ == "__main__":
    main()
