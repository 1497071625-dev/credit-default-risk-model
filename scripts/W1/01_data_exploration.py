"""
01 数据探索：加载 + 质量检查
项目：信贷违约预测模型（阿里数据分析实习 W1）
输出：控制台报告 + outputs/W1/data_quality_report.txt
"""
from pathlib import Path
import pandas as pd

# 向上查找含 data/ 的目录作为项目根（脚本放 scripts/ 或 scripts/W1/ 均可运行）
BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
TRAIN = BASE_DIR / "data" / "train.csv"
TEST = BASE_DIR / "data" / "testA.csv"
OUT = BASE_DIR / "outputs" / "W1"
OUT.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 60)

buf = []

def log(msg=""):
    print(msg)
    buf.append(str(msg))

log("=" * 60)
log("信贷违约预测项目 - 数据质量检查报告")
log("=" * 60)

# 1. 加载
log("\n[1] 数据加载")
train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)
log(f"train.csv: {len(train):,} 行 x {train.shape[1]} 列")
log(f"testA.csv: {len(test):,} 行 x {test.shape[1]} 列")
log(f"testA 比 train 少的列: {sorted(set(train.columns) - set(test.columns))}")

# 2. 列类型总览
log("\n[2] 列类型总览（train）")
log(train.dtypes.astype(str).value_counts().to_string())

# 3. 缺失值
log("\n[3] 缺失值（train）")
miss = train.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
log(f"有缺失的列: {len(miss)}/{train.shape[1]}，缺失格总数: {int(train.isna().sum().sum()):,}")
for k, v in miss.items():
    log(f"  {k}: {v:,} ({v / len(train) * 100:.2f}%)")

# 4. 唯一性
log("\n[4] 唯一性检查")
log(f"train: 行数 {len(train):,}，id 唯一数 {train['id'].nunique():,}，重复 id {train['id'].duplicated().sum():,}")
log(f"test : 行数 {len(test):,}，id 唯一数 {test['id'].nunique():,}，重复 id {test['id'].duplicated().sum():,}")

# 5. 目标变量
log("\n[5] 目标变量 isDefault 分布")
vc = train["isDefault"].value_counts().sort_index()
for k, v in vc.items():
    log(f"  isDefault={k}: {v:,} ({v / len(train) * 100:.2f}%)")

# 6. 关键数值列分布（找异常值线索）
log("\n[6] 关键数值列 describe（min/max 看异常）")
KEY_NUM = ["loanAmnt", "term", "interestRate", "installment", "annualIncome",
           "dti", "ficoRangeLow", "openAcc", "pubRec", "pubRecBankruptcies",
           "revolBal", "revolUtil", "totalAcc", "delinquency_2years"]
log(train[KEY_NUM].describe().T.to_string())

# 7. 类别列
log("\n[7] 类别列取值")
for c in train.select_dtypes(include="str").columns:
    n = train[c].nunique()
    log(f"  {c}: 唯一值 {n} 个")
    if n <= 15:
        log("     " + str(train[c].value_counts(dropna=False).head(10).to_dict()))

report = OUT / "data_quality_report.txt"
with open(report, "w", encoding="utf-8") as f:
    f.write("\n".join(buf))
log(f"\n报告已保存: {report}")
