"""
feature_engineering.py - 特征工程（W2 交付物）
项目：信贷违约预测模型
输入：data/train_clean.csv（含 isDefault）、data/testA_clean.csv（无 isDefault）
输出：data/train_scaled.csv、data/testA_scaled.csv（Step 2 变换+标准化后）

完整流程（对应项目任务 3.1-3.4）：
  Step 1 类别编码（3.1）：序数编码（grade/subGrade/employmentLength）、
                          独热编码（homeOwnership/verificationStatus/purpose）、
                          目标编码（regionCode，仅用训练集违约率，防数据泄漏）
  Step 2 数值变换与标准化（3.2）：右偏列 log1p 变换；连续列 z-score 标准化
  Step 3 特征创建（3.3）：fico 合并、信用历史长度、月供收入比、交互项
  Step 4 特征选择（3.4）：过滤法（相关性）+ 嵌入法（随机森林特征重要性）
当前版本：Step 1 + Step 2 + Step 3 + Step 4
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 向上查找含 data/ 的目录作为项目根（脚本放 scripts/ 或 scripts/W2/ 均可运行）
BASE_DIR = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
     if (p / "data").is_dir()),
    Path(__file__).resolve().parents[2],
)
DATA = BASE_DIR / "data"

train = pd.read_csv(DATA / "train_clean.csv")
test = pd.read_csv(DATA / "testA_clean.csv")
print(f"加载: train {train.shape}, testA {test.shape}")

# ---------- 删除列（无信息 / 高基数无业务价值，已与业务确认） ----------
DROP_COLS = ["id", "policyCode", "employmentTitle", "postCode", "title"]
# 理由：id 纯标识；policyCode 仅 1 个取值（常量）；后三列取值数过多（数千~数万），
#       独热/目标编码都会稀疏且无业务解释价值，删除。
print(f"\n删除列（{len(DROP_COLS)} 列）: {DROP_COLS}")
train = train.drop(columns=DROP_COLS)
test = test.drop(columns=DROP_COLS)

# ---------- 1) 序数编码：grade（A=1 ... G=7） ----------
grade_order = sorted(train["grade"].unique())
grade_map = {g: i + 1 for i, g in enumerate(grade_order)}
train["grade_enc"] = train["grade"].map(grade_map)
test["grade_enc"] = test["grade"].map(grade_map)
print(f"\n[1] grade 序数编码: {grade_map}")

# ---------- 2) 序数编码：subGrade（A1 < A2 < ... < G5，共 35 档） ----------
subgrade_order = sorted(train["subGrade"].unique(),
                        key=lambda s: (s[0], int(s[1:])))
subgrade_map = {g: i + 1 for i, g in enumerate(subgrade_order)}
train["subGrade_enc"] = train["subGrade"].map(subgrade_map)
test["subGrade_enc"] = test["subGrade"].map(subgrade_map)
print(f"[2] subGrade 序数编码: 共 {len(subgrade_map)} 档，范围 {min(subgrade_map.values())}-{max(subgrade_map.values())}")

# ---------- 3) 序数编码：employmentLength（年限；<1 年=0；未知=-1） ----------
def parse_emp_len(x):
    x = str(x).strip()
    if x == "未知":
        return -1
    if x.startswith("<"):
        return 0
    if "+" in x:
        return int(x.split("+")[0])
    return int(x.split()[0])

train["employmentLength_enc"] = train["employmentLength"].map(parse_emp_len)
test["employmentLength_enc"] = test["employmentLength"].map(parse_emp_len)
print(f"[3] employmentLength 序数编码: 取值 {sorted(train['employmentLength_enc'].unique())}")

# ---------- 4) 独热编码：homeOwnership / verificationStatus / purpose ----------
# drop_first=True：删掉基准类，避免计量里的"虚拟变量陷阱"（多重共线性）
for col in ["homeOwnership", "verificationStatus", "purpose"]:
    train = pd.get_dummies(train, columns=[col], prefix=col, drop_first=True, dtype=int)
    test = pd.get_dummies(test, columns=[col], prefix=col, drop_first=True, dtype=int)
    # 保证 train/test 列一致（测试集若缺某类，补 0 列）
    # 注意排除 isDefault：测试集没有标签，不能生成假的 isDefault 列
    feat_cols = [c for c in train.columns if c != "isDefault"]
    for c in feat_cols:
        if c not in test.columns:
            test[c] = 0
    test = test[feat_cols]
print(f"[4] 独热编码完成（homeOwnership/verificationStatus/purpose，drop_first）")

# ---------- 5) 目标编码：regionCode（用训练集违约率，防数据泄漏） ----------
region_map = train.groupby("regionCode")["isDefault"].mean()
train["regionCode_enc"] = train["regionCode"].map(region_map)
test["regionCode_enc"] = test["regionCode"].map(region_map).fillna(train["isDefault"].mean())
print(f"[5] regionCode 目标编码: 训练集 {len(region_map)} 个地区，违约率范围 "
      f"{region_map.min():.3f}-{region_map.max():.3f}；测试集未知地区填总体违约率 "
      f"{train['isDefault'].mean():.3f}")

# ---------- 保留数值的类别列（term/initialListStatus/applicationType） ----------
# 本身就是 0/1 或有序数值（36/60），无需额外编码，保留原列即可。
print(f"\n保留原数值的类别列: term / initialListStatus / applicationType")

# ---------- 整理：去掉原始类别列，只留编码后列 + 数值列 ----------
RAW_CAT_DROP = ["grade", "subGrade", "employmentLength", "regionCode"]
train = train.drop(columns=RAW_CAT_DROP)
test = test.drop(columns=RAW_CAT_DROP)

train.to_csv(DATA / "train_encoded.csv", index=False)
test.to_csv(DATA / "testA_encoded.csv", index=False)
print(f"\nStep 1 完成：train_encoded.csv {train.shape}，testA_encoded.csv {test.shape}")
print(f"已保存: {DATA / 'train_encoded.csv'}、{DATA / 'testA_encoded.csv'}")

# ==================== Step 2 数值变换与标准化（任务 3.2） ====================

# ---------- 2-1) 右偏金额列 log1p 变换（EDA 已确认右偏；log1p 兼容 0 值） ----------
LOG_COLS = ["loanAmnt", "installment", "annualIncome", "revolBal"]
print(f"\n========== Step 2 数值变换与标准化（任务 3.2） ==========")
print(f"[2-1] log1p 变换右偏列: {LOG_COLS}")
for col in LOG_COLS:
    before = train[col]
    train[f"{col}_log"] = np.log1p(train[col])
    test[f"{col}_log"] = np.log1p(test[col])
    train = train.drop(columns=[col])
    test = test.drop(columns=[col])
    print(f"  {col}: 原范围 {before.min():,.0f}-{before.max():,.0f} -> "
          f"log 后 {train[f'{col}_log'].min():.2f}-{train[f'{col}_log'].max():.2f}")

# ---------- 2-2) z-score 标准化（仅连续数值列） ----------
# 不缩放：目标变量、0/1 二值列（initialListStatus/applicationType/独热列）、
#         时间字符串列（issueDate/earliesCreditLine 归 3.3 处理）
NOT_SCALE = ["isDefault", "initialListStatus", "applicationType",
             "issueDate", "earliesCreditLine"]
NOT_SCALE += [c for c in train.columns
              if c.startswith(("homeOwnership_", "verificationStatus_", "purpose_"))]
SCALE_COLS = [c for c in train.columns if c not in NOT_SCALE]

# 标准化只 fit 训练集，再 transform 训练/测试集（防数据泄漏，同目标编码逻辑）
scaler = StandardScaler()
train[SCALE_COLS] = scaler.fit_transform(train[SCALE_COLS])
test[SCALE_COLS] = scaler.transform(test[SCALE_COLS])
print(f"[2-2] z-score 标准化 {len(SCALE_COLS)} 列（只 fit 训练集，防泄漏）")
print(f"      示例列标准化后均值/标准差（应≈0/1）:")
for c in SCALE_COLS[:4]:
    print(f"      {c:<20} mean={train[c].mean():.4f}  std={train[c].std():.3f}")
print(f"      未缩放: 目标 isDefault、0/1 列、时间列（issueDate/earliesCreditLine）")

train.to_csv(DATA / "train_scaled.csv", index=False)
test.to_csv(DATA / "testA_scaled.csv", index=False)
print(f"\nStep 2 完成：train_scaled.csv {train.shape}，testA_scaled.csv {test.shape}")
print(f"已保存: {DATA / 'train_scaled.csv'}、{DATA / 'testA_scaled.csv'}")

# ==================== Step 3 特征创建（任务 3.3） ====================
# 业务逻辑：把领域知识变成新特征。新建特征都要求"能说出一句业务理由"。
print(f"\n========== Step 3 特征创建（任务 3.3） ==========")

# 3-1) fico 均值：ficoRangeLow/High 是同一评分的上下限，合并为一个值
train["fico_avg"] = (train["ficoRangeLow"] + train["ficoRangeHigh"]) / 2
test["fico_avg"] = (test["ficoRangeLow"] + test["ficoRangeHigh"]) / 2
train = train.drop(columns=["ficoRangeLow", "ficoRangeHigh"])
test = test.drop(columns=["ficoRangeLow", "ficoRangeHigh"])
print("[3-1] fico_avg: ficoRangeLow/High 合并为均值")

# 3-2) 信用历史长度 = 发放年份 - 最早开卡年份（信用越久越可信）
#      issueDate 格式 "2014-07-01"，earliesCreditLine 格式 "Aug-2001"
def issue_year(s):
    try:
        return int(str(s)[:4])
    except (ValueError, TypeError):
        return np.nan

def earliest_year(s):
    try:
        return int(str(s).split("-")[-1])
    except (ValueError, TypeError):
        return np.nan

train["credit_history_years"] = train["issueDate"].map(issue_year) - \
    train["earliesCreditLine"].map(earliest_year)
test["credit_history_years"] = test["issueDate"].map(issue_year) - \
    test["earliesCreditLine"].map(earliest_year)
# 解析失败用训练集中位数填充（只从训练集取，防泄漏）
med_years = train["credit_history_years"].median()
train["credit_history_years"] = train["credit_history_years"].fillna(med_years)
test["credit_history_years"] = test["credit_history_years"].fillna(med_years)
train = train.drop(columns=["issueDate", "earliesCreditLine"])
test = test.drop(columns=["issueDate", "earliesCreditLine"])
print(f"[3-2] credit_history_years: 中位数 {med_years:.0f} 年（训练集）")

# 3-3) 月供收入比（log 空间）：log(月供×12) - log(年收入)，衡量月供压力
train["installment_income_ratio"] = \
    train["installment_log"] + np.log(12) - train["annualIncome_log"]
test["installment_income_ratio"] = \
    test["installment_log"] + np.log(12) - test["annualIncome_log"]
print("[3-3] installment_income_ratio: 月供压力（对数空间）")

# 3-4) 交互项：高评级×高负债、高利率×大金额（计量里的交互项思想）
train["grade_dti_inter"] = train["grade_enc"] * train["dti"]
test["grade_dti_inter"] = test["grade_enc"] * test["dti"]
train["interest_loan_inter"] = train["interestRate"] * train["loanAmnt_log"]
test["interest_loan_inter"] = test["interestRate"] * test["loanAmnt_log"]
print("[3-4] 交互项: grade×dti（高风险+高负债）、interestRate×贷款规模")

# 新特征统一标准化（只 fit 训练集，防泄漏；与 Step 2 同规则）
NEW_COLS = ["fico_avg", "credit_history_years", "installment_income_ratio",
            "grade_dti_inter", "interest_loan_inter"]
scaler_new = StandardScaler()
train[NEW_COLS] = scaler_new.fit_transform(train[NEW_COLS])
test[NEW_COLS] = scaler_new.transform(test[NEW_COLS])
print(f"[3-5] 新特征标准化完成: {NEW_COLS}")
print(f"Step 3 完成：当前数据 {train.shape}（Step 4 特征选择后将输出最终建模数据）")

# ==================== Step 4 特征选择（任务 3.4） ====================
# 过滤法（相关性）+ 嵌入法（随机森林特征重要性）；包装法（RFE）代价高，暂不采用
OUT_W2 = BASE_DIR / "outputs" / "W2"
OUT_W2.mkdir(parents=True, exist_ok=True)

print(f"\n========== Step 4 特征选择（任务 3.4） ==========")
FEATURES = [c for c in train.columns if c != "isDefault"]
X = train[FEATURES]
y = train["isDefault"]

# 4-1) 过滤法：每个特征与目标的相关系数（单变量初步信号）
corr_series = X.corrwith(y).abs().sort_values(ascending=False)
print(f"[4-1] 过滤法：与 isDefault 相关系数 Top 5: "
      f"{[(c, round(corr_series[c], 3)) for c in corr_series.index[:5]]}")

# 4-2) 嵌入法：随机森林特征重要性（限深度加速；80 万行需等待）
print("[4-2] 训练随机森林计算特征重要性（80 万行，约 1-3 分钟）...")
rf_sel = RandomForestClassifier(n_estimators=100, max_depth=12,
                                random_state=42, n_jobs=-1)
rf_sel.fit(X, y)
imp = pd.Series(rf_sel.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"      特征重要性 Top 5: {[(c, round(imp[c], 4)) for c in imp.index[:5]]}")

# 4-3) 决策：保留重要性 >= 阈值 的特征（阈值 = 均值的 1/10，删尾部弱特征）
threshold = imp.mean() / 10
keep = imp[imp >= threshold].index.tolist()
drop = imp[imp < threshold].index.tolist()
print(f"[4-3] 选择阈值 = 均值的 1/10 = {threshold:.5f}")
print(f"      保留 {len(keep)} 列 / 删除 {len(drop)} 列")
print(f"      删除: {drop}")

# 4-4) 输出：特征重要性表 + 图 + 最终建模数据
imp.to_csv(OUT_W2 / "feature_importance.csv", header=["importance"])

fig, ax = plt.subplots(figsize=(10, 8))
imp.head(20).iloc[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Feature importance (Random Forest, top 20)")
ax.set_xlabel("importance")
fig.tight_layout()
fig.savefig(OUT_W2 / "feature_importance.png", dpi=120)
plt.close(fig)
print(f"已保存: {OUT_W2 / 'feature_importance.csv'}、{OUT_W2 / 'feature_importance.png'}")

train_final = train[keep + ["isDefault"]]
test_final = test[keep]
train_final.to_csv(DATA / "train_final.csv", index=False)
test_final.to_csv(DATA / "testA_final.csv", index=False)
print(f"\nStep 4 完成：train_final.csv {train_final.shape}，testA_final.csv {test_final.shape}")
print(f"已保存: {DATA / 'train_final.csv'}、{DATA / 'testA_final.csv'}")
print("\n特征工程全部完成（3.1 编码 -> 3.2 变换/标准化 -> 3.3 创建 -> 3.4 选择）")
