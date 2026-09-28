# 本脚本的初稿与重构借助 AI 编码工具（OpenAI Codex 桌面版 26.915.31945 / codex-cli 0.155.0-alpha.9.2）生成；处理逻辑、参数口径与验收标准由本人确定，输出经本人逐项核对。
"""
pricing_headroom.py - 差异化定价：低风险客户的让价空间（对齐 1.2 节第二个业务目标）
项目：信贷违约预测模型

为什么做这个：
  1.2 节把"给低风险客户更优的利率与额度（差异化定价）"写成要解决的问题，但 10.1-10.3 节
  只回答了"谁不放款"。审批线的另一头是定价：风险低的人，能便宜多少还不出事？

  口径与 10.2 节完全一致（净收益率/净息差 m = 6%、回收率 rec = 30%）：
    每 1 元贷款的期望利润 EP = (1 - p) × m × 期限 - p × (1 - rec)
  把 EP 置零反解，得到每个风险档的**保本净息差**：
    m*(p) = p × (1 - rec) / [(1 - p) × 期限]
  m* 低于机构假设的 6%，差额就是这一档"还能让出多少价"（让价空间）。

输入：outputs/W4/test_predictions_calibrated.csv（79,850 人：原始/校准概率 + 金额/期限/利率）
产出：outputs/W4/pricing_headroom.csv   六档的让价空间表

口径说明（演示性测算）：净息差 6%、回收率 30% 均为假设值，数据集里没有这两列；
  "让价空间"是相对 6% 假设的余量，不是建议定价。数据集无授信额度字段，额度只给方向。
"""
import pandas as pd

MARGIN = 0.06      # 净息差假设（与 10.2 节中性口径一致）
RECOVERY = 0.30    # 回收率假设
SRC = 'outputs/W4/test_predictions_calibrated.csv'
OUT = 'outputs/W4/pricing_headroom.csv'

BINS = [-1, .02, .05, .10, .20, .40, 1]
LABELS = ['<2%', '2-5%', '5-10%', '10-20%', '20-40%', '>40%']


def main():
    df = pd.read_csv(SRC)
    df['band'] = pd.cut(df.prob_cal, bins=BINS, labels=LABELS)
    df['ep'] = ((1 - df.prob_cal) * MARGIN * df.term - df.prob_cal * (1 - RECOVERY)) * df.loanAmnt

    g = df.groupby('band', observed=True).agg(
        人数=('y_true', 'size'),
        实际违约率=('y_true', 'mean'),
        模型概率=('prob_cal', 'mean'),
        平均期限=('term', 'mean'),
        平均利率=('interestRate', 'mean'),
        人均放款=('loanAmnt', 'mean'),
        放款额=('loanAmnt', 'sum'),
        期望利润=('ep', 'sum'),
    )
    g['占比'] = g.人数 / len(df)
    g['保本净息差'] = g.模型概率 * (1 - RECOVERY) / ((1 - g.模型概率) * g.平均期限)
    g['让价空间'] = (MARGIN - g['保本净息差']).clip(lower=0)
    g['每万元期望利润'] = g.期望利润 / g.放款额 * 10000

    g.to_csv(OUT, encoding='utf-8-sig', float_format='%.6f')

    pd.set_option('display.width', 250)
    show = g[['人数', '占比', '实际违约率', '平均利率', '保本净息差', '让价空间', '每万元期望利润']]
    print(show.round(4).to_string())

    low = df[df.prob_cal < 0.05]
    life = (low.loanAmnt * low.term).sum()
    print(f'\n校准概率<5%：{len(low)} 人（占 {len(low) / len(df) * 100:.1f}%），'
          f'人均放款 {low.loanAmnt.mean():.0f} 元，放款额 {low.loanAmnt.sum() / 1e8:.2f} 亿元')
    print(f'利率下浮 1 个百分点：全生命周期让价 {life * 0.01 / 1e4:.0f} 万元；'
          f'下浮 2 个百分点：{life * 0.02 / 1e4:.0f} 万元')
    print(f'这批人自身期望利润：{low.ep.sum() / 1e4:.0f} 万元')
    print('\n已写出：' + OUT)


if __name__ == '__main__':
    main()
