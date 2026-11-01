import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体，以便图表能够正确显示中文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print(f"中文字体设置失败，图表可能无法正确显示中文: {e}")
    print("请确保您的系统中已安装'SimHei'字体。")


# --- 1. 加载预处理好的面板数据 ---
# 【修改】读取新的价值因子面板数据文件
try:
    panel_data = pd.read_csv('panel_data_for_value_factor_analysis.csv')
except FileNotFoundError:
    print("错误：未找到 'panel_data_for_value_factor_analysis.csv' 文件。")
    print("请先运行修改后的 '数据处理.py' 脚本来生成这个文件。")
    exit()

panel_data['date'] = pd.to_datetime(panel_data['date'])

# --- 2. 定义要分析的因子列表 ---
# 【修改】将因子列更新为新的价值因子
factor_columns = ['Val_halfyear', 'Val_oneyear']

# 循环分析每一个因子
for factor_name in factor_columns:
    print(f"\n{'='*30}")
    print(f"开始分析因子: {factor_name}")
    print(f"{'='*30}")
    
    # --- 3. IC (信息系数) 分析 ---
    # IC是因子值与下一期收益率的截面相关系数
    ic_series = panel_data.groupby('date').apply(
        lambda x: x[factor_name].corr(x['forward_return'])
    )
    ic_series = ic_series.dropna()
    
    # 计算IC统计量
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ir = ic_mean / ic_std  # 信息比率 (Information Ratio)
    positive_ratio = (ic_series > 0).sum() / len(ic_series) # IC > 0 的比例
    
    print("\n--- IC 分析结果 ---")
    print(f"因子: {factor_name}")
    print(f"IC均值 (Mean IC): {ic_mean:.4f}")
    print(f"IC标准差 (Std Dev of IC): {ic_std:.4f}")
    print(f"信息比率 (IR): {ir:.4f}")
    print(f"IC值大于0的比例: {positive_ratio:.2%}")
    
    # 绘制IC时间序列图
    plt.figure(figsize=(12, 6))
    ic_series.plot(kind='line', label='Daily IC')
    plt.axhline(ic_mean, color='red', linestyle='--', label=f'IC均值 = {ic_mean:.4f}')
    plt.axhline(0, color='gray', linestyle='-')
    # 【修改】更新图表标题和文件名
    plt.title(f'因子 "{factor_name}" 的IC时间序列图')
    plt.xlabel('日期')
    plt.ylabel('IC值')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{factor_name}_ic_series.png')
    plt.show()

    # --- 4. 分层回测分析 ---
    # 按因子值将每个截面的品种分为5组
    quantiles = panel_data.groupby('date')[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    panel_data['group'] = quantiles
    
    # 计算每个分组的日均收益率
    grouped_returns = panel_data.groupby(['date', 'group'])['forward_return'].mean().unstack()
    
    # 计算每个分组的累计收益率 (净值曲线)
    cumulative_returns = (1 + grouped_returns).cumprod()
    
    print("\n--- 分层回测结果 ---")
    print(f"因子: {factor_name}")
    print("每个分组的年化收益率:")
    # 计算并打印年化收益率
    annualized_returns = (cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns))) - 1
    print(annualized_returns)
    
    # 绘制分层回测净值曲线
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot(legend=True)
    # 【修改】更新图表标题和文件名
    plt.title(f'因子 "{factor_name}" 的分层回测净值曲线')
    plt.xlabel('日期')
    plt.ylabel('累计收益 (净值)')
    plt.grid(True)
    plt.savefig(f'{factor_name}_layered_backtest.png')
    plt.show()

print("\n所有因子分析完成！")