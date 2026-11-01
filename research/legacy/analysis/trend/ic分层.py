import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字体设置失败: {e}")

# --- 1. 加载预处理好的面板数据 ---
# 【修改】读取新的趋势因子面板数据文件
try:
    panel_data = pd.read_csv('panel_data_for_trend_factor_analysis.csv')
except FileNotFoundError:
    print("错误：未找到 'panel_data_for_trend_factor_analysis.csv' 文件。")
    print("请先运行修改后的 '数据处理.py' 脚本来生成这个文件。")
    exit()

panel_data['date'] = pd.to_datetime(panel_data['date'])

# --- 2. 定义要分析的因子列表 ---
# 【修改】将因子列更新为新的趋势因子
factor_columns = ['t_macross', 't_mamom', 't_maratio', 't_mom']

# 循环分析每一个因子
for factor_name in factor_columns:
    print(f"\n{'='*30}\n开始分析因子: {factor_name}\n{'='*30}")
    
    # --- 3. IC (信息系数) 分析 ---
    ic_series = panel_data.groupby('date').apply(
        lambda x: x[factor_name].corr(x['forward_return'])
    ).dropna()
    
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ir = ic_mean / ic_std
    positive_ratio = (ic_series > 0).sum() / len(ic_series)
    
    print("\n--- IC 分析结果 ---")
    print(f"因子: {factor_name}")
    print(f"IC均值: {ic_mean:.4f}, IC标准差: {ic_std:.4f}")
    print(f"信息比率 (IR): {ir:.4f}, IC>0比例: {positive_ratio:.2%}")
    
    # 绘制IC时间序列图
    plt.figure(figsize=(12, 6))
    ic_series.plot(kind='line', label='Daily IC')
    plt.axhline(ic_mean, color='red', linestyle='--', label=f'IC均值 = {ic_mean:.4f}')
    plt.axhline(0, color='gray', linestyle='-')
    plt.title(f'因子 "{factor_name}" 的IC时间序列图')
    plt.xlabel('日期'); plt.ylabel('IC值')
    plt.legend(); plt.grid(True)
    plt.savefig(f'{factor_name}_ic_series.png')
    plt.show()

    # --- 4. 分层回测分析 ---
    quantiles = panel_data.groupby('date')[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    panel_data['group'] = quantiles
    
    grouped_returns = panel_data.groupby(['date', 'group'])['forward_return'].mean().unstack()
    cumulative_returns = (1 + grouped_returns).cumprod()
    
    print("\n--- 分层回测结果 (年化收益率) ---")
    annualized_returns = (cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns))) - 1
    print(annualized_returns)
    
    # 绘制分层回测净值曲线
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot(legend=True)
    plt.title(f'因子 "{factor_name}" 的分层回测净值曲线')
    plt.xlabel('日期'); plt.ylabel('累计收益 (净值)')
    plt.grid(True)
    plt.savefig(f'{factor_name}_layered_backtest.png')
    plt.show()

print("\n所有因子分析完成！")