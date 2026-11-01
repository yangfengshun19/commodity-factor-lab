import pandas as pd

# --- 1. 加载预处理好的面板数据 ---
# 【修改】读取新的趋势因子面板数据文件
try:
    panel_data = pd.read_csv('panel_data_for_trend_factor_analysis.csv')
except FileNotFoundError:
    print("错误：未找到 'panel_data_for_trend_factor_analysis.csv' 文件。")
    print("请先运行修改后的 '数据处理.py' 脚本来生成这个文件。")
    exit()

panel_data['date'] = pd.to_datetime(panel_data['date'])
panel_data.set_index('date', inplace=True)


# --- 2. 换手率计算函数 (代码不变) ---
def calculate_turnover(panel_data, factor_name, group_to_track='top'):
    panel_data['group'] = panel_data.groupby(panel_data.index)[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    
    target_group_label = 4 if group_to_track == 'top' else 0
    target_group_df = panel_data[panel_data['group'] == target_group_label]
    
    daily_holdings = target_group_df.groupby(target_group_df.index)['code'].apply(set)
    daily_holdings = daily_holdings.asfreq('B').ffill() # 使用 .ffill() 避免警告
    
    turnovers = []
    for i in range(1, len(daily_holdings)):
        last_day_holdings = daily_holdings.iloc[i-1]
        current_day_holdings = daily_holdings.iloc[i]
        
        if not last_day_holdings or not current_day_holdings:
            turnovers.append(0)
            continue
        
        sell_count = len(last_day_holdings - current_day_holdings)
        last_day_total_count = len(last_day_holdings)
        
        daily_turnover = sell_count / last_day_total_count
        turnovers.append(daily_turnover)
        
    avg_daily_turnover = sum(turnovers) / len(turnovers) if turnovers else 0
    annualized_turnover = avg_daily_turnover * 252
    
    return annualized_turnover

# --- 3. 对每个因子进行分析 ---
# 【修改】将因子列更新为新的趋势因子
factor_columns = ['t_macross', 't_mamom', 't_maratio', 't_mom']

print("--- 趋势类因子组合换手率分析 ---")
for factor in factor_columns:
    print(f"\n--- 正在分析因子: {factor} ---")
    
    top_turnover = calculate_turnover(panel_data.copy(), factor, group_to_track='top') # 使用copy避免后续计算干扰
    print(f"多头组合 (Top Quintile) 年化换手率: {top_turnover:.2%}")
    
    bottom_turnover = calculate_turnover(panel_data.copy(), factor, group_to_track='bottom')
    print(f"空头组合 (Bottom Quintile) 年化换手率: {bottom_turnover:.2%}")

print("\n--- 换手率解读 ---")
print("换手率反映了维持投资组合所需的交易频率。")
print("较低的换手率通常意味着更低的交易成本和更稳定的持仓。")
print("非常高的换手率（如超过1000%）可能表明因子变化过快，策略在实际操作中可能面临较高的成本和冲击。")