import pandas as pd

# --- 1. 加载预处理好的面板数据 ---
# 【修改】读取新的价值因子面板数据文件
try:
    panel_data = pd.read_csv('panel_data_for_value_factor_analysis.csv')
except FileNotFoundError:
    print("错误：未找到 'panel_data_for_value_factor_analysis.csv' 文件。")
    print("请先运行修改后的 '数据处理.py' 脚本来生成这个文件。")
    exit()

panel_data['date'] = pd.to_datetime(panel_data['date'])
panel_data.set_index('date', inplace=True)


# --- 2. 换手率计算函数 ---
def calculate_turnover(panel_data, factor_name, group_to_track='top'):
    """
    计算指定因子、指定分组的换手率
    
    参数:
    - panel_data: 包含日期、代码、因子值的数据框
    - factor_name: 要分析的因子列名
    - group_to_track: 要追踪的分组, 'top' (多头) 或 'bottom' (空头)
    
    返回:
    - 年化换手率
    """
    
    # 按因子值分组 (分为5组)
    panel_data['group'] = panel_data.groupby(panel_data.index)[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    
    # 确定目标分组的标签
    if group_to_track == 'top':
        target_group_label = 4  # qcut 标签从0到4，4是最大的一组
    elif group_to_track == 'bottom':
        target_group_label = 0  # 0是最小的一组
    else:
        raise ValueError("group_to_track 必须是 'top' 或 'bottom'")

    # 筛选出目标分组的数据
    target_group_df = panel_data[panel_data['group'] == target_group_label]
    
    # 获取每个交易日的持仓列表
    daily_holdings = target_group_df.groupby(target_group_df.index)['code'].apply(set)
    
    # 转换为时间序列，并填充缺失的日期（如果有的话）
    daily_holdings = daily_holdings.asfreq('B').ffill()
    
    turnovers = []
    # 从第二个交易日开始计算
    for i in range(1, len(daily_holdings)):
        last_day_holdings = daily_holdings.iloc[i-1]
        current_day_holdings = daily_holdings.iloc[i]
        
        # 如果任一天的持仓为空，则当天的换手率为0
        if not last_day_holdings or not current_day_holdings:
            turnovers.append(0)
            continue
        
        # 计算卖出的品种数量
        sell_count = len(last_day_holdings - current_day_holdings)
        # 前一天的总持仓数量
        last_day_total_count = len(last_day_holdings)
        
        # 计算当日换手率
        daily_turnover = sell_count / last_day_total_count
        turnovers.append(daily_turnover)
        
    # 计算平均日换手率
    avg_daily_turnover = sum(turnovers) / len(turnovers) if turnovers else 0
    
    # 年化换手率 (假设一年252个交易日)
    annualized_turnover = avg_daily_turnover * 252
    
    return annualized_turnover

# --- 3. 对每个因子进行分析 ---
# 【修改】将因子列更新为新的价值因子
factor_columns = ['Val_halfyear', 'Val_oneyear']

print("--- 因子组合换手率分析 ---")
for factor in factor_columns:
    print(f"\n--- 正在分析因子: {factor} ---")
    
    # 计算多头组合（因子值最大的一组）的换手率
    top_turnover = calculate_turnover(panel_data, factor, group_to_track='top')
    print(f"多头组合 (Top Quintile) 年化换手率: {top_turnover:.2%}")
    
    # 计算空头组合（因子值最小的一组）的换手率
    bottom_turnover = calculate_turnover(panel_data, factor, group_to_track='bottom')
    print(f"空头组合 (Bottom Quintile) 年化换手率: {bottom_turnover:.2%}")

print("\n--- 换手率解读 ---")
print("换手率反映了维持投资组合所需的交易频率。")
print("较低的换手率通常意味着更低的交易成本和更稳定的持仓。")
print("非常高的换手率（如超过1000%）可能表明因子变化过快，策略在实际操作中可能面临较高的成本和冲击。")