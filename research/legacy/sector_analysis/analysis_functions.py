# analysis_functions.py
# (V2 - 优化结果输出，无需改动)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os 

# --- 解决中文乱码问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False   
# --- 修改结束 ---


# --- 1. 来自 'ic分层.py' 的逻辑 ---
def analyze_ic_layering(df, factor_name, output_dir, sector_name):
    """
    执行IC分析和分层回测
    (修改：返回IC指标)
    """
    print(f"\n... (IC与分层) 正在分析: {factor_name} ...")
    
    # 初始化返回值
    ic_mean, ic_ir, ic_positive_rate = np.nan, np.nan, np.nan
    
    if df.empty or df[factor_name].isnull().all():
        print(f"  [警告] {factor_name} 数据为空或全为NaN，跳过IC分层。")
        return ic_mean, ic_ir, ic_positive_rate # 返回NaN

    # 1.1 IC 分析
    ic_series = df.groupby('date').apply(
        lambda x: x[factor_name].corr(x['forward_return'], method='spearman')
    )
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std != 0 else np.nan
    ic_positive_rate = (ic_series > 0).mean()
    
    print(f"  IC均值: {ic_mean:.4f}, ICIR: {ic_ir:.4f}")

    # 1.2 分层回测 (绘图部分)
    df['group'] = df.groupby('date')[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    grouped_returns = df.groupby(['date', 'group'])['forward_return'].mean().unstack()
    cumulative_returns = (1 + grouped_returns.fillna(0)).cumprod()
    
    plt.figure(figsize=(12, 6))
    plot_title = f'{sector_name} - {factor_name} - 分层回测'
    cumulative_returns.plot(title=plot_title) 
    plt.legend(title='分组(Group)')
    plt.ylabel('累计收益(Cumulative Return)')
    plt.xlabel('日期(Date)')
    plt.grid(True)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, f"{sector_name}_{factor_name}_layering.png"))
    plt.close()
    print(f"  已保存分层图表。")
    
    return ic_mean, ic_ir, ic_positive_rate


# --- 2. 来自 '换手率.py' 的逻辑 ---
def analyze_turnover(df, factor_name):
    """
    计算Top和Bottom组的年化换手率
    """
    print(f"... (换手率) 正在分析: {factor_name} ...")
    
    if df.empty or df[factor_name].isnull().all():
        print(f"  [警告] {factor_name} 数据为空或全为NaN，跳过换手率。")
        return np.nan

    df['group'] = df.groupby('date')[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=[0, 1, 2, 3, 4], duplicates='drop')
    )
    target_groups = df[df['group'].isin([0, 4])]
    daily_holdings = target_groups.groupby(['date', 'group'])['code'].apply(set)
    turnovers = []
    unique_indices = daily_holdings.index.get_level_values(0).unique()[1:]
    
    for date in unique_indices:
        for group in [0, 4]:
            if (date, group) not in daily_holdings.index: continue
            prev_date = daily_holdings.index.get_level_values(0)[daily_holdings.index.get_level_values(0) < date].max()
            if (prev_date, group) not in daily_holdings.index: continue
            
            last_day_holdings = daily_holdings.loc[(prev_date, group)]
            current_day_holdings = daily_holdings.loc[(date, group)]
            
            if not last_day_holdings: continue

            sell_count = len(last_day_holdings - current_day_holdings)
            buy_count = len(current_day_holdings - last_day_holdings)
            daily_turnover = (sell_count + buy_count) / (2 * len(last_day_holdings))
            turnovers.append(daily_turnover)

    avg_daily_turnover = np.mean(turnovers) if turnovers else 0
    annualized_turnover = avg_daily_turnover * 252 
    
    print(f"  年化换手率 (多空组平均): {annualized_turnover:.2%}")
    return annualized_turnover


# --- 3. 来自 '回归分析.py' 的逻辑 ---
def analyze_regression(df, factor_list, output_dir, sector_name):
    """
    执行多元回归
    (修改：将结果保存到txt文件)
    """
    print(f"\n... (多元回归) 正在分析板块: {sector_name} ...")
    
    valid_factors = [f for f in df.columns if f in factor_list and not df[f].isnull().all()]
    if not valid_factors:
        print("  [警告] 回归所需因子均无效或缺失，跳过。")
        return

    df_reg = df[['forward_return'] + valid_factors].dropna()
    
    if df_reg.empty:
        print("  [警告] 剔除NaN后回归数据为空，跳过。")
        return

    Y = df_reg['forward_return']
    X = df_reg[valid_factors]
    X = sm.add_constant(X)
    
    try:
        model = sm.OLS(Y, X)
        results = model.fit()
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        summary_path = os.path.join(output_dir, f"{sector_name}_regression_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"--- {sector_name} 多元回归结果 (因子: {', '.join(valid_factors)}) ---\n\n")
            f.write(results.summary().as_text())
        
        print(f"  已保存回归报告: {summary_path}")

    except Exception as e:
        print(f"  [错误] {sector_name} 回归分析失败: {e}")