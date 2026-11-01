# run_backtest_simulation.py
# (V3 - 增加关键性能指标(KPI)的文字数据输出)
#
# 功能：
# 1. 执行向量化回测。
# 2. 计算毛/净收益的 年化收益、波动率、夏普、最大回撤、卡玛。
# 3. 在控制台打印KPI。
# 4. 将所有KPI汇总保存到 'backtest_performance_summary.csv'。

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 全局配置 ---

# 设置中文字体，解决图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 假设此脚本位于 '00分项分析' 文件夹
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# 回测数据仓库的路径
DATA_REPO_DIR = os.path.join(BASE_DIR, 'Backtest_Data_Repository')
# 回测图表与KPI报告输出路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'Backtest_Results') # 统一输出

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 交易成本配置
FEE_RATE = 0.0003     # 万分之三 (双边手续费)
SLIPPAGE = 0.0002     # 万分之二 (滑点成本)
TRADING_DAYS = 252    # 年化交易日数

# --- 2. 新增：绩效计算函数 ---
def calculate_performance_metrics(daily_returns, label=""):
    """计算详细的绩效指标"""
    
    # 确保 daily_returns 是 Series
    daily_returns = pd.Series(daily_returns)

    # 1. 累计净值
    cumulative_returns = (1 + daily_returns).cumprod()
    
    # 2. 年化收益率
    total_days = len(cumulative_returns)
    annualized_return = cumulative_returns.iloc[-1] ** (TRADING_DAYS / total_days) - 1
    
    # 3. 年化波动率
    annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)
    
    # 4. 夏普比率 (假设无风险利率为0)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    
    # 5. 最大回撤 (Max Drawdown)
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # 6. 卡玛比率 (Calmar Ratio)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    metrics = {
        f'{label}_年化收益': annualized_return,
        f'{label}_年化波动': annualized_volatility,
        f'{label}_夏普比率': sharpe_ratio,
        f'{label}_最大回撤': max_drawdown,
        f'{label}_卡玛比率': calmar_ratio
    }
    
    return metrics

# --- 3. 核心回测函数 (已修改) ---

def run_backtest(panel_data_path, factor_name, is_ic_negative=False):
    """
    执行向量化回测并返回绩效指标
    """
    
    sector = os.path.basename(panel_data_path).split('_')[0]
    print(f"\n--- 开始回测: {sector} | 因子: {factor_name} ---")

    # --- 3.A. 加载和准备数据 ---
    try:
        df = pd.read_csv(panel_data_path, parse_dates=['date'])
    except Exception as e:
        print(f"  [错误] 加载数据失败: {e}")
        return None

    cols_to_use = ['date', 'code', 'forward_return', factor_name]
    if factor_name not in df.columns:
        print(f"  [严重错误] 因子 '{factor_name}' 不在 {panel_data_path} 中。跳过...")
        return None
    df_factor = df[cols_to_use].dropna()
    if df_factor.empty:
        print("  [错误] 因子数据为空或 forward_return 为空。")
        return None

    # --- 3.B. 生成交易信号 (Positions) ---
    df_factor['group'] = df_factor.groupby('date')[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    df_factor['position'] = 0
    if is_ic_negative:
        df_factor.loc[df_factor['group'] == 0, 'position'] = 1
        df_factor.loc[df_factor['group'] == 4, 'position'] = -1
    else:
        df_factor.loc[df_factor['group'] == 4, 'position'] = 1
        df_factor.loc[df_factor['group'] == 0, 'position'] = -1
    print("  已生成交易信号...")

    # --- 3.C. 计算收益与成本 (向量化) ---
    positions_df = df_factor.pivot(index='date', columns='code', values='position').fillna(0)
    returns_df = df_factor.pivot(index='date', columns='code', values='forward_return').fillna(0)
    returns_df, positions_df = returns_df.align(positions_df, join='right', axis=None, fill_value=0)

    daily_gross_returns = (positions_df.shift(1) * returns_df).sum(axis=1)
    daily_turnover = (positions_df - positions_df.shift(1)).abs().sum(axis=1)
    daily_costs = daily_turnover * (FEE_RATE + SLIPPAGE)
    daily_net_returns = daily_gross_returns - daily_costs
    print("  已计算净值曲线...")

    # --- 3.D. 绩效评估与绘图 (已修改) ---
    
    # 1. (新增) 计算毛/净绩效指标
    gross_metrics = calculate_performance_metrics(daily_gross_returns, "毛收益")
    net_metrics = calculate_performance_metrics(daily_net_returns, "净收益")
    
    # 2. (新增) 合并所有指标用于返回和打印
    all_metrics = {
        '板块': sector,
        '因子': factor_name,
        **gross_metrics,
        **net_metrics
    }

    # 3. (新增) 在控制台打印详细文字数据
    print("\n  --- 回测性能指标 ---")
    print(f"  毛收益 年化: {all_metrics['毛收益_年化收益']:.2%}")
    print(f"  净收益 年化: {all_metrics['净收益_年化收益']:.2%}")
    print(f"  ---")
    print(f"  毛收益 夏普: {all_metrics['毛收益_夏普比率']:.2f}")
    print(f"  净收益 夏普: {all_metrics['净收益_夏普比率']:.2f}")
    print(f"  ---")
    print(f"  毛收益 最大回撤: {all_metrics['毛收益_最大回撤']:.2%}")
    print(f"  净收益 最大回撤: {all_metrics['净收益_最大回撤']:.2%}")
    print(f"  ----------------------")
    
    # 4. 绘图
    results_df = pd.DataFrame({
        '毛收益(Gross)': (1 + daily_gross_returns).cumprod(),
        '净收益(Net)': (1 + daily_net_returns).cumprod()
    })
    
    plt.figure(figsize=(14, 7))
    results_df.plot(ax=plt.gca(), grid=True)
    
    title = f'回测对比: {sector} - {factor_name}\n'
    title += f"毛收益Sharpe: {all_metrics['毛收益_夏普比率']:.2f} (回撤: {all_metrics['毛收益_最大回撤']:.2%}) | "
    title += f"净收益Sharpe: {all_metrics['净收益_夏普比率']:.2f} (回撤: {all_metrics['净收益_最大回撤']:.2%})\n"
    title += f'成本(Fee+Slip): {FEE_RATE*10000 + SLIPPAGE*10000:.1f} BP'
    
    plt.title(title, fontsize=10)
    plt.ylabel('累计净值')
    plt.xlabel('日期')
    
    img_path = os.path.join(OUTPUT_DIR, f'Backtest_{sector}_{factor_name}.png')
    plt.savefig(img_path)
    plt.close()
    
    print(f"  [成功] 回测图表已保存至: {img_path}")
    
    # 5. (新增) 返回文字数据
    return all_metrics

# --- 4. 执行回测 (扩展列表) ---

if __name__ == "__main__":
    
    # (新增) 用于存储所有回测的文字数据
    all_performance_results = []
    
    # --- 定义回测任务 ---
    # (板块文件名, 因子名, IC是否为负)
    backtest_tasks = [
        ('有色金属_master_panel.csv', 'warrant_count', True),       # 最佳实战
        ('新能源与新材料_master_panel.csv', 'pr_lschgratio_top10', False), # 高成本陷阱
        ('金融期货_master_panel.csv', 'pr_netmom_top5', False),
        ('黑色金属_master_panel.csv', 'pr_netmom_top5', False),
        ('能源化工_master_panel.csv', 't_mamom', True),
        ('农产品_master_panel.csv', 'C_mainsub', False),
        # --- (可选) 您可以按此格式添加更多任务 ---
        # ('贵金属_master_panel.csv', 'pr_lschgratio_top5', True),
        # ('其他_master_panel.csv', 'Val_oneyear', True),
    ]

    print(f"--- 准备执行 {len(backtest_tasks)} 项回测任务 ---")

    for task in backtest_tasks:
        panel_file, factor, is_negative = task
        panel_path = os.path.join(DATA_REPO_DIR, panel_file)
        
        # 检查文件是否存在
        if not os.path.exists(panel_path):
            print(f"\n[跳过] 未找到文件: {panel_file}")
            continue
            
        # 执行回测并捕获结果
        metrics = run_backtest(
            panel_data_path=panel_path,
            factor_name=factor,
            is_ic_negative=is_negative
        )
        
        if metrics:
            all_performance_results.append(metrics)

    # --- 5. (新增) 保存汇总的文字数据报告 ---
    if all_performance_results:
        print("\n--- 所有回测执行完毕 ---")
        
        # 转换为DataFrame
        summary_df = pd.DataFrame(all_performance_results)
        
        # 设置输出列的顺序
        columns_order = [
            '板块', '因子', 
            '净收益_年化收益', '净收益_夏普比率', '净收益_最大回撤', '净收益_卡玛比率', '净收益_年化波动',
            '毛收益_年化收益', '毛收益_夏普比率', '毛收益_最大回撤', '毛收益_卡玛比率', '毛收益_年化波动'
        ]
        # 确保所有列都存在
        final_columns = [col for col in columns_order if col in summary_df.columns]
        summary_df = summary_df[final_columns]
        
        # 按净夏普排序
        summary_df.sort_values(by='净收益_夏普比率', ascending=False, inplace=True)

        # 保存为CSV
        summary_path = os.path.join(OUTPUT_DIR, 'backtest_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig', float_format='%.4f')
        
        print(f"\n[成功] 所有回测的【文字数据汇总报告】已保存至:\n{summary_path}")
        
        # 在控制台打印最终的汇总表
        print("\n--- 最终回测绩效汇总 (按净夏普排序) ---")
        print(summary_df.to_string(float_format="%.4f"))

    else:
        print("\n--- 未能成功执行任何回测 ---") 