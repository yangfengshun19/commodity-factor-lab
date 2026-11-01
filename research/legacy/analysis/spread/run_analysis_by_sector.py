# run_analysis_by_sector.py
# (V2 - 优化结果输出)

import os
import pandas as pd
import analysis_functions as af 

# --- 1. 配置路径 ---
# *** 注意：已按您的要求更新为新路径 ***
BASE_DIR = r'F:\Document\面向大宗商品期货的量化因子构建与策略应用研究'
FACTOR_FOLDER_NAME = '1价差类' # !! 当您分析'2仓单类'时，请修改这里

# 自动构建路径
PANEL_DATA_DIR = os.path.join(BASE_DIR, FACTOR_FOLDER_NAME, '分项')
OUTPUT_DIR = os.path.join(PANEL_DATA_DIR, 'analysis_results')

# --- 2. 配置因子 ---
# 因子配置 (方便切换)
FACTOR_CONFIG = {
    '1价差类': ['C_frontnext', 'C_mainsub', 'C_nsnormal'],
    '2仓单类': ['W_yoy', 'W_pct', 'W_ratio'],
    '3期现类': ['Fs_basis', 'Fs_carry', 'Fs_exmom'],
    '4量价趋势类': ['T_macross', 'T_mamom', 'T_maratio', 'T_mom'],
    '5持仓类': ['Pr_netmom', 'Pr_lschgratio'],
    '6价值类': ['Val_halfyear', 'Val_oneyear'],
    '7波动类': ['Vol_std', 'Vol_GK', 'Vol_RS']
}

FACTORS_TO_TEST = FACTOR_CONFIG.get(FACTOR_FOLDER_NAME)
if FACTORS_TO_TEST is None:
    print(f"错误: 未在 FACTOR_CONFIG 中找到 '{FACTOR_FOLDER_NAME}' 的配置。")
    exit()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 3. 循环分析每个板块 ---
print(f"开始分析 {PANEL_DATA_DIR} 中的所有板块...")

# *** 新增：用于存储所有结果的列表 ***
all_summary_results = []

for filename in os.listdir(PANEL_DATA_DIR):
    
    if not filename.endswith('.csv') or not 'panel_data' in filename:
        continue
        
    file_path = os.path.join(PANEL_DATA_DIR, filename)
    sector_name = filename.replace('_panel_data.csv', '')
    
    print(f"\n==============================================")
    print(f"  正在加载板块: {sector_name}")
    print(f"  测试因子: {', '.join(FACTORS_TO_TEST)}")
    print(f"==============================================")
    
    try:
        df_sector = pd.read_csv(file_path, parse_dates=['date'])
        
        # --- 3.A. 执行多元回归 (调用已修改的函数) ---
        af.analyze_regression(df_sector, FACTORS_TO_TEST, OUTPUT_DIR, sector_name)
        
        # --- 3.B. 循环分析该板块的每一个因子 ---
        for factor in FACTORS_TO_TEST:
            
            df_factor = df_sector[['date', 'code', 'forward_return', factor]].dropna()

            # --- 捕获IC/分层结果 ---
            ic_mean, ic_ir, ic_pos_rate = af.analyze_ic_layering(
                df_factor, factor, OUTPUT_DIR, sector_name
            )
            
            # --- 捕获换手率结果 ---
            turnover = af.analyze_turnover(df_factor, factor)
            
            # --- 新增：将结果存入字典 ---
            result_row = {
                '板块': sector_name,
                '因子': factor,
                'IC均值': ic_mean,
                'ICIR': ic_ir,
                'IC>0比例': ic_pos_rate,
                '年化换手率': turnover
            }
            # 将该行的结果添加到总列表中
            all_summary_results.append(result_row)
            
    except Exception as e:
        print(f"  [严重错误] 分析 {sector_name} 时失败: {e}")

# --- 4. (核心) 保存汇总报告 ---
print("\n--- 所有板块分析完成 ---")

if all_summary_results:
    # 将结果列表转换为DataFrame
    summary_df = pd.DataFrame(all_summary_results)
    
    # 格式化输出
    summary_df['IC均值'] = summary_df['IC均值'].map('{:,.4f}'.format)
    summary_df['ICIR'] = summary_df['ICIR'].map('{:,.4f}'.format)
    summary_df['IC>0比例'] = summary_df['IC>0比例'].map('{:,.2%}'.format)
    summary_df['年化换手率'] = summary_df['年化换手率'].map('{:,.2%}'.format)
    
    # 按板块和ICIR排序，方便查看
    summary_df.sort_values(by=['板块', 'ICIR'], ascending=[True, False], inplace=True)
    
    # 定义汇总报告的路径
    summary_report_path = os.path.join(PANEL_DATA_DIR, 'analysis_summary_report.csv')
    
    # 保存为CSV
    summary_df.to_csv(summary_report_path, index=False, encoding='utf-8-sig')
    
    print(f"已成功生成汇总报告: {summary_report_path}")
    print("\n汇总结果预览：")
    print(summary_df.to_string())
else:
    print("未能生成任何分析结果。")