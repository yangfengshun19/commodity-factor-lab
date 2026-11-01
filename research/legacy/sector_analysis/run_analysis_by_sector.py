# run_all_analyses.py
# (V4 - 修正了 FACTOR_CONFIG)
#
# 功能：
# 1. 自动循环处理 1价差类 到 7波动类。
# 2. 自动管理路径，将所有分析结果（图表、汇总CSV、回归TXT）
#    输出到 '00分项分析' 下的对应结果文件夹。
# 3. 自动调用预处理和分析。

import os
import pandas as pd
import numpy as np

# (关键) 导入我们放在同个文件夹下的工具和映射
import analysis_functions as af
from product_sector_mapping import get_sector_from_filename

# --- 1. 全局配置 ---

# 您的项目根目录
BASE_DATA_DIR = r'F:\Document\面向大宗商品期货的量化因子构建与策略应用研究'

# 本脚本所在的目录 (也是所有分析结果的输出根目录)
BASE_ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))

# =================== START: 核心修正区域 ===================
#
# FACTOR_CONFIG (已根据您上传的 '螺纹钢' CSV 文件列名修正)
#
FACTOR_CONFIG = {
    # 1. 价差类 (经核对，原始配置正确)
    '1价差类': ['C_frontnext', 'C_mainsub', 'C_nsnormal'],
    
    # 2. 仓单类 (W_yoy -> warrant_count, 大小写)
    '2仓单类': ['warrant_count', 'W_pct', 'W_ratio'], 
    
    # 3. 期现类 (修正大小写：Fs -> fs)
    '3期现类': ['fs_basis', 'fs_carry', 'fs_exmom'],
    
    # 4. 量价趋势类 (修正大小写：T_ -> t_)
    '4量价趋势类': ['t_macross', 't_mamom', 't_maratio', 't_mom'],
    
    # 5. 持仓类 (修正为文件中具体的 'top5', 'top10' ... 名称)
    '5持仓类': [
        'pr_netmom_top5', 'pr_lschgratio_top5',
        'pr_netmom_top10', 'pr_lschgratio_top10',
        'pr_netmom_top15', 'pr_lschgratio_top15',
        'pr_netmom_top20', 'pr_lschgratio_top20'
    ],
    
    # 6. 价值类 (经核对，原始配置正确)
    '6价值类': ['Val_halfyear', 'Val_oneyear'],
    
    # 7. 波动类 (修正大小写：Vol_ -> vol_)
    '7波动类': ['vol_std', 'vol_gk', 'vol_rs']
}
# =================== END: 核心修正区域 ===================


# 自动获取所有要处理的因子文件夹名称
FACTOR_FOLDERS_TO_RUN = list(FACTOR_CONFIG.keys())

print(f"--- 自动化分析脚本启动 ---")
print(f"数据根目录: {BASE_DATA_DIR}")
print(f"分析根目录: {BASE_ANALYSIS_DIR}")
print(f"将处理以下因子大类: {', '.join(FACTOR_FOLDERS_TO_RUN)}\n")


# --- 2. 主循环：遍历所有因子大类 ---

for factor_folder_name in FACTOR_FOLDERS_TO_RUN:
    
    print(f"\n==========================================================")
    print(f"开始处理大类: {factor_folder_name}")
    print(f"==========================================================")

    # --- 2.A. 定义路径 ---
    
    # 1. 原始数据输入路径
    INPUT_DATA_DIR = os.path.join(BASE_DATA_DIR, factor_folder_name)
    
    # 2. 最终分析结果输出路径
    OUTPUT_RESULTS_DIR = os.path.join(BASE_ANALYSIS_DIR, f"{factor_folder_name}分析结果")
    
    # 3. 中间Panel Data的存放路径
    PANEL_DATA_DIR = os.path.join(OUTPUT_RESULTS_DIR, "Panel_Data")

    # 4. 获取当前大类对应的因子列表
    FACTORS_TO_TEST = FACTOR_CONFIG.get(factor_folder_name)

    if not os.path.exists(PANEL_DATA_DIR):
        os.makedirs(PANEL_DATA_DIR)
        print(f"已创建结果目录: {OUTPUT_RESULTS_DIR}")
        
    if not os.path.exists(INPUT_DATA_DIR):
        print(f"  [严重警告] 找不到原始数据文件夹: {INPUT_DATA_DIR}")
        print(f"  跳过 {factor_folder_name}")
        continue
    
    if not FACTORS_TO_TEST:
        print(f"  [严重警告] {factor_folder_name} 的因子配置为空。")
        print(f"  跳过 {factor_folder_name}")
        continue

    # --- 2.B. 执行预处理 (逻辑来自 preprocess_by_sector.py) ---
    print(f"\n--- (1/2) 预处理 {factor_folder_name} ---")
    sector_data_frames = {} 
    all_files = os.listdir(INPUT_DATA_DIR)

    for filename in all_files:
        if not filename.endswith('.csv') or 'panel_data' in filename:
            continue
            
        file_path = os.path.join(INPUT_DATA_DIR, filename)
        sector = get_sector_from_filename(filename)
        
        if sector is None:
            continue
            
        try:
            # 尝试 'utf-8'，不行回退到 'gbk'
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, parse_dates=['date'], encoding='gbk')

            df['forward_return'] = df['close'].shift(-1) / df['close'] - 1
            df['code'] = filename.split('_')[1].lower()
            df['sector'] = sector
            
            if sector not in sector_data_frames:
                sector_data_frames[sector] = []
            sector_data_frames[sector].append(df)
            
        except Exception as e:
            print(f"  [错误] 预处理 {filename} 失败: {e}")
    
    print("  预处理完成，正在保存分板块Panel数据...")
    
    for sector, df_list in sector_data_frames.items():
        if not df_list: continue
        sector_panel_df = pd.concat(df_list, ignore_index=True)
        sector_panel_df.sort_values(by=['date', 'code'], inplace=True)
        output_path = os.path.join(PANEL_DATA_DIR, f"{sector}_panel_data.csv")
        sector_panel_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
    print(f"  Panel数据已保存至: {PANEL_DATA_DIR}")

    # --- 2.C. 执行分析 (逻辑来自 run_analysis_by_sector.py) ---
    print(f"\n--- (2/2) 分析 {factor_folder_name} ---")
    
    all_summary_results = [] 

    for panel_filename in os.listdir(PANEL_DATA_DIR):
        
        if not panel_filename.endswith('.csv'): continue
            
        panel_file_path = os.path.join(PANEL_DATA_DIR, panel_filename)
        sector_name = panel_filename.replace('_panel_data.csv', '')
        
        print(f"\n--- N正在分析板块: {sector_name} ---")
        
        try:
            df_sector = pd.read_csv(panel_file_path, parse_dates=['date'])
            
            # 1. 执行回归
            af.analyze_regression(df_sector, FACTORS_TO_TEST, OUTPUT_RESULTS_DIR, sector_name)
            
            # 2. 循环分析每个因子
            for factor in FACTORS_TO_TEST:
                
                # 检查因子列是否存在
                if factor not in df_sector.columns:
                    print(f"  [警告] 因子 '{factor}' 不在 {sector_name} 的数据中，跳过。")
                    continue

                df_factor = df_sector[['date', 'code', 'forward_return', factor]].dropna()

                # 2.1 IC和分层
                ic_mean, ic_ir, ic_pos_rate = af.analyze_ic_layering(
                    df_factor, factor, OUTPUT_RESULTS_DIR, sector_name
                )
                
                # 2.2 换手率
                turnover = af.analyze_turnover(df_factor, factor)
                
                # 2.3 汇总结果
                result_row = {
                    '板块': sector_name,
                    '因子': factor,
                    'IC均值': ic_mean,
                    'ICIR': ic_ir,
                    'IC>0比例': ic_pos_rate,
                    '年化换手率': turnover
                }
                all_summary_results.append(result_row)
                
        except Exception as e:
            print(f"  [严重错误] 分析 {sector_name} 时失败: {e}")

    # --- 2.D. 保存该大类的汇总报告 ---
    if all_summary_results:
        summary_df = pd.DataFrame(all_summary_results)
        summary_df['IC均值'] = summary_df['IC均值'].map('{:,.4f}'.format)
        summary_df['ICIR'] = summary_df['ICIR'].map('{:,.4f}'.format)
        summary_df['IC>0比例'] = summary_df['IC>0比例'].map('{:,.2%}'.format)
        summary_df['年化换手率'] = summary_df['年化换手率'].map('{:,.2%}'.format)
        summary_df.sort_values(by=['板块', 'ICIR'], ascending=[True, False], inplace=True)
        
        summary_report_path = os.path.join(OUTPUT_RESULTS_DIR, f'analysis_summary_report_{factor_folder_name}.csv')
        summary_df.to_csv(summary_report_path, index=False, encoding='utf-8-sig')
        
        print(f"\n*** {factor_folder_name} 汇总报告已生成: {summary_report_path} ***")
        print(summary_df.to_string(max_rows=15)) # 打印前15行预览
    else:
        print(f"\n[警告] 未能为 {factor_folder_name} 生成任何分析结果。")

print(f"\n==========================================================")
print(f"--- 所有 {len(FACTOR_FOLDERS_TO_RUN)} 个因子大类已全部处理完毕 ---")
print(f"请检查 '00分项分析' 文件夹下的 '...分析结果' 子文件夹。")
print(f"==========================================================")