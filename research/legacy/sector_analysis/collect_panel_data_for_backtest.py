# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 10:39:58 2025

@author: AKIRA
"""

# collect_panel_data_for_backtest.py
#
# 功能：
# 这是一个用于“阶段二：回测”的专属预处理脚本。
# 它会扫描所有 '...分析结果' 文件夹中的 'Panel_Data'，
# 按“板块”将所有分散的 _panel_data.csv 文件合并，
# 最终生成一个“Backtest_Data_Repository”文件夹，
# 内部包含按板块划分的“大师面板数据”（Master Panel）。

import os
import pandas as pd
import glob
import time

print("--- 回测预处理脚本启动 ---")

# --- 1. 定义路径 ---

# 脚本所在的 '00分项分析' 文件夹
BASE_ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))

# 新的、统一的回测数据输出文件夹
NEW_BACKTEST_DATA_DIR = os.path.join(BASE_ANALYSIS_DIR, 'Backtest_Data_Repository')

# 确保输出目录存在
if not os.path.exists(NEW_BACKTEST_DATA_DIR):
    os.makedirs(NEW_BACKTEST_DATA_DIR)
    print(f"已创建回测数据仓库: {NEW_BACKTEST_DATA_DIR}")

# --- 2. 查找所有分散的 _panel_data.csv 文件 ---

# 搜索路径： F:\...\00分项分析\ *分析结果 \Panel_Data\ *.csv
search_pattern = os.path.join(BASE_ANALYSIS_DIR, '*分析结果', 'Panel_Data', '*_panel_data.csv')

print(f"正在搜索路径: {search_pattern}")
all_panel_files = glob.glob(search_pattern)

if not all_panel_files:
    print("\n[严重错误] 未找到任何 '..._panel_data.csv' 文件。")
    print("请确保您已成功运行 'run_all_analyses.py' 脚本，")
    print("并且数据位于 '...分析结果\\Panel_Data\\' 文件夹中。")
    exit()

print(f"\n成功找到 {len(all_panel_files)} 个分散的Panel数据文件。")

# --- 3. 按“板块”对文件进行分组 ---

# (e.g., '有色金属': [file1_path, file2_path, ...])
sector_file_groups = {}

for f_path in all_panel_files:
    # 从文件名 '...\\有色金属_panel_data.csv' 中提取 '有色金属'
    sector_name = os.path.basename(f_path).replace('_panel_data.csv', '')
    
    if sector_name not in sector_file_groups:
        sector_file_groups[sector_name] = []
        
    sector_file_groups[sector_name].append(f_path)

print(f"已将文件按 {len(sector_file_groups)} 个板块进行分组。")

# --- 4. 循环合并每个板块的数据 ---

# 回测的关键合并键
KEY_COLS = ['date', 'code']
# 需要去重的基础列 (防止合并时产生 open_x, open_y 等)
COMMON_COLS = ['open', 'high', 'low', 'close', 'forward_return', 'sector']

total_sectors_processed = 0

for sector_name, df_path_list in sector_file_groups.items():
    
    print(f"\n--- 正在处理板块: {sector_name} ---")
    
    if not df_path_list:
        continue
        
    master_df = None # 该板块的大师DF
    
    # 循环读取该板块的每一个文件 (来自 1价差类, 2仓单类...)
    for i, f_path in enumerate(df_path_list):
        try:
            # 尝试 'utf-8'，不行回退到 'gbk'
            try:
                df = pd.read_csv(f_path, parse_dates=['date'])
            except UnicodeDecodeError:
                df = pd.read_csv(f_path, parse_dates=['date'], encoding='gbk')

            # 提取因子列 (排除 KEY_COLS 和 COMMON_COLS)
            factor_cols = [col for col in df.columns if col not in KEY_COLS and col not in COMMON_COLS]
            
            # (重要) 只保留 键 + 因子列
            # 对于第一个文件(i==0)，我们保留基础数据
            if i == 0:
                master_df = df
                print(f"  (1/{len(df_path_list)}) 加载基础数据 (来自: {f_path})")
            else:
                # 对于后续文件，我们只提取 键 + 因子列 进行合并
                df_to_merge = df[KEY_COLS + factor_cols]
                
                # 使用 'outer' 合并，确保所有日期都被保留
                master_df = pd.merge(master_df, df_to_merge, on=KEY_COLS, how='outer')
                print(f"  ({i+1}/{len(df_path_list)}) 合并因子: {', '.join(factor_cols)}")
        
        except Exception as e:
            print(f"  [错误] 处理文件 {f_path} 时失败: {e}")
            
    # --- 5. 保存大师面板数据 (Master Panel) ---
    if master_df is not None:
        # 按日期和代码排序
        master_df.sort_values(by=KEY_COLS, inplace=True)
        
        output_filename = f"{sector_name}_master_panel.csv"
        output_path = os.path.join(NEW_BACKTEST_DATA_DIR, output_filename)
        
        master_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  [成功] 已保存大师面板数据: {output_filename}")
        print(f"  总计 {len(master_df.columns)} 列 (包含 {len(master_df.columns) - len(COMMON_COLS) - len(KEY_COLS) + 1} 个因子)")
        total_sectors_processed += 1

print(f"\n--- 所有 {total_sectors_processed} 个板块处理完毕 ---")
print(f"回测数据现已准备就绪，位于: {NEW_BACKTEST_DATA_DIR}")