# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:54:07 2025

@author: AKIRA
"""

# preprocess_by_sector.py
# (替换您旧的 '数据处理.py')

import os
import pandas as pd
import numpy as np
# 从我们刚创建的文件中导入映射关系
from product_sector_mapping import get_sector_from_filename

# --- 1. 配置路径 ---
# !! 修改这里: 您的原始因子数据文件夹 (例如价差类)
FACTOR_DATA_DIR = r'F:\Document\面向大宗商品期货的量化因子构建与策略应用研究\1价差类'

# 输出目录 (将自动创建在父文件夹下)
PANEL_OUTPUT_DIR = os.path.join(FACTOR_DATA_DIR, '分项')

# --- 2. 确保输出目录存在 ---
if not os.path.exists(PANEL_OUTPUT_DIR):
    os.makedirs(PANEL_OUTPUT_DIR)
    print(f"已创建输出文件夹: {PANEL_OUTPUT_DIR}")

# --- 3. 数据处理与分组 ---
# 使用字典来存储每个板块的数据
sector_data_frames = {}

# 遍历原始因子文件夹中的所有文件
all_files = os.listdir(FACTOR_DATA_DIR)
print(f"开始处理 {FACTOR_DATA_DIR} 中的 {len(all_files)} 个文件...")

for filename in all_files:
    
    # 仅处理CSV文件，并跳过任何已存在的panel_data文件
    if not filename.endswith('.csv') or 'panel_data' in filename:
        continue
        
    file_path = os.path.join(FACTOR_DATA_DIR, filename)

    # (关键步骤) 使用辅助函数从文件名获取板块
    sector = get_sector_from_filename(filename)
    
    # 如果无法识别或不是因子文件，则跳过
    if sector is None:
        print(f"  [跳过] 无法解析文件名: {filename}")
        continue
        
    try:
        # 读取单个品种的因子数据
        df = pd.read_csv(file_path, parse_dates=['date'])
        
        # 提取产品代码 (例如 'sr')
        product_code = filename.split('_')[1].lower()
        
        # --- 核心预处理逻辑 (与您旧脚本保持一致) ---
        # 1. 计算未来收益率 (用于IC和分层)
        #    (参考 'ic分层.py' 逻辑)
        df['forward_return'] = df['close'].shift(-1) / df['close'] - 1
        
        # 2. 添加品种和板块标识
        df['code'] = product_code
        df['sector'] = sector
        
        # 3. 确保数据完整性 (按需添加)
        # df = df.dropna(subset=['close', 'forward_return']) # 视情况决定是否在此处dropna
        
        # 4. 将处理后的DF添加到字典中
        if sector not in sector_data_frames:
            sector_data_frames[sector] = []
            
        sector_data_frames[sector].append(df)
        print(f"  [处理] {filename} -> 归类于: {sector}")

    except Exception as e:
        print(f"  [错误] 处理 {filename} 时失败: {e}")

# --- 4. 合并并保存每个板块的Panel Data ---
print("\n开始合并和保存分板块的Panel数据...")

for sector, df_list in sector_data_frames.items():
    if not df_list:
        continue
        
    # 合并该板块下的所有品种数据
    sector_panel_df = pd.concat(df_list, ignore_index=True)
    
    # 按日期和代码排序
    sector_panel_df.sort_values(by=['date', 'code'], inplace=True)
    
    # 定义输出文件名
    output_filename = f"{sector}_panel_data.csv"
    output_path = os.path.join(PANEL_OUTPUT_DIR, output_filename)
    
    # 保存
    sector_panel_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  [成功] 已保存: {output_filename} (包含 {len(df_list)} 个品种, 共 {len(sector_panel_df)} 行数据)")

print("\n--- 预处理完成 ---")