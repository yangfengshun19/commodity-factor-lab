# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:55:21 2025
Updated by Gemini on Sat Oct 11 2025.
Reverted to the original per-item fetching method for stability, while keeping the warm-up period logic.

@author: AKIRA (Original), Gemini Assistant (Modifications)
"""
import akshare as ak
import pandas as pd
import numpy as np
import os
import time
import re

# --- 1. 配置区 ---
PRODUCTS_TO_PROCESS = [
    ("螺纹钢", "rb"), ("热轧卷板", "hc"), ("石油沥青", "bu"), ("天然橡胶", "ru"), ("合成橡胶", "br"),
    ("燃料油", "fu"), ("纸浆", "sp"), ("铜", "cu"), ("铝", "al"), ("氧化铝", "ao"),
    ("铅", "pb"), ("锌", "zn"), ("锡", "sn"), ("镍", "ni"), ("不锈钢", "ss"),
    ("黄金", "au"), ("白银", "ag"), ("线材", "wr"),
    ("20号胶", "nr"), ("低硫燃料油", "lu"), ("国际铜", "bc"), ("原油", "sc"), ("集运指数(欧线)", "ec"),
    ("黄大豆1号", "a"), ("黄大豆2号", "b"), ("黄玉米", "c"), ("玉米淀粉", "cs"), ("豆粕", "m"),
    ("豆油", "y"), ("棕榈油", "p"), ("铁矿石", "i"), ("焦炭", "j"), ("焦煤", "jm"),
    ("聚乙烯", "l"), ("聚氯乙烯", "v"), ("聚丙烯", "pp"), ("乙二醇", "eg"), ("粳米", "rr"),
    ("苯乙烯", "eb"), ("液化石油气", "pg"), ("鸡蛋", "jd"), ("纤维板", "fb"), ("胶合板", "bb"),
    ("生猪", "lh"),
    ("菜粕", "RM"), ("菜籽油", "OI"), ("一号棉花", "CF"), ("精对苯二甲酸", "TA"), ("对二甲苯", "PX"),
    ("白砂糖", "SR"), ("甲醇", "MA"), ("玻璃", "FG"), ("动力煤", "ZC"), ("棉纱", "CY"),
    ("纯碱", "SA"), ("烧碱", "SH"), ("短纤", "PF"), ("瓶片", "PR"), ("粳稻", "JR"),
    ("菜籽", "RS"), ("普通小麦", "PM"), ("强麦", "WH"), ("早籼稻", "RI"), ("晚籼稻", "LR"),
    ("硅铁", "SF"), ("锰硅", "SM"), ("苹果", "AP"), ("红枣", "CJ"), ("尿素", "UR"),
    ("花生", "PK"),
    ("工业硅", "SI"), ("碳酸锂", "LC"), ("多晶硅", "PS"),
]
OUTPUT_FOLDER = "期货期现因子_CSV输出"
START_DATE_FILTER = "2023-01-01"
END_DATE_FILTER = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')

# --- 2. 辅助及主处理函数 ---
def sanitize_filename(filename):
    """移除或替换文件名中的非法字符"""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized

def process_single_future(product_name, product_code_short, start_date, end_date, output_folder):
    """
    处理单个期货品种，使用稳定但较慢的逐个获取现货数据的方法。
    """
    akshare_symbol = product_code_short.upper() + "0"
    print(f"\n正在处理品种: {product_name} ({product_code_short}), Akshare代码: {akshare_symbol}")

    try:
        # 步骤 1: 获取完整的历史期货数据
        df_futures_ohlc = ak.futures_zh_daily_sina(symbol=akshare_symbol)
        if df_futures_ohlc.empty:
            print(f"未能获取到 {product_name} ({akshare_symbol}) 的期货数据。")
            return
        
        # 步骤 2: 期货数据预处理
        df_futures_ohlc.rename(columns={'日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close'}, inplace=True)
        df_processed = df_futures_ohlc[['date', 'open', 'high', 'low', 'close']].copy()
        df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        df_processed.dropna(subset=['date'], inplace=True)
        df_processed.set_index('date', inplace=True)
        df_processed.sort_index(inplace=True)
        
        # 预热期 (Warm-up) 逻辑
        warm_up_days = 5 
        calculation_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=warm_up_days)).strftime('%Y-%m-%d')
        df_with_factors = df_processed.loc[calculation_start_date:end_date].copy()
        if df_with_factors.empty:
            print(f"{product_name} 在包含预热期的日期范围 {calculation_start_date} 到 {end_date} 内无期货数据。")
            return

        # 步骤 3: [已复原] 在循环内为当前品种获取现货数据
        print(f"正在为 {product_name} 获取现货价格数据...")
        try:
            spot_price_df = ak.futures_spot_price_daily(
                start_day=calculation_start_date.replace('-', ''),
                end_day=end_date.replace('-', ''),
                vars_list=[product_code_short.upper()]
            )
            
            if not spot_price_df.empty:
                spot_price_df['date'] = pd.to_datetime(spot_price_df['date'])
                spot_price_df.set_index('date', inplace=True)
                spot_data_to_merge = spot_price_df[['spot_price']]
                spot_data_to_merge['spot_price'] = pd.to_numeric(spot_data_to_merge['spot_price'], errors='coerce')
                df_with_factors = df_with_factors.join(spot_data_to_merge, how='left')
                print(f"成功获取并合并 {product_name} 的现货数据。")

                df_with_factors['fs_basis'] = df_with_factors['spot_price'] - df_with_factors['close']
                df_with_factors['fs_carry'] = df_with_factors['fs_basis'] / df_with_factors['close']
                futures_return = df_with_factors['close'].pct_change()
                spot_return = df_with_factors['spot_price'].pct_change()
                df_with_factors['fs_exmom'] = futures_return - spot_return
            else:
                print(f"警告: 未能获取到 {product_name} 的现货价格数据。Fs因子将为空。")
                for col in ['spot_price', 'fs_basis', 'fs_carry', 'fs_exmom']:
                    df_with_factors[col] = np.nan
        except Exception as e:
            print(f"获取 {product_name} 现货数据时发生错误: {e}。Fs因子将为空。")
            for col in ['spot_price', 'fs_basis', 'fs_carry', 'fs_exmom']:
                df_with_factors[col] = np.nan
        
        # 步骤 4: 筛选最终结果并保存
        df_final_output = df_with_factors.loc[start_date:end_date]
        if df_final_output.empty:
             print(f"{product_name} 在指定的最终输出日期范围 {start_date} 到 {end_date} 内无数据。")
             return

        safe_product_name = sanitize_filename(product_name)
        safe_product_code = sanitize_filename(product_code_short)
        csv_filename = f"{safe_product_name}_{safe_product_code}_basis_factors.csv"
        csv_filepath = os.path.join(output_folder, csv_filename)
        df_final_output.to_csv(csv_filepath, index=True, float_format='%.6f')
        print(f"成功为 {product_name} 计算基差因子并保存到: {csv_filepath}")

    except Exception as e:
        print(f"处理品种 {product_name} ({akshare_symbol}) 时发生严重错误: {e}")
        import traceback
        print(traceback.format_exc())

# --- 3. 主程序入口 ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"已创建输出文件夹: {OUTPUT_FOLDER}")

    total_products = len(PRODUCTS_TO_PROCESS)
    print("="*50)
    print(f"开始处理 {total_products} 个期货品种的期现因子...")
    print(f"数据最终输出日期范围: {START_DATE_FILTER} 至 {END_DATE_FILTER}")
    print("="*50)
    
    # [已复原] 简单的循环，不进行批量获取
    for i, (product_name, product_code_short) in enumerate(PRODUCTS_TO_PROCESS):
        print(f"\n--- [处理进度: {i+1}/{total_products}] ---")
        process_single_future(
            product_name, 
            product_code_short, 
            START_DATE_FILTER, 
            END_DATE_FILTER, 
            OUTPUT_FOLDER
        )
        # 为避免API请求过于频繁，保留一个合理的延时
        if i < total_products - 1:
            time.sleep(1.5)

    print("\n\n所有品种处理完毕。")
    print(f"请在文件夹 '{OUTPUT_FOLDER}' 中查看输出的CSV文件。")