# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 23:30:00 2025
@author: Gemini Assistant (Final Robust Version 2 on 2025-09-27)
Description: 修复了因数据缺失导致的TypeError，增强了计算的健壮性。
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import time
import re
from datetime import timedelta
import requests

pd.set_option('future.no_silent_downcasting', True)

# --- 1. 总配置区 ---
IS_TEST_RUN = False # 您可以设置为 False 来进行完整计算
USER_START_DATE = "2023-01-01"
USER_END_DATE = "2023-12-31"

PRODUCTS_TO_PROCESS = [
    ("螺纹钢", "RB"), ("热轧卷板", "HC"), ("石油沥青", "BU"), ("天然橡胶", "RU"), ("合成橡胶", "BR"),
    ("燃料油", "FU"), ("纸浆", "SP"), ("铜", "CU"), ("铝", "AL"), ("氧化铝", "ao"),
    ("铅", "PB"), ("锌", "ZN"), ("锡", "SN"), ("镍", "NI"), ("不锈钢", "SS"),
    ("黄金", "AU"), ("白银", "AG"), ("线材", "WR"),
    ("20号胶", "NR"), ("低硫燃料油", "LU"), ("国际铜", "BC"), ("原油", "SC"), ("集运指数(欧线)", "EC"),
    ("黄大豆1号", "A"), ("黄大豆2号", "B"), ("玉米", "C"), ("玉米淀粉", "CS"), ("豆粕", "M"),
    ("豆油", "Y"), ("棕榈油", "P"), ("铁矿石", "I"), ("焦炭", "J"), ("焦煤", "JM"),
    ("聚乙烯", "L"), ("聚氯乙烯", "V"), ("聚丙烯", "PP"), ("乙二醇", "EG"), ("粳米", "RR"),
    ("苯乙烯", "EB"), ("液化石油气", "PG"), ("鸡蛋", "JD"), ("纤维板", "FB"), ("胶合板", "BB"),
    ("生猪", "LH"),
    ("菜粕", "RM"), ("菜籽油", "OI"), ("一号棉花", "CF"), ("精对苯二甲酸", "TA"), ("对二甲苯", "PX"),
    ("白砂糖", "SR"), ("甲醇", "MA"), ("玻璃", "FG"), ("动力煤", "ZC"), ("棉纱", "CY"),
    ("纯碱", "SA"), ("烧碱", "SH"), ("短纤", "PF"), ("瓶片", "PR"), ("粳稻", "JR"),
    ("菜籽", "RS"), ("普通小麦", "PM"), ("强麦", "WH"), ("早籼稻", "RI"), ("晚籼稻", "LR"),
    ("硅铁", "SF"), ("锰硅", "SM"), ("苹果", "AP"), ("红枣", "CJ"), ("尿素", "UR"),
    ("花生", "PK"),
    ("工业硅", "SI"), ("碳酸锂", "LC"), ("多晶硅", "PS"),
    ("沪深300指数", "IF"), ("上证50指数", "IH"), ("中证500指数", "IC"), ("中证1000指数", "IM"),
    ("2年期国债", "TS"), ("5年期国债", "TF"), ("10年期国债", "T"), ("30年期国债", "TL")
]

OUTPUT_FOLDER = "期货持仓因子_CSV输出"
PR_NETMOM_LOOKBACK_PERIOD = 240
RANKS = ['top5', 'top10', 'top15', 'top20']
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 10

# --- 2. 辅助函数 ---
def sanitize_filename(filename):
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized

# --- 3. 主程序入口 ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    if IS_TEST_RUN:
        print("--- 正在以【快速测试模式】运行 ---")
        products_to_run = [("玉米", "C")] # 您可以只放一个品种来测试
        report_start_date = "2023-01-01"
        report_end_date = "2023-12-31"
    else:
        print("--- 正在以【完整计算模式】运行 ---")
        products_to_run = PRODUCTS_TO_PROCESS
        report_start_date = USER_START_DATE
        report_end_date = USER_END_DATE

    product_mapping = {code.upper(): name for name, code in products_to_run}
    all_product_codes = list(product_mapping.keys())
    
    # ... (数据获取部分不变) ...
    all_contracts_df = pd.DataFrame()
    fetch_end_date = pd.to_datetime(report_end_date)
    fetch_start_date = pd.to_datetime(report_start_date) - timedelta(days=PR_NETMOM_LOOKBACK_PERIOD + 180)
    start_date_str = fetch_start_date.strftime('%Y%m%d')
    end_date_str = fetch_end_date.strftime('%Y%m%d')
    for attempt in range(MAX_RETRIES):
        try:
            print(f"--- Attempt {attempt + 1} of {MAX_RETRIES} ---")
            print(f"步骤1: 批量获取 {len(all_product_codes)} 个品种的持仓数据...")
            all_contracts_df = ak.get_rank_sum_daily(start_day=start_date_str, end_day=end_date_str, vars_list=all_product_codes)
            print("持仓数据下载成功！")
            break
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError) as e:
            print(f"发生网络错误: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"将在 {RETRY_DELAY_SECONDS} 秒后重试...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("已达到最大重试次数，程序将终止。")
                all_contracts_df = pd.DataFrame()
                break
    
    if not all_contracts_df.empty:
        try:
            print("持仓数据汇总中...")
            position_cols = [col for col in all_contracts_df.columns if 'interest' in col]
            agg_dict = {col: 'sum' for col in position_cols}
            all_varieties_df = all_contracts_df.groupby(['date', 'variety']).agg(agg_dict).reset_index()
            all_varieties_df['date'] = pd.to_datetime(all_varieties_df['date'])
            all_varieties_df.set_index('date', inplace=True)
            grouped_by_variety = all_varieties_df.groupby('variety')
            print("持仓数据汇总完毕。")

            total_products = len(grouped_by_variety)
            for i, (product_code, group_df) in enumerate(grouped_by_variety):
                try:
                    product_name = product_mapping.get(product_code, product_code)
                    print(f"\n--- [{i+1}/{total_products}] 正在处理: {product_name} ({product_code}) ---")
                    
                    df_factors = group_df.copy()
                    full_date_range = pd.date_range(start=df_factors.index.min(), end=df_factors.index.max())
                    df_factors = df_factors.reindex(full_date_range).sort_index()

                    for rank in RANKS:
                        long_col = f'long_open_interest_{rank}'
                        short_col = f'short_open_interest_{rank}'
                        if long_col not in df_factors.columns or short_col not in df_factors.columns: continue
                        
                        # --- [FIXED] 修复TypeError错误 ---
                        # 1. 先将可能存在问题的列强制转换为数字，无法转换的变为NaN（无效数字）
                        long_positions = pd.to_numeric(df_factors[long_col], errors='coerce')
                        short_positions = pd.to_numeric(df_factors[short_col], errors='coerce')
                        
                        # 2. 使用.sub()方法进行减法，并用fill_value=0将NaN视作0来处理
                        df_factors[f'net_long_{rank}'] = long_positions.sub(short_positions, fill_value=0)
                        
                        df_factors[f'pr_netmom_{rank}'] = df_factors[f'net_long_{rank}'].diff(PR_NETMOM_LOOKBACK_PERIOD)
                        
                        long_chg = long_positions.diff(1)
                        short_chg = short_positions.diff(1)
                        denominator = short_chg.replace(0, np.nan)
                        df_factors[f'pr_lschgratio_{rank}'] = long_chg / denominator
                    
                    df_factors.replace([np.inf, -np.inf], np.nan, inplace=True)
                    
                    print(f"  > 正在获取 {product_name} 的价格数据...")
                    df_ohlc = ak.futures_zh_daily_sina(symbol=product_code.upper() + "0")
                    df_ohlc['date'] = pd.to_datetime(df_ohlc['date'])
                    df_ohlc.set_index('date', inplace=True)
                    df_ohlc = df_ohlc[['open', 'high', 'low', 'close']].copy()

                    df_merged = df_ohlc.join(df_factors, how='inner')
                    columns_to_keep = ['open', 'high', 'low', 'close']
                    factor_columns = [col for col in df_merged.columns if 'pr_netmom' in col or 'pr_lschgratio' in col]
                    columns_to_keep.extend(factor_columns)
                    df_final = df_merged[columns_to_keep]
                    df_output = df_final.loc[report_start_date:report_end_date].copy().dropna(how='all', subset=factor_columns)

                    if df_output.empty:
                         print(f"  > {product_name} 在指定日期范围内无有效数据可输出。")
                         continue
                    
                    safe_product_name = sanitize_filename(product_name)
                    safe_product_code = sanitize_filename(product_code)
                    csv_filename = f"{safe_product_name}_{safe_product_code}_final_factors.csv"
                    csv_filepath = os.path.join(OUTPUT_FOLDER, csv_filename)
                    
                    df_output.to_csv(csv_filepath, index=True, float_format='%.4f')
                    print(f"  > 成功为 {product_name} 生成最终因子文件: {csv_filepath}")
                
                except Exception as e:
                    product_name_for_error = product_mapping.get(product_code, product_code)
                    print(f"\n  > [错误] 处理品种 {product_name_for_error} ({product_code}) 时发生意外，已跳过。")
                    print(f"  > 错误详情: {e}")

        except Exception as e:
            print(f"\n数据处理过程中发生严重错误: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print("\n数据下载失败，程序已终止。")
    
    print("\n所有品种处理完毕。")