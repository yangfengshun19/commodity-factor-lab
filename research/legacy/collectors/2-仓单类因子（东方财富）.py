# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:21:55 2025
Updated by Gemini on Sun Oct 12 2025 to merge OHLC data and align with other factor scripts.

@author: AKIRA (Original), Gemini Assistant (Modifications)
"""
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import os
import re
from datetime import datetime, timedelta
import time
import akshare as ak

# 全局禁用SSL证书验证 (在某些网络环境下需要)

# --- 1. 配置区 ---
PRODUCTS_TO_PROCESS = [
    ("螺纹钢", "rb"), ("热轧卷板", "hc"), ("石油沥青", "bu"), ("天然橡胶", "ru"), ("合成橡胶", "br"),
    ("燃料油", "fu"), ("纸浆", "sp"), ("铜", "cu"), ("铝", "al"), ("氧化铝", "ao"),
    ("铅", "pb"), ("锌", "zn"), ("锡", "sn"), ("镍", "ni"), ("不锈钢", "ss"),
    ("黄金", "au"), ("白银", "ag"), ("线材", "wr"),
    ("20号胶", "nr"), ("低硫燃料油", "lu"), ("国际铜", "bc"), ("原油", "sc"),
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
    # 注意：金融指数和国债没有仓单数据，会在后续逻辑中被自动跳过
    ("沪深300指数", "IF"), ("上证50指数", "IH"), ("中证500指数", "IC"), ("中证1000指数", "IM"),
    ("2年期国债", "TS"), ("5年期国债", "TF"), ("10年期国债", "T"), ("30年期国债", "TL"), ("集运指数(欧线)", "ec"),
]

# 东方财富仓单接口的名称映射
SYMBOL_MAP_EM = {
    "铜": "沪铜", "铝": "沪铝", "铅": "沪铅", "锌": "沪锌", "黄金": "沪金",
    "白银": "沪银", "黄大豆1号": "豆一", "黄大豆2号": "豆二", "黄玉米": "玉米", "一号棉花": "郑棉",
    "精对苯二甲酸": "PTA", "聚氯乙烯": "PVC", "聚乙烯": "塑料", "棕榈油": "棕榈", "白砂糖": "白糖",
    "菜籽油": "菜油", "合成橡胶": "丁二烯橡胶", "石油沥青": "沥青", "燃料油": "燃油", "天然橡胶": "橡胶",
    "热轧卷板": "热卷"
}

# 因子计算参数
LOOKBACK_PERIOD_EM = 50 # 仓单因子回看窗口 (约2个多月)

# 输出与日期配置
OUTPUT_FOLDER = "期货仓单因子_CSV输出_东方财富"
START_DATE_FILTER = "2023-01-01"
END_DATE_FILTER = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')

# 定义哪些品种不是实物商品，不应查询仓单
NON_PHYSICAL_PRODUCTS = {
    "IF", "IH", "IC", "IM", "TS", "TF", "T", "TL", "ec"
}


# --- 2. 因子计算与辅助函数 ---
def calculate_w_pct(series_warrant_count, window=LOOKBACK_PERIOD_EM):
    """计算仓单量的滚动百分位"""
    return series_warrant_count.rolling(window=window, min_periods=max(1, int(window * 0.8))).apply(
        lambda x: percentileofscore(x[~np.isnan(x)], x[-1], kind='rank') if pd.notna(x[-1]) and np.count_nonzero(~np.isnan(x)) > 1 else np.nan,
        raw=True
    ) / 100.0

def calculate_w_ratio(series_warrant_count, window=LOOKBACK_PERIOD_EM):
    """计算当前仓单量与滚动均值的比率"""
    rolling_mean = series_warrant_count.rolling(window=window, min_periods=max(1, int(window * 0.8))).mean()
    w_ratio = series_warrant_count / rolling_mean
    return w_ratio

def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized


# --- 3. 主处理函数 ---
def process_single_future(product_name, product_code, start_date, end_date, output_folder):
    """
    处理单个期货品种：获取OHLC和仓单数据，合并，计算因子，然后保存。
    (已修正pandas警告)
    """
    if product_code in NON_PHYSICAL_PRODUCTS:
        print(f"\n--- 品种: {product_name} ({product_code}) ---")
        print(f"    - 跳过: 非实物商品，无仓单数据。")
        return

    akshare_symbol = product_code.upper() + "0"
    print(f"\n--- 正在处理品种: {product_name} ({product_code}), Akshare代码: {akshare_symbol} ---")

    try:
        # --- 步骤 1: 获取OHLC价格数据 ---
        df_ohlc = ak.futures_zh_daily_sina(symbol=akshare_symbol)
        if df_ohlc.empty:
            print(f"    - 未能获取到 {product_name} 的OHLC数据。跳过。")
            return
        
        df_ohlc.rename(columns={'日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close'}, inplace=True)
        df_ohlc_processed = df_ohlc[['date', 'open', 'high', 'low', 'close']].copy()
        df_ohlc_processed['date'] = pd.to_datetime(df_ohlc_processed['date'], errors='coerce')
        df_ohlc_processed.dropna(subset=['date'], inplace=True)
        df_ohlc_processed.set_index('date', inplace=True)
        df_ohlc_processed.sort_index(inplace=True)
        print(f"    > 获取到 {product_name} 原始OHLC数据 {df_ohlc.shape[0]} 条。")

        # --- 步骤 2: 获取仓单数据 ---
        api_symbol = SYMBOL_MAP_EM.get(product_name, product_name)
        print(f"    > 正在使用查询名称 '{api_symbol}' 获取仓单数据...")
        df_warrant_raw = ak.futures_inventory_em(symbol=api_symbol)
        
        if df_warrant_raw.empty:
            print(f"    - 未能获取到 {product_name} 的仓单数据。跳过。")
            return

        df_warrant_raw.rename(columns={'日期': 'date', '库存': 'warrant_count'}, inplace=True)
        df_warrant_processed = df_warrant_raw[['date', 'warrant_count']].copy()
        df_warrant_processed['date'] = pd.to_datetime(df_warrant_processed['date'], errors='coerce')
        df_warrant_processed.dropna(subset=['date'], inplace=True)
        df_warrant_processed['warrant_count'] = pd.to_numeric(df_warrant_processed['warrant_count'], errors='coerce')
        df_warrant_processed.set_index('date', inplace=True)
        df_warrant_processed.sort_index(inplace=True)
        print(f"    > 获取到 {product_name} 仓单数据 {df_warrant_processed.shape[0]} 条。")

        # --- 步骤 3: 确定包含预热期的数据范围 ---
        warm_up_days = int(LOOKBACK_PERIOD_EM * 1.5)
        calculation_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=warm_up_days)).strftime('%Y-%m-%d')
        
        df_ohlc_final = df_ohlc_processed.loc[calculation_start_date:end_date]

        # --- 步骤 4: 合并OHLC和仓单数据 ---
        df_merged = df_ohlc_final.join(df_warrant_processed, how='left')
        
        # --- [已修正] 使用 .ffill() 并重新赋值 ---
        df_merged['warrant_count'] = df_merged['warrant_count'].ffill()
        print(f"    > 数据合并与填充完成。")

        # --- 步骤 5: 计算仓单因子 ---
        df_merged['W_pct'] = calculate_w_pct(df_merged['warrant_count'])
        df_merged['W_ratio'] = calculate_w_ratio(df_merged['warrant_count'])

        # --- 步骤 6: 整理输出并保存 ---
        output_cols = ['open', 'high', 'low', 'close', 'warrant_count', 'W_pct', 'W_ratio']
        df_final = df_merged[output_cols].copy()
        df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df_final_output = df_final.loc[start_date:end_date]
        
        # --- [已修正] 使用 .dropna() 并重新赋值 ---
        df_final_output = df_final_output.dropna(subset=['W_pct', 'W_ratio'], how='all')

        if df_final_output.empty:
            print(f"    - {product_name} 在指定日期范围内没有可计算的有效因子值。")
            return

        safe_product_name = sanitize_filename(product_name)
        safe_product_code = sanitize_filename(product_code)
        csv_filename = f"{safe_product_name}_{safe_product_code}_warrant_factors_ohlc.csv"
        csv_filepath = os.path.join(output_folder, csv_filename)

        df_final_output.to_csv(csv_filepath, index=True, float_format='%.4f')
        print(f"    ✅ 成功保存文件: {csv_filepath}")

    except Exception as e:
        print(f"    - 处理品种 {product_name} ({akshare_symbol}) 时发生错误: {e}")
        import traceback
        print(traceback.format_exc())

# --- 4. 主程序入口 ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"已创建输出文件夹: {OUTPUT_FOLDER}")
    
    total_products = len(PRODUCTS_TO_PROCESS)
    print("="*50)
    print(f"--- 开始处理 {total_products} 个期货品种的仓单因子计算 ---")
    print(f"--- 数据最终输出日期范围: {START_DATE_FILTER} 至 {END_DATE_FILTER} ---")
    print("="*50)
    
    for i, (p_name, p_code) in enumerate(PRODUCTS_TO_PROCESS):
        print(f"\n[处理进度: {i+1}/{total_products}]")
        process_single_future(
            product_name=p_name,
            product_code=p_code,
            start_date=START_DATE_FILTER,
            end_date=END_DATE_FILTER,
            output_folder=OUTPUT_FOLDER
        )
        # 加入短暂延时，避免过于频繁地请求接口
        if i < total_products - 1:
            time.sleep(1)

    print("\n所有任务处理完毕！")