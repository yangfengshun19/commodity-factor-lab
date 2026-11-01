import os
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:16:44 2026

@author: AKIRA
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:21:55 2025
Updated for MySQL Incremental Update.

Description:
This script retrieves inventory (warrant) data from Eastmoney and OHLC from Sina,
calculates warrant-based factors, and stores them in a local MySQL Database.
Includes warm-up period logic for rolling calculations.
"""
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import time
from datetime import datetime, timedelta
import akshare as ak
import warnings

# --- 新增：MySQL 数据库所需库 ---
from sqlalchemy import create_engine, text
import pymysql

warnings.filterwarnings('ignore')

# ================= 1. MySQL 数据库配置区 =================

DB_USER = os.environ.get("DB_USER", "researcher")
DB_PASS = os.environ.get("DB_PASS", "")
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = os.environ.get("DB_PORT", "3306")
DB_NAME = os.environ.get("DB_NAME", "futures_data")
TABLE_NAME = 'warrant_factors' # 仓单因子存入单独的表

# --- 自动建库逻辑 ---
try:
    temp_conn = pymysql.connect(host=DB_HOST, port=int(DB_PORT), user=DB_USER, password=DB_PASS)
    temp_cursor = temp_conn.cursor()
    temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    temp_cursor.close()
    temp_conn.close()
except Exception as e:
    print(f"尝试自动创建数据库时发生错误: {e}")

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(DB_URL)

# ================= 2. 品种与时间配置区 =================

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

NON_PHYSICAL_PRODUCTS = {"IF", "IH", "IC", "IM", "TS", "TF", "T", "TL", "ec"}
LOOKBACK_PERIOD_EM = 50 
DEFAULT_START_DATE = "2023-01-01"
END_DATE_FILTER = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


# ================= 3. 因子计算与辅助函数 =================

def calculate_w_pct(series_warrant_count, window=LOOKBACK_PERIOD_EM):
    return series_warrant_count.rolling(window=window, min_periods=max(1, int(window * 0.8))).apply(
        lambda x: percentileofscore(x[~np.isnan(x)], x[-1], kind='rank') if pd.notna(x[-1]) and np.count_nonzero(~np.isnan(x)) > 1 else np.nan,
        raw=True
    ) / 100.0

def calculate_w_ratio(series_warrant_count, window=LOOKBACK_PERIOD_EM):
    rolling_mean = series_warrant_count.rolling(window=window, min_periods=max(1, int(window * 0.8))).mean()
    return series_warrant_count / rolling_mean

def get_latest_date_from_mysql(db_engine, table_name, product_code):
    try:
        with db_engine.connect() as conn:
            query = text(f"SELECT MAX(date) FROM {table_name} WHERE product_code = :code")
            result = conn.execute(query, {"code": product_code}).scalar()
            if result:
                return str(result).split(' ')[0] 
    except Exception:
        pass
    return None


# ================= 4. 主处理函数 =================

def process_single_future(product_name, product_code, target_start_date, end_date):
    
    if product_code in NON_PHYSICAL_PRODUCTS:
        print(f"\n[*] 跳过非实物商品: {product_name} ({product_code})，无仓单数据。")
        return True

    target_start_dt = pd.to_datetime(target_start_date)
    end_dt = pd.to_datetime(end_date)
    
    if target_start_dt > end_dt:
         print(f"[*] {product_name} 的数据已是最新，无需更新。")
         return True

    akshare_symbol = product_code.upper() + "0"
    print(f"\n[*] 正在处理仓单品种: {product_name} ({product_code})，计算区间: {target_start_date} 至 {end_date}")

    try:
        # --- 步骤 1: 获取OHLC数据 ---
        df_ohlc = ak.futures_zh_daily_sina(symbol=akshare_symbol)
        if df_ohlc.empty:
            print(f"    [!] 未能获取到 {product_name} 的OHLC数据。")
            return False
        
        df_ohlc.rename(columns={'日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close'}, inplace=True)
        df_ohlc_processed = df_ohlc[['date', 'open', 'high', 'low', 'close']].copy()
        df_ohlc_processed['date'] = pd.to_datetime(df_ohlc_processed['date'], errors='coerce')
        df_ohlc_processed.dropna(subset=['date'], inplace=True)
        df_ohlc_processed.set_index('date', inplace=True)
        df_ohlc_processed.sort_index(inplace=True)

        # --- 步骤 2: 获取仓单数据 ---
        api_symbol = SYMBOL_MAP_EM.get(product_name, product_name)
        df_warrant_raw = ak.futures_inventory_em(symbol=api_symbol)
        
        if df_warrant_raw.empty:
            print(f"    [!] 未能获取到 {product_name} ('{api_symbol}') 的东方财富仓单数据。")
            return False

        df_warrant_raw.rename(columns={'日期': 'date', '库存': 'warrant_count'}, inplace=True)
        df_warrant_processed = df_warrant_raw[['date', 'warrant_count']].copy()
        df_warrant_processed['date'] = pd.to_datetime(df_warrant_processed['date'], errors='coerce')
        df_warrant_processed.dropna(subset=['date'], inplace=True)
        df_warrant_processed['warrant_count'] = pd.to_numeric(df_warrant_processed['warrant_count'], errors='coerce')
        df_warrant_processed.set_index('date', inplace=True)
        df_warrant_processed.sort_index(inplace=True)

        # --- 步骤 3: 确定包含预热期的数据范围 ---
        warm_up_days = int(LOOKBACK_PERIOD_EM * 1.5)
        calculation_start_date = (target_start_dt - pd.Timedelta(days=warm_up_days)).strftime('%Y-%m-%d')
        
        df_ohlc_final = df_ohlc_processed.loc[calculation_start_date:end_date]

        # --- 步骤 4: 合并数据与计算因子 ---
        df_merged = df_ohlc_final.join(df_warrant_processed, how='left')
        df_merged['warrant_count'] = df_merged['warrant_count'].ffill()

        df_merged['w_pct'] = calculate_w_pct(df_merged['warrant_count'])
        df_merged['w_ratio'] = calculate_w_ratio(df_merged['warrant_count'])

        # --- 步骤 5: 截取目标日期并入库 ---
        output_cols = ['open', 'high', 'low', 'close', 'warrant_count', 'w_pct', 'w_ratio']
        df_final = df_merged[output_cols].copy()
        df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 截取真正的目标时间段
        df_final_output = df_final.loc[target_start_date:end_date].copy()
        df_final_output = df_final_output.dropna(subset=['w_pct', 'w_ratio'], how='all')

        if df_final_output.empty:
            print(f"    [!] {product_name} 过滤后无有效数据写入。")
            return False

        # 整理成数据库标准格式
        df_final_output.reset_index(inplace=True)
        df_final_output['product_name'] = product_name
        df_final_output['product_code'] = product_code
        
        # 调整列顺序
        db_cols = ['date', 'product_name', 'product_code', 'open', 'high', 'low', 'close', 'warrant_count', 'w_pct', 'w_ratio']
        df_final_output = df_final_output[db_cols]
        df_final_output['date'] = pd.to_datetime(df_final_output['date'])

        # 写入 MySQL
        df_final_output.to_sql(TABLE_NAME, con=engine, if_exists='append', index=False)
        print(f"    [√] 成功将 {len(df_final_output)} 条仓单因子数据追加到 MySQL ({TABLE_NAME})！")
        return True

    except Exception as e:
        print(f"    [X] 处理品种 {product_name} 时发生错误: {e}")
        return False

# ================= 5. 主程序入口 =================

if __name__ == "__main__":
    total_products = len(PRODUCTS_TO_PROCESS)
    print("="*60)
    print(f"开始处理 {total_products} 个期货品种的【仓单类因子】...")
    print(f"目标更新结束日期: {END_DATE_FILTER}")
    print("="*60)
    
    for i, (p_name, p_code) in enumerate(PRODUCTS_TO_PROCESS):
        
        latest_date_str = get_latest_date_from_mysql(engine, TABLE_NAME, p_code)
        
        if latest_date_str:
            latest_date_dt = datetime.strptime(latest_date_str, '%Y-%m-%d')
            target_start_dt = latest_date_dt + timedelta(days=1)
            target_start_date = target_start_dt.strftime('%Y-%m-%d')
            if target_start_date <= END_DATE_FILTER:
                print(f"\n--- [进度: {i+1}/{total_products}] 数据库已有记录至 {latest_date_str} ---")
        else:
            target_start_date = DEFAULT_START_DATE
            print(f"\n--- [进度: {i+1}/{total_products}] 数据库暂无记录，全量拉取 ---")

        process_single_future(
            product_name=p_name,
            product_code=p_code,
            target_start_date=target_start_date,
            end_date=END_DATE_FILTER
        )
        
        if i < total_products - 1:
            time.sleep(1)

    print("\n\n========================================================")
    print(f"所有品种的仓单因子处理完毕！请在 DBeaver 中查看 {TABLE_NAME} 表。")
    print("========================================================")