# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 22:30:00 2025
@author: Gemini Assistant (Revised)
Description: 批量计算多个期货品种的价值类因子 (Val_halfyear, Val_oneyear)。
             修改为使用 ak.futures_zh_daily_sina 获取OHLC数据，并基于收盘价计算。
"""
import akshare as ak
import pandas as pd
import os
import time
import re

# --- 1. 配置区 ---

# --- 配置区 ---
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
    ("沪深300指数", "IF"), ("上证50指数", "IH"), ("中证500指数", "IC"), ("中证1000指数", "IM"),
    ("2年期国债", "TS"), ("5年期国债", "TF"), ("10年期国债", "T"), ("30年期国债", "TL")
]

# 输出文件夹
OUTPUT_FOLDER = "期货价值类因子_按品种输出_OHLC"

# --- 时间设置 ---
# 为了计算2023-01-01的一年期因子，数据获取需要从2022-01-01开始
# 注意：akshare接口会自动获取尽可能多的历史数据，这里的START_DATE主要用于最终结果的筛选
START_DATE = "2022-01-01"
# 最终输出结果的起始日期
CALCULATION_START_DATE = "2023-01-01"
# 获取当前日期作为结束日期
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')


# --- 2. 辅助函数 ---
def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized

def process_single_future(product_name, product_code, start_date, end_date, output_folder):
    """
    处理单个期货品种：获取数据、计算因子、保存文件。
    """
    # 拼接akshare所需的symbol，如 'rb' -> 'RB0'
    akshare_symbol = product_code.upper() + "0"
    print(f"\n--- 正在处理品种: {product_name} ({product_code}), Akshare代码: {akshare_symbol} ---")

    try:
        # --- 步骤 1: 获取单个品种的完整日线OHLC数据 ---
        df_ohlc = ak.futures_zh_daily_sina(symbol=akshare_symbol)

        if df_ohlc.empty:
            print(f"    - 未能获取到 {product_name} ({akshare_symbol}) 的数据。")
            return

        print(f"    > 获取到 {product_name} 原始数据 {df_ohlc.shape[0]} 条。")

        # --- 步骤 2: 数据预处理 ---
        # 重命名列以匹配标准格式
        df_ohlc.rename(columns={'日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close'}, inplace=True)
        
        # 筛选需要的列
        required_cols = ['date', 'open', 'high', 'low', 'close']
        if not all(col in df_ohlc.columns for col in required_cols):
            print(f"    - 错误: {product_name} 数据中缺少必要的OHLC列。跳过。")
            return
        
        df_processed = df_ohlc[required_cols].copy()
        
        # 转换数据类型
        df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        df_processed.dropna(subset=['date'], inplace=True)
        df_processed.set_index('date', inplace=True)
        
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        df_processed.dropna(subset=price_cols, how='any', inplace=True)
        df_processed.sort_index(inplace=True)
        
        if df_processed.empty:
            print(f"    - {product_name} 数据预处理后为空。跳过。")
            return

        # --- 步骤 3: 计算价值因子 ---
        # 使用交易日进行shift，更准确。252约等于1年，126约等于半年。
        df_processed['close_half_year_ago'] = df_processed['close'].shift(126)
        df_processed['close_one_year_ago'] = df_processed['close'].shift(252)

        # 计算因子
        df_processed['Val_halfyear'] = (df_processed['close'] / df_processed['close_half_year_ago']) - 1
        df_processed['Val_oneyear'] = (df_processed['close'] / df_processed['close_one_year_ago']) - 1

        # --- 步骤 4: 整理输出并保存 ---
        # 筛选最终需要的列
        output_cols = ['open', 'high', 'low', 'close', 'Val_halfyear', 'Val_oneyear']
        output_df = df_processed[output_cols].copy()
        
        # 过滤最终结果的日期
        output_df = output_df.loc[start_date:end_date]
        
        # 删除因子全为空的行
        output_df.dropna(how='all', subset=['Val_halfyear', 'Val_oneyear'], inplace=True)

        if output_df.empty:
            print(f"    - {product_name} 在 {start_date} 之后没有足够的历史数据来计算因子。")
            return

        # 保存文件
        safe_product_name = sanitize_filename(product_name)
        safe_product_code = sanitize_filename(product_code)
        csv_filename = f"{safe_product_name}_{safe_product_code}_value_factors_ohlc.csv"
        csv_filepath = os.path.join(output_folder, csv_filename)

        output_df.to_csv(csv_filepath, index=True, float_format='%.6f')
        print(f"    ✅ 成功保存文件: {csv_filepath}")

    except Exception as e:
        print(f"    - 处理品种 {product_name} ({akshare_symbol}) 时发生错误: {e}")


# --- 3. 主程序入口 ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"已创建输出文件夹: {OUTPUT_FOLDER}")
    
    total_products = len(PRODUCTS_TO_PROCESS)
    print("="*50)
    print(f"--- 开始处理 {total_products} 个期货品种的价值因子计算 ---")
    print(f"--- 数据最终输出日期范围: {CALCULATION_START_DATE} 至 {END_DATE} ---")
    print("="*50)

    for i, (p_name, p_code) in enumerate(PRODUCTS_TO_PROCESS):
        print(f"\n[处理进度: {i+1}/{total_products}]")
        process_single_future(
            product_name=p_name,
            product_code=p_code,
            start_date=CALCULATION_START_DATE,
            end_date=END_DATE,
            output_folder=OUTPUT_FOLDER
        )
        # 加入短暂延时，避免过于频繁地请求接口
        if i < total_products - 1:
            time.sleep(1)

    print("\n所有任务处理完毕！")