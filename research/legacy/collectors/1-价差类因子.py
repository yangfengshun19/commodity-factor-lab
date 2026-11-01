# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:00:00 2025
Description:
This script calculates three types of spread-based factors and retrieves the OHLC prices
of the main contract for all Chinese futures products.

Factors Calculated:
1. C_frontnext: Annualized spread ratio between the front-month and next-month contracts.
2. C_mainsub: Annualized spread ratio between the main and sub-main contracts.
3. C_nsnormal: The curvature parameter (beta2) from the Nelson-Siegel model fit.

Output CSV Columns:
- date, open, high, low, close, C_frontnext, C_mainsub, C_nsnormal
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import re
import time
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import warnings

# --- 配置区 ---

# 1. 定义要处理的期货品种列表和其所属交易所
PRODUCTS_BY_EXCHANGE = {
    "SHFE": [
        ("螺纹钢", "rb"), ("热轧卷板", "hc"), ("石油沥青", "bu"), ("天然橡胶", "ru"),
        ("燃料油", "fu"), ("纸浆", "sp"), ("铜", "cu"), ("铝", "al"), ("铅", "pb"),
        ("锌", "zn"), ("锡", "sn"), ("镍", "ni"), ("不锈钢", "ss"), ("黄金", "au"),
        ("白银", "ag"), ("线材", "wr"), ("氧化铝", "ao"), ("合成橡胶", "br")
    ],
    "INE": [
        ("原油", "sc"), ("20号胶", "nr"), ("低硫燃料油", "lu"), ("国际铜", "bc"), ("集运指数(欧线)", "ec")
    ],
    "DCE": [
        ("黄大豆1号", "a"), ("黄大豆2号", "b"), ("黄玉米", "c"), ("玉米淀粉", "cs"),
        ("豆粕", "m"), ("豆油", "y"), ("棕榈油", "p"), ("铁矿石", "i"), ("焦炭", "j"),
        ("焦煤", "jm"), ("聚乙烯", "l"), ("聚氯乙烯", "v"), ("聚丙烯", "pp"),
        ("乙二醇", "eg"), ("粳米", "rr"), ("苯乙烯", "eb"), ("液化石油气", "pg"),
        ("鸡蛋", "jd"), ("纤维板", "fb"), ("胶合板", "bb"), ("生猪", "lh")
    ],
    "CZCE": [
        ("一号棉花", "CF"), ("棉纱", "CY"), ("白砂糖", "SR"), ("菜粕", "RM"),
        ("菜籽油", "OI"), ("精对苯二甲酸", "TA"), ("甲醇", "MA"), ("玻璃", "FG"),
        ("动力煤", "ZC"), ("纯碱", "SA"), ("烧碱", "SH"), ("短纤", "PF"),
        ("瓶片", "PR"), ("硅铁", "SF"), ("锰硅", "SM"), ("苹果", "AP"),
        ("红枣", "CJ"), ("尿素", "UR"), ("花生", "PK"), ("对二甲苯", "PX"),
        ("菜籽", "RS"), ("普通小麦", "PM"), ("强麦", "WH"), ("早籼稻", "RI"),
        ("晚籼稻", "LR"), ("粳稻", "JR")
    ],
    "GFEX": [
        ("工业硅", "SI"), ("碳酸锂", "LC"), ("多晶硅", "PS")
    ]
}

# 2. 输出文件夹名称
OUTPUT_FOLDER = "期货价差因子_输出"

# 3. 数据时间范围
START_DATE = "2023-01-01"
# 自动获取当前日期并设置为一天前
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


# --- 辅助函数 ---

def sanitize_filename(filename):
    """移除或替换文件名中的非法字符"""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def parse_futures_symbol(symbol_code, calculation_date):
    """从合约代码中解析出品种、年份和月份"""
    try:
        underlying = ''.join(filter(str.isalpha, symbol_code))
        digits = ''.join(filter(str.isdigit, symbol_code))
        if len(digits) == 4:
            year = int("20" + digits[:2])
            month = int(digits[2:])
            return underlying, year, month
        elif len(digits) == 3:
            contract_y_digit = int(digits[0])
            calc_year = calculation_date.year
            year_prefix = calc_year // 10
            year_guess = year_prefix * 10 + contract_y_digit
            if year_guess < calc_year - 1:
                year_guess += 10
            year = year_guess
            month = int(digits[1:])
            return underlying, year, month
    except Exception:
        return None, None, None
    return None, None, None

def get_approx_expiry_date(symbol_code, calculation_date):
    """根据合约代码和计算日期估算近似的到期日"""
    _, year, month = parse_futures_symbol(symbol_code, calculation_date)
    if year and month:
        try:
            return datetime(year, month, 15)
        except ValueError:
            return pd.NaT
    return pd.NaT

def time_to_maturity_years(date1, date2):
    """计算两个日期之间以年为单位的时间差"""
    if pd.isna(date1) or pd.isna(date2):
        return np.nan
    return (date2 - date1).days / 365.25

def nelson_siegel_price_curve(tau, beta0, beta1, beta2, lambda_val):
    """Nelson-Siegel曲线函数"""
    lt = lambda_val * tau
    exp_lt = np.exp(-lt)
    term2_val = (1 - exp_lt) / lt
    term2_val[np.isclose(lt, 0)] = 1.0
    term3_val = term2_val - exp_lt
    return beta0 + beta1 * term2_val + beta2 * term3_val

# --- 核心计算函数 ---

def process_data_for_day(daily_df, calculation_date_str):
    """
    为单个品种在单个交易日的数据计算因子并提取主力合约OHLC。
    """
    # 初始化返回结果
    results = {
        'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan,
        'C_frontnext': np.nan, 'C_mainsub': np.nan, 'C_nsnormal': np.nan
    }
    
    if daily_df.empty or len(daily_df) < 2:
        return pd.Series(results)
    
    calculation_date_dt = datetime.strptime(calculation_date_str, "%Y-%m-%d")

    # 1. 数据预处理
    df = daily_df.copy()
    for col in ['open', 'high', 'low', 'close', 'settle', 'open_interest']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['settle', 'open_interest'], inplace=True)
    if df.empty: return pd.Series(results)

    # --- 【新增】提取主力合约OHLC ---
    # 根据当日持仓量找到主力合约
    main_contract_row = df.loc[df['open_interest'].idxmax()]
    results['open'] = main_contract_row['open']
    results['high'] = main_contract_row['high']
    results['low'] = main_contract_row['low']
    results['close'] = main_contract_row['close']
    # --- OHLC提取结束 ---

    df['expiry_date'] = df['symbol'].apply(lambda x: get_approx_expiry_date(x, calculation_date_dt))
    df.dropna(subset=['expiry_date'], inplace=True)
    if df.empty: return pd.Series(results)

    df['time_to_maturity'] = df.apply(
        lambda row: time_to_maturity_years(calculation_date_dt, row['expiry_date']), axis=1
    )
    df = df[df['time_to_maturity'] > 0.01].copy()
    df.sort_values(by='expiry_date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 2:
        return pd.Series(results)

    # 2. 计算 C_frontnext
    front = df.iloc[0]
    nex = df.iloc[1]
    if front['settle'] > 0 and nex['expiry_date'] > front['expiry_date']:
        time_diff = time_to_maturity_years(front['expiry_date'], nex['expiry_date'])
        if time_diff > 1e-6:
            results['C_frontnext'] = ((nex['settle'] - front['settle']) / front['settle']) / time_diff

    # 3. 计算 C_mainsub
    sorted_by_oi = df.sort_values(by='open_interest', ascending=False).reset_index(drop=True)
    main = sorted_by_oi.iloc[0]
    subs = sorted_by_oi[sorted_by_oi['expiry_date'] > main['expiry_date']].sort_values(by='expiry_date')
    if not subs.empty:
        sub = subs.iloc[0]
        if main['settle'] > 0 and sub['expiry_date'] > main['expiry_date']:
            time_diff = time_to_maturity_years(main['expiry_date'], sub['expiry_date'])
            if time_diff > 1e-6:
                results['C_mainsub'] = ((sub['settle'] - main['settle']) / main['settle']) / time_diff

    # 4. 计算 C_nsnormal
    ns_df = df[(df['time_to_maturity'] > 0.1) & (df['time_to_maturity'] < 2.0) & (df['settle'] > 0)].copy()
    if len(ns_df) >= 4:
        tau = ns_df['time_to_maturity'].values
        log_prices = np.log(ns_df['settle'].values)
        try:
            p0 = [np.mean(log_prices), log_prices[0] - log_prices[-1], 0.0, 1.0]
            bounds = ([-np.inf, -np.inf, -np.inf, 0.001], [np.inf, np.inf, np.inf, 10.0])
            params, _ = curve_fit(nelson_siegel_price_curve, tau, log_prices, p0=p0, bounds=bounds, maxfev=5000)
            results['C_nsnormal'] = params[2]
        except (RuntimeError, ValueError):
            pass
            
    return pd.Series(results)

# --- 主程序入口 ---
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"已创建输出文件夹: {OUTPUT_FOLDER}")

    start_date_yyyymmdd = START_DATE.replace('-', '')
    end_date_yyyymmdd = END_DATE.replace('-', '')
    
    print(f"数据处理时间范围: {START_DATE} to {END_DATE}")

    # 按交易所循环
    for exchange, products in PRODUCTS_BY_EXCHANGE.items():
        print(f"\n{'='*50}")
        print(f"正在处理交易所: {exchange}")
        print(f"{'='*50}")
        
        try:
            print(f"正在从 AkShare 获取 {exchange} 的所有合约日行情数据...")
            market_data_df = ak.get_futures_daily(
                start_date=start_date_yyyymmdd,
                end_date=end_date_yyyymmdd,
                market=exchange
            )
            print(f"成功获取 {exchange} 数据 {market_data_df.shape[0]} 条。")
            
            if market_data_df.empty:
                print(f"警告: 在指定日期范围内未获取到 {exchange} 的任何数据，跳过该交易所。")
                continue
                
            for product_name, product_code in products:
                safe_product_name = sanitize_filename(product_name)
                safe_product_code = sanitize_filename(product_code)
                csv_filename = f"{safe_product_name}_{safe_product_code}_factors_with_ohlc.csv"
                csv_filepath = os.path.join(OUTPUT_FOLDER, csv_filename)
                
                print(f"\n--- 正在处理品种: {product_name} ({product_code}) ---")
                
                product_df = market_data_df[
                    market_data_df['symbol'].str.lower().str.startswith(product_code.lower(), na=False)
                ].copy()

                if product_df.empty:
                    print(f"未找到 {product_name} 的合约数据，跳过。")
                    continue
                
                print("按日计算因子并提取主力合约OHLC...")
                product_df['date'] = pd.to_datetime(product_df['date'])
                
                # 【修改】使用新的处理函数，它会同时返回OHLC和因子
                final_df = product_df.groupby(product_df['date'].dt.strftime('%Y-%m-%d')).apply(
                    lambda x: process_data_for_day(x, x.name)
                )

                if final_df.empty:
                    print(f"为 {product_name} 计算后未产生任何结果。")
                    continue

                final_df.index.name = 'date'
                # 【修改】按要求调整列的顺序
                output_columns = [
                    'open', 'high', 'low', 'close', 
                    'C_frontnext', 'C_mainsub', 'C_nsnormal'
                ]
                final_df = final_df[output_columns]

                final_df.to_csv(csv_filepath, float_format='%.6f')
                print(f"成功为 {product_name} 计算因子并保存到: {csv_filepath}")
            
            print("暂停2秒...")
            time.sleep(2)

        except Exception as e:
            print(f"处理交易所 {exchange} 时发生严重错误: {e}")
            import traceback
            print(traceback.format_exc())

    print("\n所有品种处理完毕。")
    print(f"请在文件夹 '{OUTPUT_FOLDER}' 中查看输出的CSV文件。")