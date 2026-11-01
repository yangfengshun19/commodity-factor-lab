import os
# -*- coding: utf-8 -*-
"""
精简测试版：仅测试 螺纹钢(rb) 的数据获取、因子计算与 MySQL 自动入库
"""

import akshare as ak
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import warnings

from sqlalchemy import create_engine, text
import pymysql

# --- 1. MySQL 数据库配置区 ---
DB_USER = os.environ.get("DB_USER", "researcher")
DB_PASS = os.environ.get("DB_PASS", "")
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = os.environ.get("DB_PORT", "3306")
DB_NAME = os.environ.get("DB_NAME", "futures_data")

# 创建连接引擎
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(DB_URL)

# --- 2. 极简测试品种 ---
PRODUCTS_BY_EXCHANGE = {
    "SHFE": [("螺纹钢", "rb")]  # 只保留螺纹钢进行测试
}

# --- 3. 测试时间范围 ---
# 为了测试速度，起始日期设为今年年初。测试成功后，你可以改回 2023-01-01
DEFAULT_START_DATE = "2025-01-01" 
END_DATE = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


# ================= 辅助函数与核心算法 (保持不变) =================

def get_latest_date_from_mysql(engine, product_code):
    try:
        with engine.connect() as conn:
            query = text("SELECT MAX(date) FROM daily_factors WHERE product_code = :code")
            result = conn.execute(query, {"code": product_code}).scalar()
            if result:
                return str(result).split(' ')[0] 
    except Exception:
        pass
    return None

def parse_futures_symbol(symbol_code, calculation_date):
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
    _, year, month = parse_futures_symbol(symbol_code, calculation_date)
    if year and month:
        try:
            return datetime(year, month, 15)
        except ValueError:
            return pd.NaT
    return pd.NaT

def time_to_maturity_years(date1, date2):
    if pd.isna(date1) or pd.isna(date2):
        return np.nan
    return (date2 - date1).days / 365.25

def nelson_siegel_price_curve(tau, beta0, beta1, beta2, lambda_val):
    lt = lambda_val * tau
    exp_lt = np.exp(-lt)
    term2_val = (1 - exp_lt) / lt
    term2_val[np.isclose(lt, 0)] = 1.0
    term3_val = term2_val - exp_lt
    return beta0 + beta1 * term2_val + beta2 * term3_val

def process_data_for_day(daily_df, calculation_date_str):
    results = {
        'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan,
        'C_frontnext': np.nan, 'C_mainsub': np.nan, 'C_nsnormal': np.nan
    }
    
    if daily_df.empty or len(daily_df) < 2:
        return pd.Series(results)
    
    calculation_date_dt = datetime.strptime(calculation_date_str, "%Y-%m-%d")

    df = daily_df.copy()
    for col in ['open', 'high', 'low', 'close', 'settle', 'open_interest']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['settle', 'open_interest'], inplace=True)
    if df.empty: return pd.Series(results)

    main_contract_row = df.loc[df['open_interest'].idxmax()]
    results['open'] = main_contract_row['open']
    results['high'] = main_contract_row['high']
    results['low'] = main_contract_row['low']
    results['close'] = main_contract_row['close']

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

    front = df.iloc[0]
    nex = df.iloc[1]
    if front['settle'] > 0 and nex['expiry_date'] > front['expiry_date']:
        time_diff = time_to_maturity_years(front['expiry_date'], nex['expiry_date'])
        if time_diff > 1e-6:
            results['C_frontnext'] = ((nex['settle'] - front['settle']) / front['settle']) / time_diff

    sorted_by_oi = df.sort_values(by='open_interest', ascending=False).reset_index(drop=True)
    main = sorted_by_oi.iloc[0]
    subs = sorted_by_oi[sorted_by_oi['expiry_date'] > main['expiry_date']].sort_values(by='expiry_date')
    if not subs.empty:
        sub = subs.iloc[0]
        if main['settle'] > 0 and sub['expiry_date'] > main['expiry_date']:
            time_diff = time_to_maturity_years(main['expiry_date'], sub['expiry_date'])
            if time_diff > 1e-6:
                results['C_mainsub'] = ((sub['settle'] - main['settle']) / main['settle']) / time_diff

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


# ================= 主程序入口 =================
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    print(f"数据处理目标结束日期: {END_DATE}")
    end_date_yyyymmdd = END_DATE.replace('-', '')

    for exchange, products in PRODUCTS_BY_EXCHANGE.items():
        for product_name, product_code in products:
            print(f"\n--- 正在处理测试品种: {product_name} ({product_code}) ---")
            
            latest_date_str = get_latest_date_from_mysql(engine, product_code)
            
            if latest_date_str:
                latest_date_dt = datetime.strptime(latest_date_str, '%Y-%m-%d')
                start_date_dt = latest_date_dt + timedelta(days=1)
                start_date_yyyymmdd = start_date_dt.strftime('%Y%m%d')
                print(f"[*] 数据库中已有记录，最新日期为 {latest_date_str}。将从 {start_date_dt.strftime('%Y-%m-%d')} 开始增量更新。")
            else:
                start_date_yyyymmdd = DEFAULT_START_DATE.replace('-', '')
                print(f"[*] 数据库中暂无记录，将从测试起点 {DEFAULT_START_DATE} 开始全量拉取。")

            if start_date_yyyymmdd > end_date_yyyymmdd:
                print(f"[*] {product_name} 的数据已是最新，测试完成！")
                continue

            try:
                print("[*] 正在从 AkShare 拉取数据，请稍候...")
                product_df = ak.get_futures_daily(
                    start_date=start_date_yyyymmdd,
                    end_date=end_date_yyyymmdd,
                    market=exchange
                )
                
                if product_df.empty:
                    print(f"[!] 未获取到任何行情数据。")
                    continue
                    
                product_df = product_df[
                    product_df['symbol'].str.lower().str.startswith(product_code.lower(), na=False)
                ].copy()

                if product_df.empty:
                    print(f"[!] 未找到螺纹钢的合约数据。")
                    continue
                
                print("[*] 数据拉取成功，正在计算因子...")
                product_df['date'] = pd.to_datetime(product_df['date'])
                
                final_df = product_df.groupby(product_df['date'].dt.strftime('%Y-%m-%d')).apply(
                    lambda x: process_data_for_day(x, x.name)
                )

                if final_df.empty:
                    print(f"[!] 计算后未产生有效结果。")
                    continue

                final_df.index.name = 'date'
                final_df.reset_index(inplace=True)
                final_df['product_name'] = product_name
                final_df['product_code'] = product_code
                
                output_columns = [
                    'date', 'product_name', 'product_code', 
                    'open', 'high', 'low', 'close', 
                    'C_frontnext', 'C_mainsub', 'C_nsnormal'
                ]
                final_df = final_df[output_columns]
                final_df['date'] = pd.to_datetime(final_df['date'])

                print("[*] 因子计算完毕，正在写入 MySQL 数据库...")
                final_df.to_sql('daily_factors', con=engine, if_exists='append', index=False)
                
                print(f"[*] 恭喜！成功将 {len(final_df)} 条数据写入数据库。")

            except Exception as e:
                print(f"[X] 发生错误: {e}")
                import traceback
                print(traceback.format_exc())

    print("\n测试流程结束。")