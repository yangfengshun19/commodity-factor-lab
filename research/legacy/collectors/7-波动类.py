# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:33:02 2025

@author: AKIRA
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import time
import re

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
RERUN_MODE = False
RERUN_PRODUCTS_LIST = [("螺纹钢", "rb"), ("碳酸锂", "LC")]
OUTPUT_FOLDER = "期货波动率因子_CSV输出"
START_DATE_FILTER = "2023-01-01"
END_DATE_FILTER = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
WINDOW_FOR_STD = 20
ANNUALIZE_VOLATILITY = False

class FuturesVolatilityFactors:
    def __init__(self, df_ohlc):
        if not all(col in df_ohlc.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("DataFrame 必须包含 'open', 'high', 'low', 'close' 列")
        self.df = df_ohlc.copy()
        for col in ['open', 'high', 'low', 'close']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    def _ensure_log_returns(self):
        if 'log_return' not in self.df.columns:
            if 'close' not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df['close']):
                self.df['log_return'] = np.nan
                return
            close_shifted = self.df['close'].shift(1)
            mask = (self.df['close'] > 0) & (close_shifted > 0)
            self.df['log_return'] = np.nan
            self.df.loc[mask, 'log_return'] = np.log(self.df['close'][mask] / close_shifted[mask])
            
            # ##################################################
            # ###      !!! 这里是修正警告的地方 !!!      ###
            # ##################################################
            # 使用建议的写法，避免FutureWarning
            self.df['log_return'] = self.df['log_return'].fillna(0)

    def calculate_vol_std(self, window=20, annualize=False, trading_days_per_year=252):
        self._ensure_log_returns()
        if 'log_return' not in self.df.columns or self.df['log_return'].isnull().all():
            return pd.Series([np.nan] * len(self.df), index=self.df.index)
        rolling_vol = self.df['log_return'].rolling(window=window, min_periods=max(1, int(window * 0.8))).std()
        expanding_vol = self.df['log_return'].expanding(min_periods=1).std(ddof=0)
        vol_std = rolling_vol.combine_first(expanding_vol)
        if annualize:
            vol_std = vol_std * np.sqrt(trading_days_per_year)
        return vol_std

    def _safe_log(self, series_num, series_den):
        valid_den = (series_den > 0)
        valid_num = (series_num > 0)
        ratio = pd.Series(np.nan, index=series_num.index)
        mask = valid_num & valid_den
        ratio[mask] = series_num[mask] / series_den[mask]
        log_ratio = pd.Series(np.nan, index=ratio.index)
        log_ratio[ratio > 0] = np.log(ratio[ratio > 0])
        log_ratio.fillna(0, inplace=True)
        return log_ratio

    def calculate_vol_gk(self, annualize=False, trading_days_per_year=252):
        log_hl = self._safe_log(self.df['high'], self.df['low'])
        log_co = self._safe_log(self.df['close'], self.df['open'])
        sigma_gk_sq = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
        sigma_gk_sq = np.maximum(sigma_gk_sq, 0)
        vol_gk = np.sqrt(sigma_gk_sq)
        if annualize:
            vol_gk = vol_gk * np.sqrt(trading_days_per_year)
        return vol_gk

    def calculate_vol_rs(self, annualize=False, trading_days_per_year=252):
        log_hc = self._safe_log(self.df['high'], self.df['close'])
        log_ho = self._safe_log(self.df['high'], self.df['open'])
        log_lc = self._safe_log(self.df['low'], self.df['close'])
        log_lo = self._safe_log(self.df['low'], self.df['open'])
        sigma_rs_sq = log_hc * log_ho + log_lc * log_lo
        sigma_rs_sq = np.maximum(sigma_rs_sq, 0)
        vol_rs = np.sqrt(sigma_rs_sq)
        if annualize:
            vol_rs = vol_rs * np.sqrt(trading_days_per_year)
        return vol_rs

    def calculate_all_factors(self, window_std=20, annualize=False, trading_days_per_year=252):
        results_df = self.df.copy()
        results_df['vol_std'] = self.calculate_vol_std(window=window_std, annualize=annualize, trading_days_per_year=trading_days_per_year)
        results_df['vol_gk'] = self.calculate_vol_gk(annualize=annualize, trading_days_per_year=trading_days_per_year)
        results_df['vol_rs'] = self.calculate_vol_rs(annualize=annualize, trading_days_per_year=trading_days_per_year)
        return results_df

def sanitize_filename(filename):
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized

def process_single_future(product_name, product_code_short, start_date, end_date, output_folder):
    akshare_symbol = product_code_short.upper() + "0"
    print(f"\n正在处理品种: {product_name} ({product_code_short}), Akshare代码: {akshare_symbol}")
    try:
        df_futures_ohlc = ak.futures_zh_daily_sina(symbol=akshare_symbol)
        if df_futures_ohlc.empty:
            print(f"未能获取到 {product_name} ({akshare_symbol}) 的数据。")
            return
        print(f"获取到 {product_name} 原始数据 {df_futures_ohlc.shape[0]} 条。")
        df_futures_ohlc.rename(columns={'日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close'}, inplace=True)
        required_cols = ['date', 'open', 'high', 'low', 'close']
        if not all(col in df_futures_ohlc.columns for col in required_cols):
            print(f"错误: {product_name} 数据中缺少必要的OHLC列。跳过。")
            return
        df_processed = df_futures_ohlc[required_cols].copy()
        df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        df_processed.dropna(subset=['date'], inplace=True)
        df_processed.set_index('date', inplace=True)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        df_processed.dropna(subset=price_cols, how='any', inplace=True)
        df_processed.sort_index(inplace=True)
        warm_up_days = 40 
        calculation_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=warm_up_days)).strftime('%Y-%m-%d')
        df_for_calculation = df_processed.loc[calculation_start_date:end_date]
        if df_for_calculation.empty:
            print(f"{product_name} 在包含预热期的日期范围 {calculation_start_date} 到 {end_date} 内无数据。")
            return
        print(f"{product_name} 用于计算的数据 (含预热期) 共 {df_for_calculation.shape[0]} 条。")
        factor_calculator = FuturesVolatilityFactors(df_for_calculation)
        df_with_factors_extended = factor_calculator.calculate_all_factors(window_std=WINDOW_FOR_STD, annualize=ANNUALIZE_VOLATILITY)
        df_final_output = df_with_factors_extended.loc[start_date:end_date]
        if df_final_output.empty:
             print(f"{product_name} 在指定的最终输出日期范围 {start_date} 到 {end_date} 内无数据。")
             return
        safe_product_name = sanitize_filename(product_name)
        safe_product_code = sanitize_filename(product_code_short)
        csv_filename = f"{safe_product_name}_{safe_product_code}_vol_factors.csv"
        csv_filepath = os.path.join(output_folder, csv_filename)
        df_final_output.to_csv(csv_filepath, index=True, float_format='%.6f')
        print(f"成功为 {product_name} 计算因子并保存到: {csv_filepath}")
    except Exception as e:
        print(f"处理品种 {product_name} ({akshare_symbol}) 时发生错误: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"已创建输出文件夹: {OUTPUT_FOLDER}")
    if RERUN_MODE:
        products_to_run = RERUN_PRODUCTS_LIST
    else:
        products_to_run = PRODUCTS_TO_PROCESS
    mode_message = "【单独重新获取】模式" if RERUN_MODE else "【全量处理】模式"
    print("="*40)
    print(f"!!! 启动 {mode_message} !!!")
    print(f"--- 将处理 {len(products_to_run)} 个期货品种。")
    print("="*40)
    total_products = len(products_to_run)
    print(f"数据最终输出日期范围: {START_DATE_FILTER} 至 {END_DATE_FILTER}")
    for i, (product_name, product_code_short) in enumerate(products_to_run):
        print(f"\n--- [处理进度: {i+1}/{total_products}] ---")
        process_single_future(product_name, product_code_short, START_DATE_FILTER, END_DATE_FILTER, OUTPUT_FOLDER)
        if i < total_products - 1:
            time.sleep(1) 
    print("\n所有指定品种处理完毕。")
    print(f"请在文件夹 '{OUTPUT_FOLDER}' 中查看输出的CSV文件。")