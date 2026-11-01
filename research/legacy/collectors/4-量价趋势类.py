# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:43:05 2025
Updated by Gemini Assistant on Sat Oct 11 2025 to include a warm-up period for calculations.

@author: AKIRA (Original), Gemini Assistant (Modifications)
"""
import akshare as ak
import pandas as pd
import numpy as np
import os
import time
import re

# --- 1. 配置区 ---
# 定义要处理的期货品种列表 (品种名称, 品种代码)
PRODUCTS_TO_PROCESS = [
    # 上海期货交易所 (SHFE)
    ("螺纹钢", "rb"), ("热轧卷板", "hc"), ("石油沥青", "bu"), ("天然橡胶", "ru"), ("合成橡胶", "br"),
    ("燃料油", "fu"), ("纸浆", "sp"), ("铜", "cu"), ("铝", "al"), ("氧化铝", "ao"),
    ("铅", "pb"), ("锌", "zn"), ("锡", "sn"), ("镍", "ni"), ("不锈钢", "ss"),
    ("黄金", "au"), ("白银", "ag"), ("线材", "wr"),
    # 上海国际能源交易中心 (INE)
    ("20号胶", "nr"), ("低硫燃料油", "lu"), ("国际铜", "bc"), ("原油", "sc"), ("集运指数(欧线)", "ec"),
    # 大连商品交易所 (DCE)
    ("黄大豆1号", "a"), ("黄大豆2号", "b"), ("黄玉米", "c"), ("玉米淀粉", "cs"), ("豆粕", "m"),
    ("豆油", "y"), ("棕榈油", "p"), ("铁矿石", "i"), ("焦炭", "j"), ("焦煤", "jm"),
    ("聚乙烯", "l"), ("聚氯乙烯", "v"), ("聚丙烯", "pp"), ("乙二醇", "eg"), ("粳米", "rr"),
    ("苯乙烯", "eb"), ("液化石油气", "pg"), ("鸡蛋", "jd"), ("纤维板", "fb"), ("胶合板", "bb"),
    ("生猪", "lh"),
    # 郑州商品交易所 (CZCE)
    ("菜粕", "RM"), ("菜籽油", "OI"), ("一号棉花", "CF"), ("精对苯二甲酸", "TA"), ("对二甲苯", "PX"),
    ("白砂糖", "SR"), ("甲醇", "MA"), ("玻璃", "FG"), ("动力煤", "ZC"), ("棉纱", "CY"),
    ("纯碱", "SA"), ("烧碱", "SH"), ("短纤", "PF"), ("瓶片", "PR"), ("粳稻", "JR"),
    ("菜籽", "RS"), ("普通小麦", "PM"), ("强麦", "WH"), ("早籼稻", "RI"), ("晚籼稻", "LR"),
    ("硅铁", "SF"), ("锰硅", "SM"), ("苹果", "AP"), ("红枣", "CJ"), ("尿素", "UR"),
    ("花生", "PK"),
    # 广州期货交易所 (GFEX)
    ("工业硅", "SI"), ("碳酸锂", "LC"), ("多晶硅", "PS"),
    # 中国金融期货交易所 (CFFEX)
    ("沪深300指数", "IF"), ("上证50指数", "IH"), ("中证500指数", "IC"), ("中证1000指数", "IM"),
    ("2年期国债", "TS"), ("5年期国债", "TF"), ("10年期国债", "T"), ("30年期国债", "TL")
]

# --- 输出与日期配置 ---
OUTPUT_FOLDER = "期货量价因子_CSV输出"
# 最终输出文件的开始和结束日期
START_DATE_FILTER = "2023-01-01"
END_DATE_FILTER = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')


# --- 量价趋势类因子参数 ---
K_TREND_SLOW_MA = 20          # 用于 T_macross (慢线) 和 T_maratio 的 K 值
K_TREND_FAST_MA = K_TREND_SLOW_MA // 2 # 用于 T_macross (快线) (K/2)
K_TREND_MAMOM_MA = 60         # 用于 T_mamom 的长周期均线窗口
N_TREND_MAMOM_LOOKBACK = 5    # 用于 T_mamom 的动量回顾期 (MA(t) - MA(t-N))
N_TREND_MOM_PERIOD = 20       # 用于 T_mom 的回顾期 (单位: 交易日)


# --- 2. 趋势因子计算类 ---
class FuturesTrendFactors:
    def __init__(self, df_ohlc):
        if not all(col in df_ohlc.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("DataFrame 必须包含 'open', 'high', 'low', 'close' 列")
        self.df = df_ohlc.copy()
        # 统一处理数据类型
        for col in ['open', 'high', 'low', 'close', 'volume', '持仓量']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    def calculate_t_macross(self, k_slow=K_TREND_SLOW_MA, k_fast=K_TREND_FAST_MA):
        if 'close' not in self.df.columns: return pd.Series(np.nan, index=self.df.index)
        ma_slow = self.df['close'].rolling(window=k_slow, min_periods=max(1, int(k_slow*0.8))).mean()
        ma_fast = self.df['close'].rolling(window=k_fast, min_periods=max(1, int(k_fast*0.8))).mean()
        t_macross = ma_fast / ma_slow
        t_macross.replace([np.inf, -np.inf], np.nan, inplace=True)
        return t_macross

    def calculate_t_mamom(self, k_ma=K_TREND_MAMOM_MA, n_lookback=N_TREND_MAMOM_LOOKBACK):
        if 'close' not in self.df.columns: return pd.Series(np.nan, index=self.df.index)
        ma_long = self.df['close'].rolling(window=k_ma, min_periods=max(1, int(k_ma*0.8))).mean()
        t_mamom = ma_long.diff(n_lookback)
        return t_mamom

    def calculate_t_maratio(self, k_ma=K_TREND_SLOW_MA):
        if 'close' not in self.df.columns: return pd.Series(np.nan, index=self.df.index)
        ma_k = self.df['close'].rolling(window=k_ma, min_periods=max(1, int(k_ma*0.8))).mean()
        t_maratio = self.df['close'] / ma_k
        t_maratio.replace([np.inf, -np.inf], np.nan, inplace=True)
        return t_maratio

    def calculate_t_mom(self, window=N_TREND_MOM_PERIOD):
        """ T_mom: 动量，过去一段时间的累计涨跌幅 (P_t / P_{t-N}) - 1 """
        if 'close' not in self.df.columns or len(self.df) <= window:
            return pd.Series(np.nan, index=self.df.index)
        close_shifted = self.df['close'].shift(window)
        t_mom = (self.df['close'] / close_shifted) - 1
        return t_mom

    def calculate_all_factors(self, k_slow_ma=K_TREND_SLOW_MA, k_fast_ma=K_TREND_FAST_MA,
                              k_mamom_ma=K_TREND_MAMOM_MA, n_mamom_lookback=N_TREND_MAMOM_LOOKBACK,
                              n_mom_period=N_TREND_MOM_PERIOD):
        results_df = self.df.copy()

        # 计算所有趋势因子
        results_df['t_macross'] = self.calculate_t_macross(k_slow=k_slow_ma, k_fast=k_fast_ma)
        results_df['t_mamom'] = self.calculate_t_mamom(k_ma=k_mamom_ma, n_lookback=n_mamom_lookback)
        results_df['t_maratio'] = self.calculate_t_maratio(k_ma=k_slow_ma)
        results_df['t_mom'] = self.calculate_t_mom(window=n_mom_period)
        
        return results_df


# --- 3. 辅助及主处理函数 ---
def sanitize_filename(filename):
    """移除或替换文件名中的非法字符"""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized

def process_single_future(product_name, product_code_short, start_date, end_date, output_folder):
    """
    处理单个期货品种：获取数据、预处理、计算因子、保存文件。
    引入了 warm-up (预热) 机制，确保输出日期范围内的数据是有效计算得出的。
    """
    akshare_symbol = product_code_short.upper() + "0"
    print(f"\n正在处理品种: {product_name} ({product_code_short}), Akshare代码: {akshare_symbol}")
    
    try:
        # 步骤 1: 获取完整的历史数据
        df_futures_ohlc = ak.futures_zh_daily_sina(symbol=akshare_symbol)

        if df_futures_ohlc.empty:
            print(f"未能获取到 {product_name} ({akshare_symbol}) 的数据。跳过。")
            return
        print(f"获取到 {product_name} 原始数据 {df_futures_ohlc.shape[0]} 条。")

        # 步骤 2: 数据预处理
        df_futures_ohlc.rename(columns={
            '日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low', '收盘价': 'close',
            '成交量': 'volume', '持仓量': '持仓量'
        }, inplace=True, errors='ignore')

        required_cols = ['date', 'open', 'high', 'low', 'close']
        if not all(col in df_futures_ohlc.columns for col in required_cols):
            print(f"错误: 数据中缺少核心OHLC列。跳过。")
            return

        cols_to_use = required_cols.copy()
        for col in ['volume', '持仓量']:
            if col in df_futures_ohlc.columns:
                cols_to_use.append(col)
            
        df_processed = df_futures_ohlc[list(dict.fromkeys(cols_to_use))].copy()
        df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        df_processed.dropna(subset=['date'], inplace=True)
        df_processed.set_index('date', inplace=True)
        df_processed.sort_index(inplace=True)

        # --- [核心修改] 预热期 (Warm-up) 逻辑 ---
        # 1. 确定计算所需的最长回看窗口
        max_lookback = max(K_TREND_SLOW_MA, K_TREND_MAMOM_MA, N_TREND_MOM_PERIOD, N_TREND_MAMOM_LOOKBACK)
        # 2. 增加一个buffer(例如乘以1.5倍)来覆盖非交易日，确保有足够的天数
        warm_up_days = int(max_lookback * 1.5) 
        # 3. 计算用于数据切片的、包含预热期的真实计算起始日
        calculation_start_date = (pd.to_datetime(start_date) - pd.Timedelta(days=warm_up_days)).strftime('%Y-%m-%d')
        
        # 使用包含预热期的时间范围来切片数据，用于后续计算
        df_for_calculation = df_processed.loc[calculation_start_date:end_date]

        if df_for_calculation.empty:
            print(f"{product_name} 在包含预热期的日期范围 {calculation_start_date} 到 {end_date} 内无数据。")
            return
        print(f"{product_name} 用于计算的数据 (含预热期) 共 {df_for_calculation.shape[0]} 条。")
        # --- [修改结束] ---

        # 步骤 3: 计算因子 (在包含预热期的数据上进行)
        factor_calculator = FuturesTrendFactors(df_for_calculation)
        df_with_factors_extended = factor_calculator.calculate_all_factors()

        # 步骤 4: 筛选最终结果并保存
        # 从计算结果中，仅截取用户指定的日期范围
        df_final_output = df_with_factors_extended.loc[start_date:end_date]

        if df_final_output.empty:
             print(f"{product_name} 在指定的最终输出日期范围 {start_date} 到 {end_date} 内无数据。")
             return

        safe_product_name = sanitize_filename(product_name)
        safe_product_code = sanitize_filename(product_code_short)
        
        csv_filename = f"{safe_product_name}_{safe_product_code}_trend_factors.csv"
        csv_filepath = os.path.join(output_folder, csv_filename)
        
        df_final_output.to_csv(csv_filepath, index=True, float_format='%.6f')
        print(f"成功为 {product_name} ({akshare_symbol}) 计算量价因子并保存到: {csv_filepath}")

    except Exception as e:
        print(f"处理品种 {product_name} ({akshare_symbol}) 时发生严重错误: {e}")
        import traceback
        print(traceback.format_exc())

# --- 4. 主程序入口 ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"已创建输出文件夹: {OUTPUT_FOLDER}")

    total_products = len(PRODUCTS_TO_PROCESS)
    print("="*50)
    print(f"开始处理 {total_products} 个期货品种的量价因子...")
    print(f"数据最终输出日期范围: {START_DATE_FILTER} 至 {END_DATE_FILTER}")
    print("="*50)
    
    for i, (product_name, product_code_short) in enumerate(PRODUCTS_TO_PROCESS):
        print(f"\n--- [处理进度: {i+1}/{total_products}] ---")
        process_single_future(product_name, product_code_short, START_DATE_FILTER, END_DATE_FILTER, OUTPUT_FOLDER)
        
        # Friendly delay to avoid being blocked by the data provider
        if i < total_products - 1:
            time.sleep(1)

    print("\n\n所有品种处理完毕。")
    print(f"请在文件夹 '{OUTPUT_FOLDER}' 中查看输出的CSV文件。")