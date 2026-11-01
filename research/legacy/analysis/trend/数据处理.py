import pandas as pd
import re
import os

# --- 第一步：品种列表 ---
# 这个列表保持不变，是您所有分析的基础
PRODUCTS_TO_PROCESS = [
    ("螺纹钢", "rb"), ("热轧卷板", "hc"), ("石油沥青", "bu"), ("天然橡胶", "ru"), ("合成橡胶", "br"),
    ("燃料油", "fu"), ("纸浆", "sp"), ("铜", "cu"), ("铝", "al"), ("氧化铝", "ao"),
    ("铅", "pb"), ("锌", "zn"), ("锡", "sn"), ("镍", "ni"), ("不锈钢", "ss"),
    ("黄金", "au"), ("白银", "ag"), ("线材", "wr"),
    ("20号胶", "nr"), ("低硫燃料油", "lu"), ("国际铜", "bc"), ("原油", "sc"), ("集运指数(欧线)", "ec"),
    ("黄大豆1号", "a"), ("黄大豆2号", "b"), ("黄玉米", "c"), ("玉米淀粉", "cs"), ("豆粕", "m"),
    ("豆油", "y"), ("棕榈油", "p"), ("铁矿石", "i"), ("焦炭", "j"), ("焦煤", "jm"),
    ("聚乙烯", "l"), ("聚氯乙烯", "v"), ("聚丙烯", "pp"), ("乙二醇", "eg"), ("苯乙烯", "eb"),
    ("甲醇", "MA"), ("精对苯二甲酸", "TA"), ("短纤", "PF"), ("液化石油气", "pg"), ("玻璃", "FG"),
    ("纯碱", "SA"), ("尿素", "UR"), ("棉花", "CF"), ("棉纱", "CY"), ("白糖", "SR"),
    ("苹果", "AP"), ("红枣", "CJ"), ("鸡蛋", "jd"), ("生猪", "lh"), ("工业硅", "si"),
    ("碳酸锂", "lc"),
    ("2年期国债", "TS"), ("5年期国债", "TF"), ("10年期国债", "T"), ("30年期国债", "TL")
]

# --- 配置参数 ---

# 因子文件所在的文件夹
DATA_FOLDER = "" 

# 数据完整性阈值：数据点少于此数值的品种将被剔除
MIN_DATA_POINTS_THRESHOLD = 200

# 【修改】指定要分析的因子列（对应"4-量价趋势类.py"脚本）
FACTOR_COLUMNS = ['t_macross', 't_mamom', 't_maratio', 't_mom']


# --- 第二步：自动构建文件映射 ---
# 【修改】文件名构造逻辑更新，以匹配 "trend_factors" 文件
files_map = {}
for name, code in PRODUCTS_TO_PROCESS:
    # 构造新的文件名
    filename = os.path.join(DATA_FOLDER, f"{name}_{code}_trend_factors.csv")
    files_map[code] = filename
    
# --- 第三步：加载、筛选和合并数据 ---
all_data_list = []
print("--- 开始加载、筛选和合并数据 (趋势类因子) ---")

for code, file_name in files_map.items():
    try:
        df = pd.read_csv(file_name)
        
        # 数据筛选逻辑
        if len(df) < MIN_DATA_POINTS_THRESHOLD:
            print(f"!!! 警告: 文件 '{file_name}' (品种: {code}) 数据量不足 ({len(df)} < {MIN_DATA_POINTS_THRESHOLD})，已跳过。")
            continue
            
        df['code'] = code
        df['date'] = pd.to_datetime(df['date'])
        
        # 检查必需的因子列是否存在
        missing_cols = [col for col in FACTOR_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"!!! 警告: 文件 '{file_name}' 缺少因子列: {missing_cols}，已跳过。")
            continue

        all_data_list.append(df)
        print(f"成功: 已加载 '{file_name}' (品种: {code})，数据量: {len(df)} 行")
        
    except FileNotFoundError:
        print(f"--- 信息: 文件 '{file_name}' 未找到，自动跳过。")

if not all_data_list:
    print("\n--- 错误 ---")
    print("未能成功加载任何有效的数据文件，程序终止。请检查：")
    print("1. 您的因子CSV文件名是否符合 '品种中文名_代码_trend_factors.csv' 格式。")
    print(f"2. 是否有数据文件的数据量达到了 {MIN_DATA_POINTS_THRESHOLD} 行的最低要求。")
else:
    combined_df = pd.concat(all_data_list, ignore_index=True)

    # --- 第四步：计算未来1日收益率 ---
    print("\n--- 正在计算未来收益率 (forward_return) ---")
    combined_df.sort_values(by=['code', 'date'], inplace=True)
    combined_df['forward_return'] = combined_df.groupby('code')['close'].pct_change().shift(-1)
    print("计算完成。")

    # --- 第五步：构建并保存最终的面板数据 ---
    columns_to_keep = ['date', 'code', 'forward_return'] + FACTOR_COLUMNS
    final_df = combined_df[columns_to_keep]

    cleaned_df = final_df.dropna()
    panel_df = cleaned_df.set_index(['date', 'code'])

    print("\n--- 最终生成的面板数据 (前10行预览) ---")
    print(panel_df.head(10))

    # 【修改】更新输出文件名
    output_filename = 'panel_data_for_trend_factor_analysis.csv'
    panel_df.to_csv(output_filename)
    
    print("\n--- 全部完成！ ---")
    print(f"分析就绪的面板数据已保存至: {output_filename}")