import pandas as pd
import re
import os

# --- 第一步：从您的原始脚本中复制品种列表 ---
# 这个列表现在是唯一需要维护的地方
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

# 定义您的因子文件所在的文件夹
# 如果CSV文件和这个脚本在同一个地方，留空即可。如果在子文件夹，则填写 "your_folder_name/"
DATA_FOLDER = "" 

# 【新增】数据完整性阈值：如果一个品种的数据点少于此数值，将被剔除
# 通常一年大约有252个交易日，这里设置为200作为一个宽松的筛选标准
MIN_DATA_POINTS_THRESHOLD = 200

# 【修改】指定要分析的因子列（与您新因子脚本中计算的因子对应）
# 根据 "鸡蛋_jd_value_factors_ohlc.csv" 文件，新的因子列是 'Val_halfyear' 和 'Val_oneyear'
FACTOR_COLUMNS = ['Val_halfyear', 'Val_oneyear']


# --- 第二步：根据品种列表，自动构建文件映射 ---
# 【修改】文件名构造逻辑已更新，以匹配新的 "value_factors_ohlc" 文件
files_map = {}
for name, code in PRODUCTS_TO_PROCESS:
    # 构造新的文件名
    filename = os.path.join(DATA_FOLDER, f"{name}_{code}_value_factors_ohlc.csv")
    files_map[code] = filename
    
# --- 第三步：程序会自动加载、筛选和合并数据 ---
all_data_list = []
print("--- 开始加载、筛选和合并数据 ---")

for code, file_name in files_map.items():
    try:
        df = pd.read_csv(file_name)
        
        # 【新增】数据筛选逻辑
        if len(df) < MIN_DATA_POINTS_THRESHOLD:
            print(f"!!! 警告: 文件 '{file_name}' (品种: {code}) 数据量不足 ({len(df)} < {MIN_DATA_POINTS_THRESHOLD}个数据点)，已跳过。")
            continue # 跳过这个文件，处理下一个
            
        df['code'] = code # 添加品种代码列
        df['date'] = pd.to_datetime(df['date']) # 格式化日期
        
        # 检查必需的因子列是否存在
        missing_cols = [col for col in FACTOR_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"!!! 警告: 文件 '{file_name}' 缺少因子列: {missing_cols}，已跳过。")
            continue

        all_data_list.append(df)
        print(f"成功: 已加载文件 '{file_name}' (品种: {code})，数据量: {len(df)} 行")
        
    except FileNotFoundError:
        # 这个警告是正常的，因为列表中某些品种可能没有对应的因子文件
        print(f"--- 信息: 文件 '{file_name}' 未找到，自动跳过。")

if not all_data_list:
    print("\n--- 错误 ---")
    print("未能成功加载任何有效的数据文件，程序终止。请检查：")
    print("1. 您的因子CSV文件名是否符合 '品种中文名_代码_value_factors_ohlc.csv' 格式。")
    print(f"2. 是否有数据文件的数据量达到了 {MIN_DATA_POINTS_THRESHOLD} 行的最低要求。")
else:
    combined_df = pd.concat(all_data_list, ignore_index=True)

    # --- 第四步：计算目标变量 Y (未来1日收益率) ---
    print("\n--- 正在计算未来收益率 (forward_return) ---")
    combined_df.sort_values(by=['code', 'date'], inplace=True)
    # 使用收盘价'close'计算未来收益率
    combined_df['forward_return'] = combined_df.groupby('code')['close'].pct_change().shift(-1)
    print("计算完成。")

    # --- 第五步：构建并保存最终的面板数据 ---
    # 仅保留目标变量和指定的因子列
    columns_to_keep = ['date', 'code', 'forward_return'] + FACTOR_COLUMNS
    final_df = combined_df[columns_to_keep]

    # 剔除任何一行中存在缺失值的行（主要是最后一天没有forward_return的行）
    cleaned_df = final_df.dropna()
    
    panel_df = cleaned_df.set_index(['date', 'code'])

    print("\n--- 最终生成的面板数据 (前10行预览) ---")
    print(panel_df.head(10))

    output_filename = 'panel_data_for_value_factor_analysis.csv'
    panel_df.to_csv(output_filename)
    
    print("\n--- 全部完成！ ---")
    print(f"分析就绪的面板数据已保存至: {output_filename}")