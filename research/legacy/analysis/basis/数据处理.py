import pandas as pd
import re
import os

# --- Step 1: List of Products ---
# This list remains the foundation for all your analyses.
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

# --- Configuration Parameters ---

# Folder where your factor files are located.
DATA_FOLDER = "" 

# Data integrity threshold: products with fewer data points will be excluded.
MIN_DATA_POINTS_THRESHOLD = 200

# [MODIFIED] Specify the factor columns to be analyzed (from your futures-spot script).
FACTOR_COLUMNS = ['fs_basis', 'fs_carry', 'fs_exmom']


# --- Step 2: Automatically Build File Map ---
# [MODIFIED] File naming logic updated to match "_basis_factors.csv".
files_map = {}
for name, code in PRODUCTS_TO_PROCESS:
    filename = os.path.join(DATA_FOLDER, f"{name}_{code}_basis_factors.csv")
    files_map[code] = filename
    
# --- Step 3: Load, Filter, and Merge Data ---
all_data_list = []
print("--- Starting Data Load, Filter, and Merge (Futures-Spot Factors) ---")

for code, file_name in files_map.items():
    try:
        df = pd.read_csv(file_name)
        
        # The logic to remove tables with many missing items (data filtering)
        if len(df) < MIN_DATA_POINTS_THRESHOLD:
            print(f"!!! WARNING: File '{file_name}' (Product: {code}) has insufficient data ({len(df)} < {MIN_DATA_POINTS_THRESHOLD}). Skipped.")
            continue
            
        df['code'] = code
        df['date'] = pd.to_datetime(df['date'])
        
        # Check if required factor columns exist
        missing_cols = [col for col in FACTOR_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"!!! WARNING: File '{file_name}' is missing factor columns: {missing_cols}. Skipped.")
            continue

        all_data_list.append(df)
        print(f"SUCCESS: Loaded '{file_name}' (Product: {code}), Data size: {len(df)} rows")
        
    except FileNotFoundError:
        print(f"--- INFO: File '{file_name}' not found. Skipped automatically.")

if not all_data_list:
    print("\n--- ERROR ---")
    print("Failed to load any valid data files. The program will terminate. Please check:")
    print("1. Your factor CSV filenames match the format 'ProductName_code_basis_factors.csv'.")
    print(f"2. At least one data file meets the minimum requirement of {MIN_DATA_POINTS_THRESHOLD} rows.")
else:
    combined_df = pd.concat(all_data_list, ignore_index=True)

    # --- Step 4: Calculate the Target Variable Y (1-day forward return) ---
    print("\n--- Calculating forward returns... ---")
    combined_df.sort_values(by=['code', 'date'], inplace=True)
    
    if 'close' not in combined_df.columns:
        print("\n--- ERROR ---")
        print("The 'close' price column is required to calculate forward_return but was not found in the CSV files.")
        print("Please ensure your '3-期现类因子csv.py' script includes the 'close' price in its output CSVs.")
        exit()
        
    combined_df['forward_return'] = combined_df.groupby('code')['close'].pct_change().shift(-1)
    print("Calculation complete.")

    # --- Step 5: Construct and Save the Final Panel Data ---
    columns_to_keep = ['date', 'code', 'forward_return'] + FACTOR_COLUMNS
    final_df = combined_df[columns_to_keep]

    cleaned_df = final_df.dropna()
    panel_df = cleaned_df.set_index(['date', 'code'])

    print("\n--- Final Generated Panel Data (First 10 rows preview) ---")
    print(panel_df.head(10))

    # [MODIFIED] Update the output filename to be specific to this analysis
    output_filename = 'panel_data_for_futures_spot_factor_analysis.csv'
    panel_df.to_csv(output_filename)
    
    print("\n--- All Done! ---")
    print(f"Analysis-ready panel data has been saved to: {output_filename}")