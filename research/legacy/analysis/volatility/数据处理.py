import pandas as pd
import re
import os

# --- Step 1: List of Products ---
PRODUCTS_TO_PROCESS = [
    ("螺纹钢", "rb"), ("热轧卷板", "hc"), ("石油沥青", "bu"), ("天然橡胶", "ru"),
    ("燃料油", "fu"), ("纸浆", "sp"), ("铜", "cu"), ("铝", "al"),
    ("铅", "pb"), ("锌", "zn"), ("锡", "sn"), ("镍", "ni"),
    ("黄金", "au"), ("白银", "ag"),
    ("豆粕", "m"), ("豆油", "y"), ("棕榈油", "p"), ("铁矿石", "i"), ("焦炭", "j"), ("焦煤", "jm"),
    ("聚乙烯", "l"), ("聚氯乙烯", "v"), ("聚丙烯", "pp"), ("乙二醇", "eg"),
    ("苯乙烯", "eb"), ("液化石油气", "pg"),
    ("菜粕", "RM"), ("菜籽油", "OI"), ("一号棉花", "CF"), ("精对苯二甲酸", "TA"),
    ("白砂糖", "SR"), ("甲醇", "MA"), ("玻璃", "FG"),
    ("纯碱", "SA"), ("尿素", "UR"),
    ("工业硅", "SI"), ("碳酸锂", "LC"),
    ("沪深300指数", "IF"), ("上证50指数", "IH"), ("中证500指数", "IC"), ("中证1000指数", "IM"),
    ("2年期国债", "TS"), ("5年期国债", "TF"), ("10年期国债", "T")
]

# --- Configuration Parameters ---
DATA_FOLDER = "" 

# [MODIFIED] Data integrity threshold reverted to the original value.
MIN_DATA_POINTS_THRESHOLD = 200

# [MODIFIED] Specify the factor columns to be analyzed.
FACTOR_COLUMNS = ['vol_std', 'vol_gk', 'vol_rs']


# --- Step 2: Automatically Build File Map ---
# [MODIFIED] File naming logic updated to match "_vol_factors.csv".
files_map = {}
for name, code in PRODUCTS_TO_PROCESS:
    filename = os.path.join(DATA_FOLDER, f"{name}_{code}_vol_factors.csv")
    files_map[code] = filename
    
# --- Step 3: Load, Filter, and Merge Data ---
all_data_list = []
print("--- Starting Data Load, Filter, and Merge (Volatility Factors) ---")

for code, file_name in files_map.items():
    try:
        df = pd.read_csv(file_name)
        
        # The logic to remove tables with many missing items (data filtering)
        if len(df) < MIN_DATA_POINTS_THRESHOLD:
            print(f"!!! WARNING: File '{file_name}' (Product: {code}) has insufficient data ({len(df)} < {MIN_DATA_POINTS_THRESHOLD}). Skipped.")
            continue
            
        if 'date' not in df.columns:
            print(f"!!! ERROR: File '{file_name}' is missing the required 'date' column. Skipped.")
            continue

        df['code'] = code
        df['date'] = pd.to_datetime(df['date'])
        
        missing_cols = [col for col in FACTOR_COLUMNS if col not in df.columns]
        if missing_cols:
            print(f"!!! WARNING: File '{file_name}' is missing factor columns: {missing_cols}. Skipped.")
            continue

        all_data_list.append(df)
        print(f"SUCCESS: Loaded '{file_name}' (Product: {code}), Data size: {len(df)} rows")
        
    except FileNotFoundError:
        print(f"--- INFO: File '{file_name}' not found. Skipped automatically.")
    except Exception as e:
        print(f"!!! ERROR: Could not process file '{file_name}'. Reason: {e}. Skipped.")


if not all_data_list:
    print("\n--- ERROR ---")
    print("Failed to load any valid data files. The program will terminate.")
else:
    combined_df = pd.concat(all_data_list, ignore_index=True)

    # --- Step 4: Calculate the Target Variable Y (1-day forward return) ---
    print("\n--- Calculating forward returns... ---")
    combined_df.sort_values(by=['code', 'date'], inplace=True)
    
    if 'close' not in combined_df.columns:
        print("\n--- ERROR ---")
        print("The 'close' price column is required to calculate forward_return but was not found in the CSV files.")
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

    # [MODIFIED] Update the output filename for this specific analysis.
    output_filename = 'panel_data_for_volatility_factor_analysis.csv'
    panel_df.to_csv(output_filename)
    
    print("\n--- All Done! ---")
    print(f"Analysis-ready panel data has been saved to: {output_filename}")