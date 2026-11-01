import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# List of files
files = [
    '贵金属_panel_data.csv',
    '黑色金属_panel_data.csv',
    '能源化工_panel_data.csv',
    '农产品_panel_data.csv',
    '其他_panel_data.csv',
    '新能源与新材料_panel_data.csv',
    '有色金属_panel_data.csv'
]

# Load and combine data
all_df = []
for f in files:
    df = pd.read_csv(f)
    all_df.append(df)

df_all = pd.concat(all_df, ignore_index=True)

# Basic cleaning
df_all['date'] = pd.to_datetime(df_all['date'], errors='coerce')
df_all = df_all.dropna(subset=['date', 'forward_return'])
# Filter out 1970 dates if any
df_all = df_all[df_all['date'] > '2000-01-01']

# Check info
print(df_all.info())
print(df_all.head())

# Define factors
factors = ['C_frontnext', 'C_mainsub', 'C_nsnormal']

def calculate_ic(df, factor_cols, target_col='forward_return'):
    ic_results = []
    # Group by date to calculate cross-sectional IC
    grouped = df.groupby('date')
    
    for date, group in grouped:
        if len(group) < 5: # Need enough products to rank
            continue
        
        daily_ics = {'date': date}
        for factor in factor_cols:
            if group[factor].nunique() > 1:
                # Rank IC (Spearman)
                ic = group[factor].corr(group[target_col], method='spearman')
                daily_ics[factor] = ic
            else:
                daily_ics[factor] = np.nan
        ic_results.append(daily_ics)
        
    return pd.DataFrame(ic_results)

ic_df = calculate_ic(df_all, factors)

# IC Statistics
ic_stats = pd.DataFrame()
for factor in factors:
    mean_ic = ic_df[factor].mean()
    std_ic = ic_df[factor].std()
    ir = mean_ic / std_ic if std_ic != 0 else 0
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_df[factor].dropna())))
    
    ic_stats[factor] = [mean_ic, std_ic, ir, t_stat]

ic_stats.index = ['Mean IC', 'Std IC', 'ICIR', 't-stat']
print(ic_stats)

# Sector analysis
sector_ic_stats = {}
for sector in df_all['sector'].unique():
    sector_df = df_all[df_all['sector'] == sector]
    s_ic_df = calculate_ic(sector_df, factors)
    s_stats = s_ic_df[factors].mean()
    sector_ic_stats[sector] = s_stats

sector_summary = pd.DataFrame(sector_ic_stats)
print("\nSector Mean IC Summary:")
print(sector_summary)

# Save cumulative IC for visualization
cumulative_ic = ic_df.set_index('date')[factors].cumsum()
cumulative_ic.to_csv('cumulative_ic.csv')
ic_stats.to_csv('ic_stats.csv')
sector_summary.to_csv('sector_summary.csv')