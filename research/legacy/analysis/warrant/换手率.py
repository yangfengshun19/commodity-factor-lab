import pandas as pd

# --- 1. Load Preprocessed Panel Data ---
# [MODIFIED] Read the new unfiltered warrant factor panel data file.
try:
    panel_data = pd.read_csv('panel_data_for_warrant_factors_unfiltered_analysis.csv')
except FileNotFoundError:
    print("ERROR: 'panel_data_for_warrant_factors_unfiltered_analysis.csv' not found.")
    print("Please run the modified '数据处理.py' script first to generate this file.")
    exit()

panel_data['date'] = pd.to_datetime(panel_data['date'])
panel_data.set_index('date', inplace=True)


# --- 2. Turnover Calculation Function ---
def calculate_turnover(panel_data, factor_name, group_to_track='top'):
    """Calculates the turnover rate for a specific factor and group."""
    
    valid_data = panel_data.dropna(subset=[factor_name])
    
    if valid_data.empty:
        return 0
        
    valid_data['group'] = valid_data.groupby(valid_data.index)[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    
    target_group_label = 4 if group_to_track == 'top' else 0
    target_group_df = valid_data[valid_data['group'] == target_group_label]
    
    daily_holdings = target_group_df.groupby(target_group_df.index)['code'].apply(set)
    daily_holdings = daily_holdings.asfreq('B').ffill()
    
    turnovers = []
    for i in range(1, len(daily_holdings)):
        last_day_holdings = daily_holdings.iloc[i-1]
        current_day_holdings = daily_holdings.iloc[i]
        
        if pd.isna(last_day_holdings) or pd.isna(current_day_holdings):
            continue
        
        sell_count = len(last_day_holdings - current_day_holdings)
        last_day_total_count = len(last_day_holdings)
        
        if last_day_total_count == 0:
            turnovers.append(0)
            continue
            
        daily_turnover = sell_count / last_day_total_count
        turnovers.append(daily_turnover)
        
    avg_daily_turnover = sum(turnovers) / len(turnovers) if turnovers else 0
    annualized_turnover = avg_daily_turnover * 252
    
    return annualized_turnover

# --- 3. Analyze Each Factor ---
# [MODIFIED] Update factor columns to the warrant factors.
factor_columns = ['warrant_count', 'W_pct', 'W_ratio']

print("--- Warrant Factor (Unfiltered) Portfolio Turnover Analysis ---")
for factor in factor_columns:
    print(f"\n--- Analyzing Factor: {factor} ---")
    
    top_turnover = calculate_turnover(panel_data.copy(), factor, group_to_track='top')
    print(f"Long Portfolio (Top Quintile) Annualized Turnover: {top_turnover:.2%}")
    
    bottom_turnover = calculate_turnover(panel_data.copy(), factor, group_to_track='bottom')
    print(f"Short Portfolio (Bottom Quintile) Annualized Turnover: {bottom_turnover:.2%}")

print("\n--- Interpreting Turnover Rate ---")
print("Turnover reflects the frequency of trading required to maintain a portfolio.")
print("A lower turnover rate generally implies lower transaction costs and more stable holdings.")
print("An extremely high turnover rate (e.g., >1000%) may indicate that the factor is too volatile, leading to high potential costs in live trading.")