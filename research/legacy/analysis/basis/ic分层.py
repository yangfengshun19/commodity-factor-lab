import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set Chinese font for plotting
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Failed to set Chinese font: {e}")

# --- 1. Load Preprocessed Panel Data ---
# [MODIFIED] Read the new futures-spot factor panel data file.
try:
    panel_data = pd.read_csv('panel_data_for_futures_spot_factor_analysis.csv')
except FileNotFoundError:
    print("ERROR: 'panel_data_for_futures_spot_factor_analysis.csv' not found.")
    print("Please run the modified '数据处理.py' script first to generate this file.")
    exit()

panel_data['date'] = pd.to_datetime(panel_data['date'])

# --- 2. Define the List of Factors to Analyze ---
# [MODIFIED] Update factor columns to the new futures-spot factors.
factor_columns = ['fs_basis', 'fs_carry', 'fs_exmom']

# Loop through and analyze each factor
for factor_name in factor_columns:
    print(f"\n{'='*30}\nAnalyzing Factor: {factor_name}\n{'='*30}")
    
    # --- 3. IC (Information Coefficient) Analysis ---
    ic_series = panel_data.groupby('date').apply(
        lambda x: x[factor_name].corr(x['forward_return'])
    ).dropna()
    
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ir = ic_mean / ic_std if ic_std != 0 else 0
    positive_ratio = (ic_series > 0).sum() / len(ic_series) if len(ic_series) > 0 else 0
    
    print("\n--- IC Analysis Results ---")
    print(f"Factor: {factor_name}")
    print(f"Mean IC: {ic_mean:.4f}, Std Dev of IC: {ic_std:.4f}")
    print(f"Information Ratio (IR): {ir:.4f}, IC > 0 Ratio: {positive_ratio:.2%}")
    
    # Plot IC time series
    plt.figure(figsize=(12, 6))
    ic_series.plot(kind='line', label='Daily IC')
    plt.axhline(ic_mean, color='red', linestyle='--', label=f'Mean IC = {ic_mean:.4f}')
    plt.axhline(0, color='gray', linestyle='-')
    plt.title(f'IC Time Series for Factor "{factor_name}"')
    plt.xlabel('Date'); plt.ylabel('IC Value')
    plt.legend(); plt.grid(True)
    plt.savefig(f'{factor_name}_ic_series.png')
    plt.show()

    # --- 4. Layered Backtesting Analysis ---
    quantiles = panel_data.groupby('date')[factor_name].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    panel_data['group'] = quantiles
    
    grouped_returns = panel_data.groupby(['date', 'group'])['forward_return'].mean().unstack()
    cumulative_returns = (1 + grouped_returns).cumprod()
    
    print("\n--- Layered Backtest Results (Annualized Returns) ---")
    if not cumulative_returns.empty:
        annualized_returns = (cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns))) - 1
        print(annualized_returns)
    else:
        print("Could not calculate annualized returns due to empty backtest data.")

    # Plot layered backtest performance
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot(legend=True)
    plt.title(f'Layered Backtest Performance for Factor "{factor_name}"')
    plt.xlabel('Date'); plt.ylabel('Cumulative Return (Equity)')
    plt.grid(True)
    plt.savefig(f'{factor_name}_layered_backtest.png')
    plt.show()

print("\nAll factor analyses are complete!")