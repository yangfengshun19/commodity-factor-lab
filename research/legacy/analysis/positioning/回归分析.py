import pandas as pd
import statsmodels.api as sm

# --- 1. Load Preprocessed Panel Data ---
# [MODIFIED] Read the new positioning factor panel data file.
try:
    panel_data = pd.read_csv('panel_data_for_positioning_factor_analysis.csv')
except FileNotFoundError:
    print("ERROR: 'panel_data_for_positioning_factor_analysis.csv' not found.")
    print("Please run the modified '数据处理.py' script first to generate this file.")
    exit()

# Prepare data
panel_data['date'] = pd.to_datetime(panel_data['date'])
panel_data.set_index(['date', 'code'], inplace=True)
panel_data.dropna(inplace=True)

# --- 2. Prepare Data for Regression Analysis ---
Y = panel_data['forward_return']

# [MODIFIED] Update factor columns to the new positioning factors.
factor_columns = [
    'pr_netmom_top5', 'pr_lschgratio_top5', 
    'pr_netmom_top10', 'pr_lschgratio_top10', 
    'pr_netmom_top15', 'pr_lschgratio_top15', 
    'pr_netmom_top20', 'pr_lschgratio_top20'
]
X = panel_data[factor_columns]

X = sm.add_constant(X)

# --- 3. Perform Multiple Linear Regression ---
model = sm.OLS(Y, X)
results = model.fit()

# --- 4. Print Regression Results ---
# [MODIFIED] Update the title.
print("--- Positioning Factors Multiple Regression Analysis Results ---")
print(results.summary())

# --- 5. Guide to Interpreting Results ---
print("\n--- Guide to Interpretation ---")
print("1. R-squared: Explains the model's ability to predict the variability of future returns.")
print("2. Coef: Shows the positive/negative relationship between a factor and future returns.")
print("3. P>|t|: Determines if a coefficient is statistically significant (typically < 0.05).")
print("4. [0.025 0.975] (Confidence Interval): Assesses the stability of the coefficient.")