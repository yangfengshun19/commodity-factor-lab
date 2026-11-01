import pandas as pd
import statsmodels.api as sm

# --- 1. 加载预处理好的面板数据 ---
# 【修改】读取新的趋势因子面板数据文件
try:
    panel_data = pd.read_csv('panel_data_for_trend_factor_analysis.csv')
except FileNotFoundError:
    print("错误：未找到 'panel_data_for_trend_factor_analysis.csv' 文件。")
    print("请先运行修改后的 '数据处理.py' 脚本来生成这个文件。")
    exit()

# 准备数据
panel_data['date'] = pd.to_datetime(panel_data['date'])
panel_data.set_index(['date', 'code'], inplace=True)
panel_data.dropna(inplace=True)

# --- 2. 准备回归分析的数据 ---
Y = panel_data['forward_return']

# 【修改】将因子列更新为新的趋势因子
factor_columns = ['t_macross', 't_mamom', 't_maratio', 't_mom']
X = panel_data[factor_columns]

X = sm.add_constant(X)

# --- 3. 执行多元线性回归 ---
model = sm.OLS(Y, X)
results = model.fit()

# --- 4. 打印回归结果 ---
# 【修改】更新标题
print("--- 趋势类因子 (Trend Factors) 多元回归分析结果 ---")
print(results.summary())

# --- 5. 结果解读提示 ---
print("\n--- 结果解读提示 ---")
print("1. R-squared (R方): 解释模型对未来收益率的解释力强度。")
print("2. Coef (系数): 显示因子与未来收益率的正/负相关性。")
print("3. P>|t| (P值): 判断系数是否显著（通常 < 0.05）。")
print("4. [0.025 0.975] (置信区间): 评估系数的稳定性。")