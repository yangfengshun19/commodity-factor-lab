import pandas as pd
import statsmodels.api as sm

# --- 1. 加载预处理好的面板数据 ---
# 【修改】读取新的价值因子面板数据文件
try:
    panel_data = pd.read_csv('panel_data_for_value_factor_analysis.csv')
except FileNotFoundError:
    print("错误：未找到 'panel_data_for_value_factor_analysis.csv' 文件。")
    print("请先运行修改后的 '数据处理.py' 脚本来生成这个文件。")
    exit()

# 将 'date' 列转换为 datetime 对象
panel_data['date'] = pd.to_datetime(panel_data['date'])
panel_data.set_index(['date', 'code'], inplace=True)

# --- 2. 准备回归分析的数据 ---
# 确保数据中没有无穷大或非数值(NaN)
panel_data.dropna(inplace=True)

# 定义因变量 Y
Y = panel_data['forward_return']

# 定义自变量 X (因子)
# 【修改】将因子列更新为新的价值因子
factor_columns = ['Val_halfyear', 'Val_oneyear']
X = panel_data[factor_columns]

# 添加常数项 (截距)
X = sm.add_constant(X)

# --- 3. 执行多元线性回归 ---
model = sm.OLS(Y, X)
results = model.fit()

# --- 4. 打印回归结果 ---
# 【修改】更新标题，指明是价值因子的回归分析
print("--- 价值因子 (Value Factors) 多元回归分析结果 ---")
print(results.summary())

# --- 5. 结果解读提示 ---
print("\n--- 结果解读提示 ---")
print("1. R-squared (R方):")
print("   - 表示模型对因变量（未来收益率）变异性的解释程度。越高说明模型的解释力越强。")
print("2. Coef (系数):")
print("   - 显示了每个因子与未来收益率之间的关系。正系数表示正相关，负系数表示负相关。")
print("3. P>|t| (P值):")
print("   - 判断系数显著性的关键指标。通常，P值小于0.05被认为是统计上显著的，意味着该因子对未来收益率有显著影响。")
print("4. [0.025 0.975] (置信区间):")
print("   - 提供了系数真实值的95%置信范围。如果这个区间不包含0，那么该系数通常被认为是显著的。")