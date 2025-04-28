import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取清洗数据
df = pd.read_csv("electricity_data_cleaned.csv")

# 去除空格 & 统一大小写
df['Country'] = df['Country'].str.strip()
df['Features'] = df['Features'].str.strip().str.lower()

# 年份列表（1980–2021）
years = [str(y) for y in range(1980, 2022)]

# 创建空字典收集各国 Net Consumption 时间序列
net_consumption_dict = {}

# 找出所有国家
all_countries = df['Country'].unique()

# 遍历每个国家，构建净用电数据
for country in all_countries:
    try:
        gen = df[(df['Country'] == country) & (df['Features'] == 'net generation')][years].values.flatten().astype(float)
        imp = df[(df['Country'] == country) & (df['Features'] == 'imports')][years].values.flatten().astype(float)
        exp = df[(df['Country'] == country) & (df['Features'] == 'exports')][years].values.flatten().astype(float)
        loss = df[(df['Country'] == country) & (df['Features'] == 'distribution losses')][years].values.flatten().astype(float)
        net_cons = gen + imp - exp - loss
        net_consumption_dict[country] = net_cons
    except:
        continue  # 某个国家缺数据，跳过

# 计算所有国家的平均 Net Consumption（1980–2021）
avg_net_cons = {country: np.mean(vals) for country, vals in net_consumption_dict.items()}
top_country = max(avg_net_cons, key=avg_net_cons.get)
print(f"Net Consumption 最高国家是：{top_country}")

# 获取该国家的数据
y_values = net_consumption_dict[top_country]  # shape (42,)
x_years = np.array(range(1980, 2022)).reshape(-1, 1)  # shape (42, 1)

# 拆分训练集（80%）和测试集（20%）
split_idx = int(0.8 * len(x_years))  # = 33
X_train, X_test = x_years[:split_idx], x_years[split_idx:]
y_train, y_test = y_values[:split_idx], y_values[split_idx:]

# 初始化并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测 2022–2024
future_years = np.array([[2022], [2023], [2024]])
future_preds = model.predict(future_years)

# 输出预测结果
print("\n预测未来三年 Net Consumption：")
for i, year in enumerate([2022, 2023, 2024]):
    print(f"{year}年预测值：{future_preds[i]:.2f} TWh")

# 可视化预测图
plt.figure(figsize=(10,6))
plt.plot(x_years.flatten(), y_values, label="历史数据", marker='o')
plt.plot(future_years.flatten(), future_preds, label="预测数据", marker='x', linestyle='--', color='red')
plt.xlabel("Year")
plt.ylabel("Net Consumption (TWh)")
plt.title(f"{top_country} Net Consumption Prediction (2022–2024)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("net_consumption_prediction.png", dpi=300)
plt.show()