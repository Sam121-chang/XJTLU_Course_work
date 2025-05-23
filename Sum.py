import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# 打开选择文件窗口（不会弹出空白窗口）
Tk().withdraw()
file_path = askopenfilename(title="请选择 GlobalElectricityStatistics.csv 文件")

# 读取数据
electricity_data = pd.read_csv(file_path)

# 显示前后五行
print("前5行：")
print(electricity_data.head())

print("\n后5行：")
print(electricity_data.tail())


# 查看基本信息
print("\n数据结构信息：")
electricity_data.info()

# 查看统计信息
print("\n数据描述性统计：")
print(electricity_data.describe())


# 缺失值检测
print("\n每列缺失值数量：")
print(electricity_data.isnull().sum())

# 丢弃含缺失值的行（根据需要也可以使用填补法）
electricity_data_cleaned = electricity_data.dropna()

# 检查处理后结果
print("\n清洗后缺失值数量：")
print(electricity_data_cleaned.isnull().sum())


# 保存清洗后的数据
electricity_data_cleaned.to_csv("electricity_data_cleaned.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt

# 读取清洗后的数据
electricity_data = pd.read_csv("electricity_data_cleaned.csv")

# 清洗空格：防止匹配错误
electricity_data['Features'] = electricity_data['Features'].str.strip().str.lower()
electricity_data['Country'] = electricity_data['Country'].str.strip()

# 选定国家（现在可以写正常名称）
selected_countries = ['China', 'United States', 'Ireland', 'South Africa', 'India']

# 筛选 Feature 为 net generation 的数据
filtered_data = electricity_data[
    (electricity_data['Features'] == 'net generation') &
    (electricity_data['Country'].isin(selected_countries))
    ]

# 年份列
years = [str(year) for year in range(1980, 2022)]
x = [int(year) for year in years]

# 创建图表
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

for i, country in enumerate(selected_countries):
    country_data = filtered_data[filtered_data['Country'] == country]

    if country_data.empty:
        print(f"警告：{country} 没有 net generation 数据，跳过。")
        continue

    y = country_data[years].values.flatten().astype(float)

    plt.plot(
        x, y,
        marker='o',
        linestyle='-',
        linewidth=2,
        color=colors[i],
        label=country
    )

# 图表标题和标签
plt.title('Electricity Generation Trends (1980–2021)', fontsize=14, pad=20)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Net Generation (TWh)', fontsize=12)

# 图例
plt.legend(title='Countries', title_fontsize=12, bbox_to_anchor=(1, 1), loc='upper left')

# X轴每5年一个刻度
plt.xticks(range(1980, 2022, 5))

# 网格线
plt.grid(linestyle='--', alpha=0.6, color='gray')

# 自动调整布局防止遮挡
plt.tight_layout()

# 保存图像
plt.savefig('trend_comparison.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# 读取清洗后的数据
electricity_data = pd.read_csv("electricity_data_cleaned.csv")

# 清洗字段，防止匹配出错
electricity_data['Country'] = electricity_data['Country'].str.strip()
electricity_data['Features'] = electricity_data['Features'].str.strip().str.lower()

# 选定国家
selected_countries = ['China', 'United States', 'Ireland', 'South Africa', 'India']

# 筛选 Feature 为 net generation 且国家在列表中
filtered_data = electricity_data[
    (electricity_data['Features'] == 'net generation') &
    (electricity_data['Country'].isin(selected_countries))
    ]

# 选取 2000–2021 年的列（22年）
year_cols = [str(year) for year in range(2000, 2022)]

# 创建一个国家对应平均发电量的字典
average_generation = {}

for country in selected_countries:
    country_data = filtered_data[filtered_data['Country'] == country]

    if country_data.empty:
        print(f"警告：{country} 无 net generation 数据，跳过")
        continue

    values = country_data[year_cols].values.flatten().astype(float)
    avg = values.mean()
    average_generation[country] = avg
    print(f"{country} 年均发电量（2000–2021）：{avg:.2f} TWh")

# 找出最高和最低国家
max_country = max(average_generation, key=average_generation.get)
min_country = min(average_generation, key=average_generation.get)
print(f"\n最高发电国家：{max_country}（{average_generation[max_country]:.2f} TWh）")
print(f"最低发电国家：{min_country}（{average_generation[min_country]:.2f} TWh）")

# 柱状图可视化
plt.figure(figsize=(8, 6))
plt.bar(
    average_generation.keys(),
    average_generation.values(),
    color=['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
)

# 图表设置
plt.title('Average Electricity Generation (2000–2021)', fontsize=14, pad=15)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Average Net Generation (TWh)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存图像
plt.savefig('average_generation_bar.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

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
