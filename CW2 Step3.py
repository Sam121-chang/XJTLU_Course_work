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