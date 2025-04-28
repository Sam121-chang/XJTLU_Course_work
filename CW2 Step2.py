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