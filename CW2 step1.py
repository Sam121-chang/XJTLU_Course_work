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