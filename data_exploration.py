import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 读取数据
df = pd.read_csv('application.csv')

# 1. 基本信息探查
print("="*50)
print("1. 数据基本信息")
print("="*50)
print("\n数据维度：", df.shape)
print("\n数据类型信息：")
print(df.info())

# 2. 缺失值分析
print("\n"+"="*50)
print("2. 缺失值分析")
print("="*50)
missing_values = df.isnull().sum()
missing_values_percent = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({
    '缺失值数量': missing_values,
    '缺失值占比(%)': missing_values_percent
})
print("\n存在缺失值的列：")
print(missing_info[missing_info['缺失值数量'] > 0])

# 3. 数值型变量统计描述
print("\n"+"="*50)
print("3. 数值型变量统计描述")
print("="*50)
print(df.describe())

# 4. 类别型变量统计
print("\n"+"="*50)
print("4. 类别型变量统计")
print("="*50)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"\n{col} 的唯一值数量：{df[col].nunique()}")
    print(f"{col} 的值分布：")
    print(df[col].value_counts().head())

# 5. 可视化分析
plt.figure(figsize=(15, 10))

# 5.1 贷款金额分布
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='loan_amnt', bins=50)
plt.title('贷款金额分布')

# 5.2 年收入分布（取对数）
plt.subplot(2, 2, 2)
sns.histplot(data=df, x=np.log(df['annual_inc']), bins=50)
plt.title('年收入分布(对数)')

# 5.3 贷款目的分布
plt.subplot(2, 2, 3)
df['purpose'].value_counts().head(10).plot(kind='bar')
plt.title('贷款目的Top10分布')
plt.xticks(rotation=45)

# 5.4 住房所有权状态分布
plt.subplot(2, 2, 4)
df['home_ownership'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('住房所有权状态分布')

plt.tight_layout()
plt.savefig('data_exploration.png')
plt.close()

# 6. 相关性分析
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('数值变量相关性热力图')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()