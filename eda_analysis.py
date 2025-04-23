import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

# 读取预处理后的数据
df = pd.read_csv('application_processed.csv')

# 1. 目标变量分析
plt.figure(figsize=(10, 6))
loan_status_dist = df['loan_status_encoded'].value_counts()
plt.pie(loan_status_dist, labels=['已还清', '已违约'], autopct='%1.1f%%')
plt.title('贷款状态分布')
plt.savefig('loan_status_distribution.png')
plt.close()

# 2. 数值特征分析
def analyze_numeric_features(df):
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # 2.1 相关性分析
    plt.figure(figsize=(15, 12))
    correlation = df[numeric_features].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('数值特征相关性热力图')
    plt.tight_layout()
    plt.savefig('numeric_correlation.png')
    plt.close()
    
    # 2.2 数值特征分布
    n_features = len(numeric_features)
    n_rows = (n_features + 2) // 3
    plt.figure(figsize=(15, 5*n_rows))
    
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(n_rows, 3, i)
        sns.histplot(data=df, x=feature, bins=50)
        plt.title(f'{feature}分布')
    
    plt.tight_layout()
    plt.savefig('numeric_distributions.png')
    plt.close()
    
    # 2.3 生成数值特征统计报告
    stats_report = df[numeric_features].describe()
    stats_report.to_csv('numeric_stats_report.csv')

# 3. 类别特征分析
def analyze_categorical_features(df):
    # 获取类别特征，并排除desc和title
    categorical_features = df.select_dtypes(include=['object']).columns
    categorical_features = [col for col in categorical_features if col not in ['desc', 'title']]
    
    plt.figure(figsize=(15, 5*len(categorical_features)))
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(len(categorical_features), 1, i)
        value_counts = df[feature].value_counts()
        sns.barplot(x=value_counts.index, y=value_counts.values)
        plt.title(f'{feature}分布')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('categorical_distributions.png')
    plt.close()

# 4. 特征与目标变量的关系分析
def analyze_feature_target_relationship(df):
    # 4.1 数值特征与目标变量的关系
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 5*len(numeric_features)))
    
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(len(numeric_features), 1, i)
        sns.boxplot(x='loan_status', y=feature, data=df)
        plt.title(f'{feature}与贷款状态的关系')
    
    plt.tight_layout()
    plt.savefig('numeric_vs_target.png')
    plt.close()
    
    # 4.2 类别特征与目标变量的关系（排除desc和title）
    categorical_features = df.select_dtypes(include=['object']).columns
    categorical_features = [col for col in categorical_features if col not in ['desc', 'title']]
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        contingency_table = pd.crosstab(df[feature], df['loan_status'])
        contingency_table.plot(kind='bar', stacked=True)
        plt.title(f'{feature}与贷款状态的关系')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'categorical_{feature}_vs_target.png')
        plt.close()

# 5. 生成EDA报告
def generate_eda_report(df):
    with open('eda_report.md', 'w', encoding='utf-8') as f:
        f.write('# 信贷数据EDA分析报告\n\n')
        
        # 5.1 基本信息
        f.write('## 1. 数据基本信息\n')
        f.write(f'- 样本数量：{len(df)}\n')
        f.write(f'- 特征数量：{df.shape[1]}\n\n')
        
        # 5.2 缺失值信息
        f.write('## 2. 缺失值信息\n')
        missing_info = df.isnull().sum()
        missing_info = missing_info[missing_info > 0]
        f.write(missing_info.to_string())
        f.write('\n\n')
        
        # 5.3 数值特征统计
        f.write('## 3. 数值特征统计\n')
        numeric_stats = df.describe()
        f.write(numeric_stats.to_string())
        f.write('\n\n')
        
        # 5.4 类别特征统计（排除desc和title）
        f.write('## 4. 类别特征统计\n')
        categorical_features = df.select_dtypes(include=['object']).columns
        categorical_features = [col for col in categorical_features if col not in ['desc', 'title']]
        
        for feature in categorical_features:
            f.write(f'\n### {feature}分布\n')
            f.write(df[feature].value_counts().to_string())
            f.write('\n')

if __name__ == "__main__":
    print("开始EDA分析...")
    
    # 执行各项分析
    analyze_numeric_features(df)
    print("数值特征分析完成")
    
    analyze_categorical_features(df)
    print("类别特征分析完成")
    
    analyze_feature_target_relationship(df)
    print("特征与目标变量关系分析完成")
    
    generate_eda_report(df)
    print("EDA报告生成完成")
    
    print("\nEDA分析完成，请查看生成的图表和报告文件：")
    print("1. loan_status_distribution.png - 贷款状态分布")
    print("2. numeric_correlation.png - 数值特征相关性热力图")
    print("3. numeric_distributions.png - 数值特征分布图")
    print("4. categorical_distributions.png - 类别特征分布图")
    print("5. numeric_vs_target.png - 数值特征与目标变量关系图")
    print("6. categorical_*_vs_target.png - 类别特征与目标变量关系图")
    print("7. numeric_stats_report.csv - 数值特征统计报告")
    print("8. eda_report.md - 完整EDA分析报告")