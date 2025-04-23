import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

# 读取预处理后的数据
df = pd.read_csv('application_processed.csv')

# 1. 时间特征处理
def process_time_features(df):
    # 转换issue_d为datetime
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    
    # 计算信用历史长度（月份）
    df['credit_history_months'] = ((df['issue_d'] - df['earliest_cr_line']).dt.days / 30).astype(int)
    
    # 提取issue_d的月份和季度信息
    df['issue_month'] = df['issue_d'].dt.month
    df['issue_quarter'] = df['issue_d'].dt.quarter
    
    # 删除原始时间列
    df = df.drop(['issue_d', 'earliest_cr_line'], axis=1)
    
    return df

# 2. 类别特征处理
def process_categorical_features(df):
    # 2.1 处理home_ownership
    # 将NONE和OTHER合并为OTHER（因为样本量很小）
    df['home_ownership'] = df['home_ownership'].replace(['NONE'], 'OTHER')
    
    # 2.2 处理emp_length（已在预处理中完成数值转换）
    
    # 2.3 处理purpose（创建大类别）
    debt_related = ['debt_consolidation', 'credit_card']
    home_related = ['home_improvement', 'house']
    business_related = ['small_business']
    life_related = ['wedding', 'vacation', 'medical', 'moving', 'educational']
    
    def categorize_purpose(purpose):
        if purpose in debt_related:
            return 'debt'
        elif purpose in home_related:
            return 'home'
        elif purpose in business_related:
            return 'business'
        elif purpose in life_related:
            return 'life'
        else:
            return 'other'
    
    df['purpose_category'] = df['purpose'].apply(categorize_purpose)
    
    # 2.4 处理addr_state（按地理区域分组）
    northeast = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA']
    midwest = ['OH', 'MI', 'IN', 'IL', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS']
    south = ['DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA', 'OK', 'TX']
    west = ['MT', 'ID', 'WY', 'CO', 'NM', 'AZ', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI']
    
    df['region'] = df['addr_state'].apply(lambda x: 'northeast' if x in northeast 
                                        else 'midwest' if x in midwest 
                                        else 'south' if x in south 
                                        else 'west')
    
    return df

# 3. 数值特征处理
def process_numeric_features(df):
    # 3.1 创建收入分箱特征
    df['income_level'] = pd.qcut(df['annual_inc'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # 3.2 创建贷款金额与收入比率的分箱
    df['loan_income_ratio_level'] = pd.qcut(df['loan_income_ratio'], q=5, 
                                          labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # 3.3 创建信用利用率特征
    df['credit_utilization'] = df['open_acc'] / df['total_acc']
    
    # 3.4 创建逾期相关组合特征
    df['has_delinq'] = (df['delinq_2yrs'] > 0).astype(int)
    df['has_pub_rec'] = (df['pub_rec'] > 0).astype(int)
    
    # 3.5 创建查询频率特征
    df['inq_last_6mths_per_acc'] = df['inq_last_6mths'] / df['total_acc']
    
    return df

# 4. 特征选择
def select_features(df):
    # 4.1 删除不需要的特征
    columns_to_drop = ['member_id', 'zip_code']  # 删除ID类和邮编
    df = df.drop(columns_to_drop, axis=1)
    
    # 4.2 删除高度相关的特征
    # 保留转换后的特征，删除原始特征
    original_features_to_drop = ['home_ownership', 'purpose', 'addr_state']
    df = df.drop(original_features_to_drop, axis=1)
    
    return df

# 5. 特征标准化
def standardize_features(df):
    # 5.1 获取数值型特征
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # 5.2 标准化数值特征
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df, scaler

# 主处理流程
def engineer_features():
    print("开始特征工程...")
    
    # 1. 读取数据
    df = pd.read_csv('application_processed.csv')
    
    # 2. 时间特征处理
    df = process_time_features(df)
    print("时间特征处理完成")
    
    # 3. 类别特征处理
    df = process_categorical_features(df)
    print("类别特征处理完成")
    
    # 4. 数值特征处理
    df = process_numeric_features(df)
    print("数值特征处理完成")
    
    # 5. 特征选择
    df = select_features(df)
    print("特征选择完成")
    
    # 6. 特征标准化
    df, scaler = standardize_features(df)
    print("特征标准化完成")
    
    # 7. 保存处理后的数据
    df.to_csv('application_engineered.csv', index=False)
    print("特征工程完成，数据已保存至 application_engineered.csv")
    
    return df, scaler

if __name__ == "__main__":
    df, scaler = engineer_features()
    print("\n最终特征维度：", df.shape)