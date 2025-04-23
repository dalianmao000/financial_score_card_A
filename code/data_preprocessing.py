import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. 读取数据
df = pd.read_csv('application.csv')
print("原始数据维度：", df.shape)

# 2. 缺失值处理
def handle_missing_values(df):
    # 2.1 对高缺失率特征创建标记变量
    df['has_last_record'] = df['mths_since_last_record'].notna().astype(int)
    df['has_last_delinq'] = df['mths_since_last_delinq'].notna().astype(int)
    
    # 2.2 填充数值型特征的缺失值
    numeric_features = ['mths_since_last_record', 'mths_since_last_delinq', 'pub_rec_bankruptcies']
    for feature in numeric_features:
        df[feature] = df[feature].fillna(0)
    
    # 2.3 填充类别型特征的缺失值
    df['emp_length'] = df['emp_length'].fillna('Unknown')
    df['title'] = df['title'].fillna('Other')
    # desc字段缺失率高且为描述性文本，可以删除
    df = df.drop('desc', axis=1)
    
    return df

# 3. 特征转换
def transform_features(df):
    # 3.1 对年收入进行对数转换
    df['annual_inc_log'] = np.log(df['annual_inc'])
    
    # 3.2 处理emp_length，转换为数值
    emp_length_map = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10,
        'Unknown': -1
    }
    df['emp_length_num'] = df['emp_length'].map(emp_length_map)
    
    # 3.3 处理int_rate（去除%符号并转换为float）
    df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float)
    
    # 3.4 处理term（提取月数）
    df['term'] = df['term'].str.extract('(\d+)').astype(int)
    
    return df

# 4. 类别型变量编码
def encode_categorical_features(df):
    categorical_features = ['home_ownership', 'verification_status', 'purpose', 
                          'addr_state', 'loan_status']
    
    # 使用LabelEncoder进行编码
    le_dict = {}
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature + '_encoded'] = le.fit_transform(df[feature])
        le_dict[feature] = le
    
    return df, le_dict

# 5. 创建新特征
def create_new_features(df):
    # 5.1 债务收入比的分箱
    df['dti_bins'] = pd.qcut(df['dti'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    
    # 5.2 贷款金额与年收入的比率
    df['loan_income_ratio'] = df['loan_amnt'] / df['annual_inc']
    
    # 5.3 信用账户总数与开放账户数的比率
    df['acc_ratio'] = df['open_acc'] / df['total_acc']
    
    # 5.4 逾期和记录的组合特征
    df['delinq_pub_rec_zero'] = ((df['delinq_2yrs'] == 0) & (df['pub_rec'] == 0)).astype(int)
    
    return df

# 主处理流程
def preprocess_data():
    # 1. 读取数据
    df = pd.read_csv('application.csv')
    print("开始数据预处理...")
    
    # 2. 缺失值处理
    df = handle_missing_values(df)
    print("缺失值处理完成")
    
    # 3. 特征转换
    df = transform_features(df)
    print("特征转换完成")
    
    # 4. 类别型变量编码
    df, le_dict = encode_categorical_features(df)
    print("类别型变量编码完成")
    
    # 5. 创建新特征
    df = create_new_features(df)
    print("新特征创建完成")
    
    # 6. 保存处理后的数据
    df.to_csv('application_processed.csv', index=False)
    print("预处理完成，数据已保存至 application_processed.csv")
    
    return df, le_dict

if __name__ == "__main__":
    df, le_dict = preprocess_data()
    print("\n处理后数据维度：", df.shape)