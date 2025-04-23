import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据准备
def prepare_data():
    # 读取特征工程后的数据
    df = pd.read_csv('application_engineered.csv')
    
    # 检查并转换所有对象类型的列
    object_columns = df.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in object_columns:
        if col != 'loan_status':  # 排除目标变量
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    
    # 将目标变量转换为数值型
    le = LabelEncoder()
    if 'loan_status' in df.columns:
        df['loan_status_encoded'] = le.fit_transform(df['loan_status'])
    
    # 分离特征和目标变量
    X = df.drop(['loan_status_encoded', 'loan_status'] if 'loan_status' in df.columns else ['loan_status_encoded'], axis=1)
    y = df['loan_status_encoded']
    
    # 保存特征名称
    feature_names = X.columns
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 添加标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 将标准化后的数据转换回DataFrame以保留列名
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# 2. 模型评估函数
def evaluate_model(y_true, y_pred, y_pred_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

# 3. 训练和评估所有模型
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }
    
    results = {}
    feature_importance = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 评估
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        results[name] = metrics
        
        # 特征重要性（如果模型支持）
        if name == 'LogisticRegression':
            # 对于逻辑回归，保留系数的正负值
            feature_importance[name] = pd.Series(
                model.coef_[0],  # 不取绝对值，保留正负
                index=X_train.columns
            ).sort_values(ascending=False)
        elif hasattr(model, 'feature_importances_'):
            feature_importance[name] = pd.Series(
                model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=False)
    
    return results, feature_importance, models  # 返回models

# 4. 可视化结果
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def visualize_results(results, feature_importance, models, X_test, y_test):
    # 4.1 每个评估指标的模型对比
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metrics_names = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
    
    plt.figure(figsize=(15, 10))
    for i, (metric, metric_name) in enumerate(zip(metrics, metrics_names), 1):
        plt.subplot(2, 3, i)
        metric_values = [results[model][metric] for model in results.keys()]
        plt.bar(list(results.keys()), metric_values)
        plt.title(f'{metric_name}对比', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.2 绘制ROC曲线对比图
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('ROC曲线对比')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4.3 特征重要性可视化（横向柱状图）
    for name, importance in feature_importance.items():
        plt.figure(figsize=(12, 8))
        top_20 = importance.head(20).sort_values(ascending=True)
        plt.barh(range(len(top_20)), top_20.values)
        plt.yticks(range(len(top_20)), top_20.index)
        plt.title(f'{name} - 特征重要性 Top 20', fontsize=12)
        plt.xlabel('重要性', fontsize=10)
        plt.ylabel('特征', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

# 5. 生成评估报告
def generate_report(results, feature_importance):
    with open('model_evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write('# 模型评估报告\n\n')
        
        # 5.1 模型性能比较
        f.write('## 1. 模型性能比较\n\n')
        f.write('| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |\n')
        f.write('|------|--------|--------|--------|--------|-----|\n')
        
        for model_name, metrics in results.items():
            f.write(f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} |\n")
        
        # 5.2 特征重要性分析
        f.write('\n## 2. 特征重要性分析\n\n')
        for model_name, importance in feature_importance.items():
            f.write(f'\n### {model_name} - Top 10 重要特征\n\n')
            if model_name == 'LogisticRegression':
                f.write('注：对于逻辑回归模型，系数为正表示该特征与目标变量正相关（增加违约风险），系数为负表示该特征与目标变量负相关（降低违约风险）。\n\n')
            f.write('| 特征 | 重要性/系数 |\n')
            f.write('|------|--------|\n')
            for feature, score in importance.head(10).items():
                f.write(f'| {feature} | {score:.4f} |\n')

if __name__ == "__main__":
    print("开始模型训练和评估...")
    
    # 1. 准备数据
    X_train, X_test, y_train, y_test = prepare_data()
    print("数据准备完成")
    
    # 2. 训练和评估模型
    results, feature_importance, models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    print("模型训练和评估完成")
    
    # 3. 可视化结果
    visualize_results(results, feature_importance, models, X_test, y_test)
    print("可视化完成")
    
    # 4. 生成报告
    generate_report(results, feature_importance)
    print("评估报告生成完成")