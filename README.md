# 信贷评分卡系统

## 项目概述
本项目是一个基于机器学习的信贷评分卡系统，用于评估借款人的信用风险。系统使用了多种机器学习模型，包括逻辑回归、XGBoost、LightGBM和CatBoost，通过分析借款人的各种特征来预测违约风险。

## 数据集信息
- 样本数量：39,785条记录
- 特征数量：25个
- 目标变量：贷款状态（完全还清/已核销）
- 类别分布：
  - 完全还清：85.75%
  - 已核销：14.25%

## 项目结构
```plaintext
├── data_exploration.py          # 数据探索分析脚本
├── data_preprocessing.py        # 数据预处理脚本
├── feature_engineering.py       # 特征工程脚本
├── model_training.py           # 模型训练脚本
├── eda_analysis.py             # EDA分析脚本
├── application.csv             # 原始数据
├── application_processed.csv    # 预处理后的数据
├── application_engineered.csv   # 特征工程后的数据
└── reports/
    ├── data_exploration_report.md    # 数据探索报告
    ├── feature_definition.md         # 特征定义文档
    ├── model_evaluation_report.md    # 模型评估报告
    └── model_training_analysis.md    # 模型训练分析报告


## 主要特征
1. 借款人基本信息
   
   - 就业年限
   - 住房所有权状态
   - 年收入
   - 验证状态
2. 贷款信息
   
   - 贷款金额：500-35,000元
   - 贷款期限：36个月(73.13%)或60个月(26.87%)
   - 贷款目的：主要用于债务合并、信用卡、房屋装修等
3. 信用相关特征
   
   - 债务收入比(DTI)
   - 过去2年逾期次数
   - 信用查询次数
   - 公共记录数量
   - 破产记录数量
## 模型性能

## 数据预处理
1. 缺失值处理
   
   - 高缺失率特征：创建缺失标记变量
   - 低缺失率特征：使用中位数或众数填充
2. 特征工程
   
   - 对年收入进行对数转换
   - 类别型变量编码处理
   - 创建组合特征
## 使用说明
1. 运行数据探索分析：
```bash
python data_exploration.py
 ```

2. 执行数据预处理：
```bash
python data_preprocessing.py
 ```

3. 进行特征工程：
```bash
python feature_engineering.py
 ```

4. 训练模型：
```bash
python model_training.py
 ```

## 后续优化方向
1. 深入分析特征与目标变量的关系
2. 进行特征重要性分析
3. 考虑添加外部数据源补充信息
4. 优化模型参数
5. 处理类别不平衡问题
## 注意事项
- 数据集存在类别不平衡问题
- 部分特征存在高比例缺失值
- 需要注意处理异常值和极端值