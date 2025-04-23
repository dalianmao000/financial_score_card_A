# 模型评估报告

## 1. 模型性能比较

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|--------|--------|--------|--------|-----|
| LogisticRegression | 0.8615 | 0.8639 | 0.9962 | 0.9254 | 0.6990 |
| XGBoost | 0.8557 | 0.8665 | 0.9843 | 0.9216 | 0.6635 |
| LightGBM | 0.8604 | 0.8630 | 0.9962 | 0.9248 | 0.6901 |
| CatBoost | 0.8604 | 0.8639 | 0.9948 | 0.9247 | 0.6893 |

## 2. 特征重要性分析


### LogisticRegression - Top 10 重要特征

注：对于逻辑回归模型，系数为正表示该特征与目标变量正相关（增加违约风险），系数为负表示该特征与目标变量负相关（降低违约风险）。

| 特征 | 重要性/系数 |
|------|--------|
| annual_inc_log | 0.2843 |
| pub_rec | 0.2017 |
| has_last_delinq | 0.1674 |
| purpose_category | 0.1357 |
| loan_amnt | 0.0896 |
| pub_rec_bankruptcies | 0.0824 |
| addr_state_encoded | 0.0661 |
| delinq_pub_rec_zero | 0.0575 |
| issue_month | 0.0352 |
| open_acc | 0.0249 |

### XGBoost - Top 10 重要特征

| 特征 | 重要性/系数 |
|------|--------|
| term | 0.0872 |
| int_rate | 0.0660 |
| purpose_category | 0.0430 |
| has_last_delinq | 0.0428 |
| has_last_record | 0.0399 |
| delinq_pub_rec_zero | 0.0338 |
| annual_inc | 0.0336 |
| inq_last_6mths | 0.0319 |
| mths_since_last_record | 0.0313 |
| purpose_encoded | 0.0300 |

### LightGBM - Top 10 重要特征

| 特征 | 重要性/系数 |
|------|--------|
| int_rate | 268.0000 |
| title | 239.0000 |
| credit_history_months | 235.0000 |
| dti | 230.0000 |
| annual_inc | 223.0000 |
| acc_ratio | 203.0000 |
| loan_income_ratio | 192.0000 |
| addr_state_encoded | 152.0000 |
| inq_last_6mths_per_acc | 132.0000 |
| loan_amnt | 126.0000 |

### CatBoost - Top 10 重要特征

| 特征 | 重要性/系数 |
|------|--------|
| int_rate | 12.8376 |
| title | 7.3080 |
| credit_history_months | 6.1165 |
| addr_state_encoded | 5.7694 |
| dti | 5.6665 |
| loan_income_ratio | 5.2772 |
| loan_amnt | 3.8638 |
| total_acc | 3.8412 |
| mths_since_last_delinq | 3.7613 |
| issue_month | 3.4886 |
