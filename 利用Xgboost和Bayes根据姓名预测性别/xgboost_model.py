import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier

# 读取数据
train = pd.read_table('train.txt', ',')
test = pd.read_table('test.txt', ',')
submit = pd.read_csv('sample_submit.csv')

# 所有男生的名字
train_male = train[train['gender'] == 1]
m_cnt = len(train_male)
names_male = "".join(train_male['name'])

# 所有女生的名字
train_female = train[train['gender'] == 0]
f_cnt = len(train_female)
names_female = "".join(train_female['name'])

# 统计每个字在男生、女生名字中出现的总次数
lists_male = map(lambda x: x.encode('utf-8'), names_male)
counts_male = Counter(lists_male)
lists_female = map(lambda x: x.encode('utf-8'), names_female)
counts_female = Counter(lists_female)

# 得到训练集中每个人的每个字的词频
train_encoded = []
for i in range(len(train)):
    name = train.at[i, 'name']
    chs = list(map(lambda x: x.encode('utf-8'), name))
    row = [0., 0., 0., 0, train.at[i, 'gender']]
    for j in range(len(chs)):
        row[2* j] = counts_female[chs[j]] * 1. / f_cnt
        row[2* j + 1] = counts_male[chs[j]] * 1. / m_cnt
    train_encoded.append(row)

# 得到测试集中每个人的每个字的词频（Term Frequency，通常简称TF）
test_encoded = []
for i in range(len(test)):
    name = test.at[i, 'name']
    chs = list(map(lambda x: x.encode('utf-8'), name))
    row = [0., 0., 0., 0.,]
    for j in range(len(chs)):
        try:
            row[2 * j] = counts_female[chs[j]] * 1. / f_cnt
        except:
            pass
        try:
            row[2 * j + 1] = counts_male[chs[j]] * 1. / m_cnt
        except:
            pass
    test_encoded.append(row)

# 转换为pandas.DataFrame的形式
# 1_f是指这个人的第一个字在训练集中所有女生的字中出现的频率
# 2_f是指这个人的第二个字在训练集中所有女生的字中出现的频率
# 1_m是指这个人的第一个字在训练集中所有男生的字中出现的频率
# 2_m是指这个人的第二个字在训练集中所有男生的字中出现的频率
train_encoded = pd.DataFrame(train_encoded, columns=['1_f', '1_m', '2_f', '2_m', 'gender'])
test_encoded = pd.DataFrame(test_encoded, columns=['1_f', '1_m', '2_f', '2_m'])




# 得到训练集中每个人的每个字的词频
train_encoded = []
for i in range(len(train)):
    name = train.at[i, 'name']
    chs = list(map(lambda x: x.encode('utf-8'), name))
    row = [0., 0., 0., 0, train.at[i, 'gender']]
    for j in range(len(chs)):
        row[2* j] = counts_female[chs[j]] * 1. / f_cnt
        row[2* j + 1] = counts_male[chs[j]] * 1. / m_cnt
    train_encoded.append(row)

# 得到测试集中每个人的每个字的词频（Term Frequency，通常简称TF）
test_encoded = []
for i in range(len(test)):
    name = test.at[i, 'name']
    chs = list(map(lambda x: x.encode('utf-8'), name))
    row = [0., 0., 0., 0.,]
    for j in range(len(chs)):
        try:
            row[2 * j] = counts_female[chs[j]] * 1. / f_cnt
        except:
            pass
        try:
            row[2 * j + 1] = counts_male[chs[j]] * 1. / m_cnt
        except:
            pass
    test_encoded.append(row)

# 转换为pandas.DataFrame的形式
# 1_f是指这个人的第一个字在训练集中所有女生的字中出现的频率
# 2_f是指这个人的第二个字在训练集中所有女生的字中出现的频率
# 1_m是指这个人的第一个字在训练集中所有男生的字中出现的频率
# 2_m是指这个人的第二个字在训练集中所有男生的字中出现的频率
train_encoded = pd.DataFrame(train_encoded, columns=['1_f', '1_m', '2_f', '2_m', 'gender'])
test_encoded = pd.DataFrame(test_encoded, columns=['1_f', '1_m', '2_f', '2_m'])




# 训练xgboost

import xgboost


clf = xgboost.XGBClassifier(max_depth=4, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=-1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=0.8, colsample_bytree=0.9, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0)
clf.fit(train_encoded.drop('gender', axis=1), train_encoded['gender'])
preds = clf.predict(test_encoded)

# 输出预测结果至my_TF_GBDT_prediction.csv
submit['gender'] = np.array(preds)
submit.to_csv('my_TF_xgboost_prediction.csv', index=False)