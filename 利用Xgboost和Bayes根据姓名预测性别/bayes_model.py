import pandas as pd
from collections import defaultdict
import math
import numpy as np

# 读取train.txt
train = pd.read_csv('name.csv')
train=train.dropna()
test = pd.read_csv('test.txt')


# 把数据分为男女两部分
names_female = train[train['性别'] == '女']
names_male = train[train['性别'] == '男']
names_men=train
# totals用来存放训练集中女生、男生的总数
totals = {'f': len(names_female),
          'm': len(names_male)}

#分别计算在所有女生（男生）的名字当中，某个字出现的频率。这一步相当于是计算 P(Xi|女生)P(Xi|女生)和P(Xi|男生)
P_Male=len(names_male)/(len(names_female)+len(names_male))
P_Female=len(names_female)/(len(names_female)+len(names_male))


女人
frequency_list_f = defaultdict(int)
names_female_count=0    #字总数
names_female_kinds_count=0   #字种类个数
for name in names_female['姓名']:
    for char in name:
        frequency_list_f[char] += 1
        if frequency_list_f[char]==1:
            names_female_kinds_count+=1
        names_female_count+=1

#男人
names_male_count=0
names_male_kinds_count=0
frequency_list_m = defaultdict(int)
for name in names_male['姓名']:
    for char in name:
        frequency_list_m[char] += 1
        if frequency_list_m[char]==1:
            names_male_kinds_count+=1
        names_male_count+=1


#人类
names_men_count=0
names_men_kinds_count=0
frequency_list_men = defaultdict(int)
for name in names_men['姓名']:
    for char in name:
        frequency_list_men[char] += 1
        if frequency_list_men[char]==1:
            names_men_kinds_count+=1
        names_men_count+=1
for c in frequency_list_men:
    frequency_list_men[c]/=names_men_count
	
	

#虑到预测集中可能会有汉字并没有出现在训练集中，所以我们需要对频率进行Laplace平滑
def LaplaceSmooth(char, frequency_list, total, alpha=1.0):
    count = frequency_list[char] * total
    distinct_chars = len(frequency_list)
    freq_smooth = (count + alpha ) / (total + distinct_chars * alpha)
    return freq_smooth

def tell_Sex(str,alpha=1.0):
    m_rate=P_Male
    f_male=P_Female
    for c in str:
        m_frequence=(frequency_list_m[c] + alpha) / (names_male_count + alpha * names_male_kinds_count)
        f_frequence = (frequency_list_f[c] + alpha) / (names_female_count + alpha * names_female_kinds_count)
        men_frequence= (frequency_list_men[c] + alpha) / (names_men_count + alpha * names_men_kinds_count)
        m_rate*=m_frequence/men_frequence
        f_male*=f_frequence/men_frequence
    return m_rate>f_male
	
	
if __name__ == '__main__':
    #测试输出
    # result = []
    # for name in test['name']:
    #     gender = tell_Sex(name)
    #     result.append(int(gender))
    #
    # submit['gender'] = result
    #
    # submit.to_csv('my_NB_prediction12.csv', index=False)
    while True:
        str=input()
        if str=='0':
            break
        if tell_Sex(str):
            print('男')
        else:
            print('女')