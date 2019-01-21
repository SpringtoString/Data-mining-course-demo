
# coding: utf-8

# In[1]:


import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA

train = pd.read_csv('train.txt')
test = pd.read_csv('test.txt')
texts = list(train['text']) + list(test['text'])


# In[2]:


ndims = 50
model = Word2Vec(sentences=texts, size=ndims, window=5)

train_texts = list(train['text']) 
total = len(train_texts)
train_vecs = np.zeros([total, ndims])

for i, sentence in enumerate(train_texts):
    counts, row = 0, 0
    for char in sentence:
        try:
            if char != ' ':
                row += model.wv[char]
                counts += 1
        except:
            pass
    if counts == 0:
        print(sentence)
    train_vecs[i, :] = row / counts
    


test_texts = list(test['text']) 
total = len(test_texts)
test_vecs = np.zeros([total, ndims])

for i, sentence in enumerate(test_texts):
    counts, row = 0, 0
    for char in sentence:
        try:
            if char != ' ':
                row += model.wv[char]
                counts += 1
        except:
            pass
    if counts == 0:
        print(sentence)
    test_vecs[i, :] = row / counts

pca = PCA(n_components=2)
train_vecs = pca.fit(train_vecs).transform(train_vecs)
pca = PCA(n_components=2)
test_vecs = pca.fit(test_vecs).transform(test_vecs)


# In[3]:


import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=80, reg_alpha=0, reg_lambda=0.,
    max_depth=2, n_estimators=100, objective='binary',
    subsample=1, colsample_bytree=1, subsample_freq=1,
    learning_rate=0.1, random_state=2018
)
lgb_model.fit(train_vecs,train['y'])
res=lgb_model.predict_proba(test_vecs)[:, 1]


# In[4]:


submit = pd.read_csv('sample_submit.csv')

submit['y'] =res
submit.to_csv('my_prediction.csv', index=False)

