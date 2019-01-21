import pandas as pd
from gensim.models import Word2Vec
import numpy as np


train = pd.read_csv('train.txt')
test = pd.read_csv('test.txt')
texts = list(train['text']) + list(test['text'])

ndims = 50
model = Word2Vec(sentences=texts, size=ndims, window=5)

print(len(model.wv['之']))


texts = list(train['text']) 
ndims = 50
model = Word2Vec(sentences=texts, size=ndims, window=5)

total = len(texts)
print(total)
vecs = np.zeros([total, ndims])
for i, sentence in enumerate(texts):
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
    vecs[i, :] = row / counts

# 加入PCA将10维变量降维成2维

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
vecs_pca = pca.fit(vecs).transform(vecs)

# vecs_pca

#%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
# plt.axis([-1, 1.5, 1.5, 3])
colors = list(map(lambda x: 'red' if x == 1 else 'blue', train['y']))
plt.scatter(vecs_pca[:, 0], vecs_pca[:, 1], c=colors, alpha=0.8, s=30, lw=0)
print('Word2Vec: 白话文(蓝色)与文言文(红色)')
plt.show()




texts = list(train['text']) 
ndims = 2
model = Word2Vec(sentences=texts, size=ndims, window=5)

total = len(texts)
print(total)
vecs = np.zeros([total, ndims])
for i, sentence in enumerate(texts):
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
    vecs[i, :] = row / counts

%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
# plt.axis([-1, 1.5, 1.5, 3])
colors = list(map(lambda x: 'red' if x == 1 else 'blue', train['y']))
plt.scatter(vecs[:, 0], vecs[:, 1], c=colors, alpha=1, s=30, lw=0)
print('Word2Vec: 白话文(蓝色)与文言文(红色)')
plt.show()






