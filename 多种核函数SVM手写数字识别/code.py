from sklearn.svm import SVC
from sklearn.datasets import load_digits
# 从通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中。
import numpy as np
mnist = load_digits()

# 从sklearn.cross_validation中导入train_test_split用于数据分割。
from sklearn.cross_validation import train_test_split
 
# 随机选取75%的数据作为训练样本；其余25%的数据作为测试样本。
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2)

# 从sklearn.preprocessing里导入数据标准化模块。
from sklearn.preprocessing import StandardScaler
 
stander= StandardScaler()
stander.fit_transform(X_train)
# X_train = stander.fit_transform(X_train)
# X_test = stander.transform(X_test)

X_train = stander.transform(X_train)
X_test = stander.transform(X_test)


#model1
model1=SVC(C=1, kernel='rbf', degree=3, gamma='auto',
                 coef0=0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=2018)
model1.fit(X_train,y_train)
predict_vaule=model1.predict(X_test)
corrcect=np.equal(predict_vaule,y_test) # type:np.ndarray
corrcect=corrcect.astype(dtype=int)
print('model1 accuracy is ',np.mean(corrcect))


#model2
model2=SVC(C=1, kernel='linear', degree=3, gamma='auto',
                 coef0=0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=2018)
model2.fit(X_train,y_train)
predict_vaule=model2.predict(X_test)
corrcect=np.equal(predict_vaule,y_test) # type:np.ndarray
corrcect=corrcect.astype(dtype=int)
print('model2 accuracy is ',np.mean(corrcect))


#model3
model3=SVC(C=1, kernel='poly', degree=2, gamma='auto',
                 coef0=0.1, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=2018)
model3.fit(X_train,y_train)
predict_vaule=model3.predict(X_test)
corrcect=np.equal(predict_vaule,y_test) # type:np.ndarray
corrcect=corrcect.astype(dtype=int)
print('model3 accuracy is ',np.mean(corrcect))

#model4
model4=SVC(C=1, kernel='sigmoid', degree=3, gamma='auto',
                 coef0=-1, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=2018)
model4.fit(X_train,y_train)
predict_vaule=model4.predict(X_test)
corrcect=np.equal(predict_vaule,y_test) # type:np.ndarray
corrcect=corrcect.astype(dtype=int)
print('model4 accuracy is ',np.mean(corrcect))