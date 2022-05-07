# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:06:52 2022

@author: davidpopo
"""

from sklearn.datasets import load_boston

x=load_boston().data
y=load_boston().target
#print(x[:10],y[:10])
print(load_boston().feature_names)
#%%
#训练模型
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=22)
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

estimator=SGDRegressor()
estimator.fit(x_train,y_train)

#%%
#模型评估
from sklearn.metrics import mean_squared_error
y_predict=estimator.predict(x_test)
print('均方误差为\n',mean_squared_error(y_test,y_predict))
print('模型权重为',estimator.coef_)
print('模型偏置为',estimator.intercept_)
