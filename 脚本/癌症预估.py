# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:41:25 2022

@author: davidpopo
"""

import pandas as pd

column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']

data=pd.read_csv(r'data\breast-cancer-wisconsin_data.txt',names=column_name)
print(data.head(5))

#%%
#处理缺失值
import numpy as np
data=data.replace('?',np.nan)
data.dropna(inplace=True)
print(data.isnull().any())

#%%
#划分特征和标签
x=data.iloc[:,1:-1]
y=data['Class']
print(x.head(5))

#%%
#特征工程
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=22)
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#l2正则，惩罚系数
estimator=LogisticRegression(penalty='l2', C = 2)
estimator.fit(x_train,y_train)

#%%
#模型评估
y_predict=estimator.predict(x_test)
print('算法分类结果',y_predict==y_test)
print('模型评分',estimator.score(x_test,y_test))
print('模型权重为',estimator.coef_)
print('模型偏置为',estimator.intercept_)

#%%
#模型评估_准确率，召回率，F1系数
from sklearn.metrics import classification_report
report=classification_report(y_test, y_predict, labels=[2,4], target_names=['良性','恶性'] )
print(report)

#%%
from sklearn.metrics import roc_auc_score
y_test=np.where(y_test>3,1,0)
y_predict=np.where(y_predict>3,1,0)
auc=roc_auc_score(y_test,y_predict)
print(auc)