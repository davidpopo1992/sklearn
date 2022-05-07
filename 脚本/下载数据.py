# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:31:13 2022

@author: davidpopo
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris=load_iris()  #加载出的数据是字典格式
print('查看特征值样式: \n',iris.data[:10])
print('查看特征值结构: \n',iris.data.shape)
print('查看标签值样式: \n',iris.target[:10])
print('查看标签值结构: \n',iris.target.shape)
print('查看特征值名称: \n',iris.feature_names)
print('查看标签值名称: \n',iris.target_names)

#分割训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=42)
print('查看训练特征值样式: \n',x_train[:10])


#%%
help(train_test_split)