# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:10:46 2022

@author: davidpopo
"""

from sklearn.datasets import fetch_20newsgroups

data=fetch_20newsgroups(subset='all')
print(data[:10])

#%%
#分割训练集和测试集
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

x_train,x_test,y_train,y_test=train_test_split(data.data,data.target)

#特征抽取
transfer=TfidfVectorizer()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

print(transfer.get_feature_names())

#建立模型
nb=MultinomialNB()
nb.fit(x_train,y_train)

#模型评估
y_predict=nb.predict(x_test)
print('预测正确性:',y_predict==y_test[:100])
print('模型评分:',nb.score(x_test,y_test))


#%%