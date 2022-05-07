# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:15:36 2022

@author: davidpopo
"""
import pandas as pd

data=pd.read_csv(r'data/titanic.csv')
print(data.head())
print(data.columns)

#%%
#特征标签分离
#x=data[['pclass', 'name', 'age', 'embarked', 'home.dest', 'room', 'ticket', 'boat', 'sex']]
x=data[['pclass','age','sex']]
y=data['survived']
print(y.value_counts())

#%%
#数据处理-缺失值处理
x['age'].fillna(x['age'].mean(),inplace=True)

#%%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

#将特征转化为特征字典
x=x.to_dict(orient="records")
print(x[:10])

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=22)

transfer=DictVectorizer()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

forest=RandomForestClassifier()
params={"n_estimators": list(range(100,500,100)),'max_depth':list(range(5,20,1))}
model=GridSearchCV(forest,param_grid=params,cv=10)
model.fit(x_train,y_train)

#%%
#评估模型
y_predict=model.predict(x_test)
print('随机森林算法分类结果',y_predict==y_test)
print('模型评分',model.score(x_test,y_test))
print('最佳参数',model.best_params_)
print('最佳分数',model.best_score_)
print('最佳模型',model.best_estimator_)