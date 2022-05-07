# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:38:31 2022

@author: davidpopo
"""

import pandas as pd

#%%
f = open(r'E:\学习\算法\Machine_Learning\Machine_Learning\resources\FBlocation\train.csv')
data = pd.read_csv(f)
print(data.head(5))

#%%
#数据预处理
#筛选数据
print('数据量为:',len(data))
data_1=data.query('x>2&x<2.5&y>1&y<1.5')
print('数据量为:',len(data_1))

#%%
#只筛选地点出现次数大于20的数据
df_place=data_1.groupby('place_id').agg({'row_id':'count'}).rename(
        columns={'row_id':'place_num'}).reset_index()

df_place_need=df_place[df_place['place_num']>20]

data_2=data_1[data_1.place_id.isin(df_place_need.place_id)]
print('数据量为:',len(data_2))

#%%
#时间戳处理
date=pd.to_datetime(data_2['time'])
print(date)

date_index=pd.DatetimeIndex(date)

data_2['day']=date_index.day
data_2['weekday']=date_index.weekday
data_2['hour']=date_index.hour

#%%
#分成特征数据和标签数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

x_data=data_2[['x','y','accuracy','day','weekday','hour']]
y_data=data_2['place_id']
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data)

#标准化
transfer=StandardScaler()
x_train2=transfer.fit_transform(x_train)
x_test2=transfer.transform(x_test)

#模型
knn=KNeighborsClassifier()
params = {"n_neighbors": [3,5,7,9,11]}
model=GridSearchCV(knn,param_grid=params,cv=10)
model.fit(x_train2,y_train)

#模型评估
y_predict=model.predict(x_test2)
print('预测正确性:',y_predict==y_test)
print('模型预测结果:',model.score(x_test2,y_test))
print('最佳参数',model.best_params_)
print('最佳分数',model.best_score_)
#%%