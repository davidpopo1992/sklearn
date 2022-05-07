# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:19:54 2022

@author: davidpopo
"""

import pandas as pd

aisles=pd.read_csv(r'data/instacart/aisles.csv')
print('aisles表:\n',aisles.head(5))

order_products=pd.read_csv(r'data/instacart/order_products__prior.csv')
print('order_products__prior表:\n',order_products.head(5))

orders=pd.read_csv(r'data/instacart/orders.csv')
print('orders:\n',orders.head(5))

products=pd.read_csv(r'data/instacart/products.csv')
print('orders:\n',products.head(5))

#%%
print(orders.columns)
print(products.columns)

#%%
#合并数据
df1=pd.merge(orders,order_products,left_on='order_id',right_on='order_id',how='inner')
df2=pd.merge(df1,products,left_on='product_id',right_on='product_id',how='inner')
df3=pd.merge(df2,aisles,left_on='aisle_id',right_on='aisle_id',how='inner')

print('数据合并后表格列:\n',df3.columns)

#%%
data=pd.crosstab(df3['user_id'],df3['aisle'])
print(data.head(5))
#%%
print(data.shape)
print('用户数%d'%data.shape[0])
print('商品数%d'%data.shape[1])
#%%
#pca降维
from sklearn.decomposition import PCA

transfer=PCA(n_components=0.95)
result=transfer.fit_transform(data)
#print(result[:10])
print(result.shape)
print('原来的特征数是%d,主成分分析后成功降维到%d'%(data.shape[1],result.shape[1]))

#%%
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

kmeans=KMeans()
params = {"n_clusters": range(3,9,1)}
estimator=GridSearchCV(kmeans,param_grid=params,cv=10)
estimator.fit(result)

#模型保存
joblib.dump(estimator, r'模型/kmeans.pkl')
#%%
#模型评估
y_predict=estimator.predict(result)
from sklearn.metrics import silhouette_score
sci_score=silhouette_score(result,y_predict)
print('聚类结果sci系数为:\n',sci_score)