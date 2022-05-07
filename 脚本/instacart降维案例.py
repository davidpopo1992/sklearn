# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:11:07 2022

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
