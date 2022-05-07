# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 09:49:21 2022

@author: davidpopo
"""

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


#归一化
def data_demo():
    data=pd.read_csv(r'data/datingTestSet2.txt',sep='\t')
    x_data=data.iloc[:,:3]
    transfer=MinMaxScaler()
    result=transfer.fit_transform(x_data)
    print('归一化后数据为 \n',result)
    
#标准化
def data_demo2():
    data=pd.read_csv(r'data/datingTestSet2.txt',sep='\t')
    x_data=data.iloc[:,:3]
    transfer=StandardScaler()
    result=transfer.fit_transform(x_data)
    print('标准化后数据为 \n',result)

#方差过滤去除冗余变量
def variance_demo():
    data=pd.read_csv(r'data\factor_returns.csv')
    data_new=data.iloc[:,1:10]
    transfer=VarianceThreshold(threshold=10)
    result=transfer.fit_transform(data_new)
    print('过滤后数据为\n',result)
    print('过滤后数据尺寸为\n',result.shape)

#计算变量之间相关性
def pearsonr_demo():
    data=pd.read_csv(r'data\factor_returns.csv')
    data_new=data.iloc[:,1:10]
#    print(data_new.columns)
    #绘制散点图
    plt.figure(figsize=(20, 8), dpi=100)
    plt.scatter(data_new['revenue'],data_new['total_expense'])
    plt.show()
    result=pearsonr(data_new['revenue'],data_new['total_expense'])
    print('相关系数为:\n',result[0])

#主成分分析
def pca_demo():
    data=pd.read_csv(r'data\factor_returns.csv')
    data_new=data.iloc[:,1:10]
    transfer=PCA(n_components=6)  #传入整数表示降低到多少维度，传入小数表示保留多少信息
    result=transfer.fit_transform(data_new)
    print('过滤后数据为\n',result)
    print('过滤后数据尺寸为\n',result.shape)
    


if __name__=='__main__':
#    data_demo()
#    data_demo2()
#    variance_demo()
    pca_demo()
    
#%%
print(200000/12)
print(12000*16)                                                                                               l1