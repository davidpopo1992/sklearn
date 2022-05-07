# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:16:05 2022

@author: davidpopo
"""
#字典的特征提取
from sklearn.feature_extraction import DictVectorizer

data=[{'城市':'上海','平均薪资':9000},{'城市':'无锡','平均薪资':7000},
      {'城市':'昆明','平均薪资':6000}]

vt=DictVectorizer(sparse=False) #不返回稀疏矩阵
#稀疏矩阵好处:将0的位置节省出来,节省内存

data_clear=vt.fit_transform(data)

#查看特征名称
print('查看特征名称: \n',vt.get_feature_names())

print('查看特征提取后的结果: \n',data_clear)