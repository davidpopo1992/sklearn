# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:57:14 2022

@author: davidpopo
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def iris_analysis():
    #加载数据
    iris=load_iris()
    #分割训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=21)
    #标准化
    transfer=StandardScaler()
    x_train_std=transfer.fit_transform(x_train)
    x_test_std=transfer.transform(x_test)
    #训练KNN模型
    knn=KNeighborsClassifier()
#    knn.fit(x_train_std,y_train)
    #调参
    params = {"n_neighbors": [3,5,7,9,11]}
    model=GridSearchCV(knn,param_grid=params,cv=10)
    model.fit(x_train_std,y_train)
    
    
    #模型评估
    y_predict=model.predict(x_test_std)
    print('KNN算法分类结果',y_predict==y_test)
    print('模型评分',model.score(x_test_std,y_test))
    print('最佳参数',model.best_params_)
    print('最佳分数',model.best_score_)
    print('最佳模型',model.best_estimator_)



if __name__=='__main__':
    iris_analysis()