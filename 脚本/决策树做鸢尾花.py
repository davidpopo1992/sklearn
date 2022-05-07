# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:43:13 2022

@author: davidpopo
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris=load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)
tree=DecisionTreeClassifier()

tree.fit(x_train,y_train)

#params = {"criterion": ['gini','entropy'],'max_depth':[3,4,5,6,7,8]}
#model=GridSearchCV(tree,param_grid=params,cv=10)
#model.fit(x_train,y_train)

#模型评估
#y_predict=model.predict(x_test)
#print('KNN算法分类结果',y_predict==y_test)
#print('模型评分',model.score(x_test,y_test))
#print('最佳参数',model.best_params_)
#print('最佳分数',model.best_score_)
#print('最佳模型',model.best_estimator_)

#%%
export_graphviz(tree, out_file="./tree.dot", feature_names=iris.feature_names)