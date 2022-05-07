# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:42:46 2022

@author: davidpopo
"""

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba

def text_demo():
    data=['i love python,python makes me happy','u can never believe in python,python makes me angry']
    transfer=CountVectorizer()
    result=transfer.fit_transform(data)
    print(transfer.get_feature_names())
    print('文本特征提取结果: \n',result.toarray())

#中文文本自动分词
def cut(text):
    result=" ".join(list(jieba.cut(text)))
    return result

def chinese_text_demo():
    data=['因为python是一门有趣的语言，所以需要学习它','python不擅长做开发，我没不要学习它',
          '我不仅想学习python，我还想学习除了python之外的其他语言']
    data_new=[cut(i) for i in data]
    transfer=CountVectorizer(stop_words=['因为','所以'])
    result=transfer.fit_transform(data_new)
    print(transfer.get_feature_names())
    print('文本特征提取结果: \n',result.toarray())

#tfidf特征提取
def tfidf_text_demo():
    data=['因为python是一门有趣的语言，所以需要学习它','python不擅长做开发，我没不要学习它',
          '我不仅想学习python，我还想学习除了python之外的其他语言']
    data_new=[cut(i) for i in data]
    transfer=TfidfVectorizer(stop_words=['因为','所以'])
    result=transfer.fit_transform(data_new)
    print(transfer.get_feature_names())
    print('文本特征提取结果: \n',result.toarray())




if __name__=='__main__':
#    text_demo()
#    chinese_text_demo()
    tfidf_text_demo()
    