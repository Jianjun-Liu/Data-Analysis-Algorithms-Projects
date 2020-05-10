# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:59:05 2020

@author: CUP
"""

import matplotlib.pyplot as plt #用于画图工具
plt.rcParams['font.sans-serif'] = ['SimHei']#SimHei是黑体的意思
plt.rcParams['axes.unicode_minus'] = False#avoid negtive symbol
import seaborn as sns
import pandas as pd


file ="Online Retail_Small.xlsx"

df = pd.read_excel('Online Retail_Small.xlsx')
df.head()

#%% 判断是否有重复值。
import numpy as np

df.duplicated(subset = ['StockCode'],keep='last')#判断是否有重复值
print("共有%s行重复"%np.sum(df.duplicated()))#查看有多少行重复
np.sum(df.duplicated(subset=['StockCode']))#查看某一列有多少重复值
df.drop_duplicates(inplace=True)#删除重复行

df.drop_duplicates(subset=['StockCode'],keep='first')#根据列重复进行删除


#%% 缺失值的处理
df.isnull()#缺失值判断
np.sum(df.isnull(),axis=0)#缺失值计数 默认沿着行操作，对列进行统计
df.apply(lambda x: sum(x.isnull())/len(x),axis=0)#计算缺失值比例
## (1)删除
#df.dropna(how='any')#删除行数据缺失值：all - 全部，any- 任意
#df.head()

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
## 填充缺失值 
#5.1  分类数据用众数填补
df.Description.fillna(df.Description.mode()[0])
#5.2 数值型数据用均值，中位数填补
df.CustomerID.fillna(df.CustomerID.mean())
df.CustomerID.fillna(df.CustomerID.median())
df.CustomerID.fillna(1000)
#5.3 不同值采取不同处理方式-字典形式填补
df.fillna(value={'CustomerID':df.CustomerID.mean(),
                  'Description':df.Description.mode()[0]})
#5.4 前向填补和后向填补
df.fillna(method='ffill')         #第一个数据缺失无法填充
df.fillna(method='bfill')        #最后一个数据缺失无法填充

#%% 异常值处理
#df['price'] = df['Quantity']*df['UnitPrice']#计算每项交易总交易额
#q_mean = df['price'].mean()
#q_std = df['price'].std()
#any(df['price']>q_mean+2*q_std)
#any(df['price']<q_mean-2*q_std)
#
##2.1 下四分位数
#Q1=df['price'].quantile(0.25)
##2.2 上四分位数
#Q3=df['price'].quantile(0.75)
##2.3 分位差
#IQR= Q3-Q1
#
#print(any(df['price']>Q3 + 1.5*IQR))
#
#print(any(df['price']>Q1 - 1.5*IQR))
#
##2.4 箱线图
#plt.figure(figsize=(6.4,4.8))
#df['price'].plot(kind="box")
#
##3. 异常值处理 - 替换值
#UL = Q3+0.5*IQR
##3.1 计算数据在正常范围内的最大值
#replace_value  = df['price'][df['price']<UL].max()    
#print("异常值用%.4f替换"%replace_value)
##3.2 进行离群点的替换
#print("异常值为",df.loc[df['price']<UL,'price'])
#df.loc[df['price']<UL,'price'] = replace_value
#df['price'].describe()

#%% 
df = df[~df['Description'].str.contains('POSTAGE')]

#%% 
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(df).transform(df)#可以通过inverse_transform() 进行反编码

basket = (df[df['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.9)
rules.sort_values(by='leverage',ascending=False,inplace=True)    #关联规则可以按leverage排序
# print(association_rule)


rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]

