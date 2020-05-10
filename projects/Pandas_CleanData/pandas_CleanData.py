# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:15:58 2019

@author: CUP
"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']#SimHei是黑体的意思
plt.rcParams['axes.unicode_minus'] = False#avoid negtive symbol
import numpy as np
import pandas as pd

#%% 1. 查看了解数据
df = pd.read_csv('./patient_heart_rate.csv',encoding='utf-8')
df.head()

#%% 2. 清洗数据
# 增加列头
column_names= ['id', 'name', 'gender','age', 'weight','m0006','m0612','m1218',
                'f0006','f0612','f1218']
df = pd.read_csv('./patient_heart_rate.csv', names = column_names)
df.head()

#%% 将列表拆成新的列，再将原来的 Name 列删除。
# 切分名字，删除源数据列
df[['first_name','last_name']] = df['name'].str.split(expand=True)
df.drop('name', axis=1, inplace=True)

#%%  Weight 列的单位不统一。有的单位是 kgs，有的单位是 lbs.
# 获取 weight 数据列中单位为 lbs 的数据
rows_with_lbs = df['weight'].str.contains('lbs').fillna(False)
df[rows_with_lbs]

# 将 lbs 的数据转换为 kgs 数据
for i,lbs_row in df[rows_with_lbs].iterrows():
    weight = int(float(lbs_row['weight'][:-3])/2.2)
    df.at[i,'weight'] = '{}kgs'.format(weight)
#%% 删除空行.
# 查找空行.all; .any查找空元素
print(df[df.isnull().T.all()])
# 删除全空的行
df.dropna(how='all',inplace=True)

#%% 处理非 ASCII 数据方式有多种，包括删除、替换。下面使用删除的方式：
# 删除非 ASCII 字符
df['first_name'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)
df['last_name'].replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

#%%  查看缺失情况
df.apply(lambda col:sum(col.isnull())/col.size)