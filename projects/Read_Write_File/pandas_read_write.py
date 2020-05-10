# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:36:57 2020

@author: CUP
"""
import pandas as pd

#%% 1. pandas read_table & read_csv
df1 = pd.read_table("test.txt", sep=' ')
print('\nread_table读取：\n',df1)
df2 = pd.read_csv("test.txt", sep=' ')
print('\n read_csv读取：\n',df2)

#%% 1. pandas 两种数据格式
s = pd.Series([1,2,3],index=['a','b','c'])
print(s)

d = pd.DataFrame([[1,2,3],[4,5,6]], columns=['a','b','c'])
print(d)

#%% 2. 写入文件
data_set=[[1, '张三','200010134','数学与应用数学','湖南省'],
      [2, '李四','2020010235','统计学','河北省'],
      [3, '张三','2020010336','应用化学','上海市']]
data = pd.DataFrame(columns=("ID", "姓名", "学号",'专业', "籍贯"))
for i in range(len(data_set)):
    data.loc[i] = data_set[i]      #逐行写入
data.to_csv(path_or_buf="students_info.csv", index=False) #写入csv 文件
