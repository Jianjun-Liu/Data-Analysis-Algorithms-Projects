# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:30:34 2020

@author: CUP
"""

#-------------------------------------
#%% 1.使用python的内置函数open()，会创建一个file对象．使用该对象的方法可以对文件进行操作．
file_handle = open('ChinaWeather.txt',encoding='UTF-8')

index = 0
data=[]
for line in file_handle.readlines(): # 依次读取每一行
    line = line.strip()
    #去掉每行的头尾空白
    list_from_line = line.split(',')
    data.append(list_from_line)
    index +=1

file_handle.close()

#%% 2. 写入txt
file_handle = open('info.txt', 'w')
file_handle.write("index"+","+"City"+","+"Data"+","+"Temprature"+'\n') # 写列名
serise = str(0)+","+"Changping-Beijing"+","+"2020-01-14"+","+str(12)  # 每个元素都是字符串，使用逗号分割拼接成一个字符串
file_handle.write(serise+'\n') # 末尾使用换行分割每一行．
file_handle.close()

#%% 3. loadtxt()
import numpy as np
a = np.loadtxt('info.txt',dtype=str,delimiter=",")#最普通的loadtxt
print(a)


#----------------------------------------
#%%
from sklearn.datasets import load_iris
import csv

#%% 
iris=load_iris() #载入数据集iris
#%% 写入csv文件
f = open('iris.csv','w')#打开iris.csv文件，若不存在，则创建
writer = csv.writer(f, lineterminator='\n')
writer.writerow(iris.feature_names)#写入iris数据标题行
for item in iris.data:
    writer.writerow(item)#写入iris.data中每行数据

f.close()
print("write over")

#读取csv文件
f = open('iris.csv',encoding = 'utf-8') #参数encoding = 'utf-8'防止出现乱码
data=[]
for row in f:
    data.append(row)
    
f.close()