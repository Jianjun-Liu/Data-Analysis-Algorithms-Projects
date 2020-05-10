# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:53:50 2020

@author: CUP
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
#第一步读取数据: 使用read_csv()函数读取csv文件中的数据
df = pd.read_csv('ClientData.csv', index_col=0)
#第二步利用pandas的plot方法绘制折线图
df.plot(x = "Age", y = "Score")
#第三步: 通过plt的show()方法展示所绘制图形
plt.show()