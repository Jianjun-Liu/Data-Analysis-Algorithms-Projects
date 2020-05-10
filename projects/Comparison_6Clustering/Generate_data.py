# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 22:56:39 2020

@author: CUP
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入第三方模块
from sklearn.datasets import make_moons,make_blobs,make_circles
import pandas as pd

# 构造非球形样本点
X1,y1 = make_moons(n_samples=2000, noise = 0.05, random_state = 1)
# 构造球形样本点
X2,y2 = make_blobs(n_samples=1000, centers = [[3,3]], cluster_std = 0.5, random_state = 1234)
# 将y2的值替换为2(为了避免与y1的值冲突，因为原始y1和y2中都有0这个值)
y2 = np.where(y2 == 0,2,0)

X3,y3 = make_circles(n_samples=1000, factor=0.5, noise=0.08 , random_state = 0)
# 将y2的值替换为2(为了避免与y1的值冲突，因为原始y1和y2中都有0这个值)
y3 = np.where(y3 == 0,3,4)
X3[:,1]=X3[:,1]+3

# 构造球形样本点
X4,y4 = make_blobs(n_samples=500, centers = [[3.5,0]], cluster_std = 0.3, random_state = 12)
y4 = np.where(y4 == 0,5,0)

# 将模拟得到的数组转换为数据框，用于绘图
plot_data = pd.DataFrame(np.row_stack([np.column_stack((X1,y1)),np.column_stack((X2,y2)),np.column_stack((X3,y3)),np.column_stack((X4,y4))]), columns = ['x1','x2','y'])

# 绘制散点图（用不同的形状代表不同的簇）

plt.scatter(plot_data.iloc[:,0],plot_data.iloc[:,1],c=plot_data.iloc[:,2],s=12)
# 显示图形
plt.show()

data1=[plot_data.iloc[:,0],plot_data.iloc[:,1]]
