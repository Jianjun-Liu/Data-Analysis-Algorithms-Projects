# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 00:42:57 2020

@author: CUP
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(101)
n = 800
m = 50
k = 50
t = np.random.random(size=n) * 2 * np.pi -np.pi
x1 = np.cos(t)
x2 = np.sin(t)

for i in range(n):
    len = np.sqrt(np.random.random())
    x1[i] = x1[i] * len
    x2[i] = x2[i] * len
#------------------------------------------------------
s = np.random.random(size=m) * 2 * np.pi -np.pi
x3 = np.cos(s)
x4 = np.sin(s)
for j in range(m):
    len = np.sqrt(np.random.random())/5
    x3[j] = x3[j] * len + 0.85
    x4[j] = x4[j] * len + 0.85
#------------------------------------------------
w = np.random.random(size=k) * 2 * np.pi -np.pi
x5 = np.cos(w)
x6 = np.sin(w)
for r in range(k):
    len = np.sqrt(np.random.random())/5
    x5[r] = x5[r] * len - 0.85
    x6[r] = x6[r] * len + 0.85
#------------------已经生成数据，下面是数据整合并存储-------------------
X1 = np.zeros((n,3))
for i in range(n):
    X1[i,0] = x1[i]
    X1[i,1] = x2[i]
    X1[i,2] = 0
X2 = np.zeros((m,3))
for j in range(m):
    X2[j,0] = x3[j]
    X2[j,1] = x4[j]
    X2[j,2] = 1
X3 = np.zeros((k,3))
for z in range(k):
    X3[z,0] = x5[z]
    X3[z,1] = x6[z]
    X3[z,2] = 2

X = np.vstack((X1,X2,X3))
data = pd.DataFrame(X)
data.to_csv(r'\bolbs_hard.csv',index=None,header=None)



# plt.figure(1)
plt.scatter(x1,x2,marker='*',c='r',linewidths=0.0005)
plt.scatter(x3,x4,marker='o',c='b',linewidths=0.0005)
plt.scatter(x5,x6,marker='o',c='b',linewidths=0.0005)


plt.show()
