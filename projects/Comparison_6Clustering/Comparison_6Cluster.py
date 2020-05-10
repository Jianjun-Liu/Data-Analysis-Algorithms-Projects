# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:23:51 2020

@author: CUP
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns
import sklearn.cluster as cluster
import sklearn.mixture as mixture
import time

#%%
plt.close('all')
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 20, 'linewidths':0}

#%%
def generate_data():

    """
    generate 6 groups
    """
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
    X4,y4 = make_blobs(n_samples=500, centers = [[3.5,0]], cluster_std = 0.1, random_state = 12)
    y4 = np.where(y4 == 0,5,0)

    # 将模拟得到的数组转换为数据框，用于绘图
    plot_data = pd.DataFrame(np.row_stack([np.column_stack((X1,y1)),np.column_stack((X2,y2)),np.column_stack((X3,y3)),np.column_stack((X4,y4))]), columns = ['x1','x2','y'])

#    # 绘制散点图（用不同的形状代表不同的簇）#
#    plt.scatter(plot_data.iloc[:,0],plot_data.iloc[:,1],c=plot_data.iloc[:,2],s=12)
#    plt.show()# 显示图形

    return plot_data.values
#%%
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette=sns.color_palette('deep', np.unique(labels).max() + 1)
    colors=[palette[x] if x>=0 else (0.0,0.0,0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    T=end_time - start_time
    plt.title('Clusters by {},'.format(str(algorithm.__name__))+' took {:.2f} s'.format(T),
               fontsize=12)
    plt.tight_layout()

#%%
    ### 数据集1
#data = np.load('clusterable_data.npy')
    ### 生成数据
data=generate_data()
plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)



#plt.figure()
#plot_clusters(data, cluster.KMeans, (), {'n_clusters':6})
#
#plt.figure()
#plot_clusters(data, cluster.AffinityPropagation, (),
#              {'preference':-5.0, 'damping':0.95})
#
#plt.figure()
#plot_clusters(data, cluster.MeanShift, (0.175,), {'cluster_all':False})
#
#plt.figure()
#plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters':6})
#
#plt.figure()
#plot_clusters(data, cluster.AgglomerativeClustering,
#              (), {'n_clusters':6, 'linkage':'ward'})
plt.figure()
plot_clusters(data, cluster.DBSCAN, (), {'eps':0.25})
###
###plt.figure()
###plot_clusters(data, cluster.OPTICS, (), {'eps':0.025})
#
#plt.figure()
#plot_clusters(data, cluster.Birch, (), {'n_clusters':6, 'threshold':0.1})
#
#plt.figure()
#plot_clusters(data, mixture.GaussianMixture, (), {'n_components':6})