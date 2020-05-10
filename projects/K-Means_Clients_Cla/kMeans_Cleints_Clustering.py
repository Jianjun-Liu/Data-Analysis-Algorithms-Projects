imes# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:19:33 2020

@author: CUP
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')
df = pd.read_csv('Mall_Customers.csv', index_col=0)
df.columns = [ 'Gender', 'Age', 'Income', 'Score']
df.head()


plt.figure()
df.Age.plot(kind='kde', figsize = None, legend=False, style=None, color = "b", alpha = None)
plt.xlabel('Age')
plt.ylabel('KDE')
#
plt.figure()
df.Income.plot(kind='kde', figsize = None, legend=False, style=None, color = "b", alpha = None)
plt.xlabel('Income')
plt.ylabel('KDE')

from sklearn.preprocessing import minmax_scale
x = df[['Income','Score']]
X = minmax_scale(x)
X = pd.DataFrame(X, columns=['Income', 'Score'])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

K = range(2,11)
TSSE = []
score = []
for k in K:
    SSE = []
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    score.append(silhouette_score(X, labels))
    for label in set(labels):
        SSE.append(np.sum((X.loc[label == labels] - centers[label,:])**2))
    TSSE.append(np.sum(SSE))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(K, TSSE, 'b*-')
plt.xlabel(r'$k$')
plt.ylabel('SSE')

plt.subplot(1,2,2)
plt.plot(K, score, 'b*-')
plt.xlabel(r'$k$')
plt.ylabel('silhouette_score')

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
df['cluster'] = kmeans.labels_

plt.figure()
colors='rbgkm'
for i in range(5):
    plt.scatter(df[df.cluster == i]['Income'], df[df.cluster == i]['Score'],color=colors[i])
    plt.scatter(df[df.cluster == i]['Income'].mean(), df[df.cluster == i]['Score'].mean(), marker='*',color=colors[i],s=100)
plt.xlabel('Income')
plt.ylabel('Score')

#%%


