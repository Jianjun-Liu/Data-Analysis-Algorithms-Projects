# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:41:22 2019

@author: CUPBs
"""
import matplotlib.pyplot as plt #用于画图工具
plt.rcParams['font.sans-serif'] = ['SimHei']#SimHei是黑体的意思
plt.rcParams['axes.unicode_minus'] = False#avoid negtive symbol
import seaborn as sns
import pandas as pd

#%% 导入数据
data = pd.read_csv('cardio_train.csv')
data.head()

#%% 将年龄的单位转换为年，并进行四舍五入保留整数.
data['age'] = round(data['age']/365).astype(int)
fig = plt.figure(figsize = (6,4))
plt.hist(data['age'],bins=15)
plt.title("年龄分布")
plt.show()

#%% 剔除异常值,身高低于140cm且体重低于40kg则认为是异常值，按照这个方法从数据中剔除异常数据．
data=data[(data['height']>=140)&(data['weight']>=40)]
print(data.shape)

#%% 高低血压的变化范围，我们将收缩压ap\_hi变化区间限定为[60,250]，将舒张压ap\_lo的变化区间限定为[30,180]
data = data[(60<=data['ap_hi'] )&(data['ap_hi']<=250)
            &(30<=data ['ap_lo'])&(data['ap_lo']<=180)]

#%% 使用直方图观察4个离散型取值特征（年龄age、性别gender、胆固醇水平cholesterol、血糖浓度gluc）的分布情况．
col = data.columns[[1,2,7,8]].tolist()
fig,ax = plt.subplots(2,2,figsize= (15,8))
title=['年龄age','性别gender','胆固醇水平cholesterol','血糖浓度gluc']
for i in range(len(col)):
    plt.subplot( "22"+str(i+1))
    plt.hist(data[col[i]])
    plt.xlabel(col[i],fontsize =15)
    plt.ylabel("Count",fontsize =13)
    plt.title(title[i])
plt.tight_layout()
plt.show()

#%% 核密度曲线观察连续性特征的数值分布情况．
col = data.columns[[3,4,5,6]].tolist()
fig,ax=plt.subplots(2,2,figsize= (15,8))
title=['身高','体重','收缩压','舒张压']
for i in range(len(col)):
    plt.subplot( '22'+str(i+1))
    x = data[col[i]]
    sns.distplot(x,color ='lightpink')
    plt.title(title[i])
plt.tight_layout()
plt.show()

##%% 将年龄和性别进行分组，探索不同分组的患病情况．
fig, [axl,ax2] =plt.subplots(1,2,figsize= (15,5))
sns.countplot(x='age', hue= 'cardio', data = data, palette="Set2" ,ax=axl)
sns.countplot(x='gender', hue= 'cardio', data = data, palette="Set2",ax=ax2)
axl.set_xlabel('年龄',fontsize = 12)
ax2.set_xlabel('性别',fontsize = 12)
axl.set_title( '年龄与患病情况分布',fontsize = 12)
ax2.set_title( '性别与患病情况分布',fontsize = 12)
axl.legend(['不患病','患病' ],fontsize = 12)
ax2.legend(['不患病','患病' ],fontsize = 12)
plt.show()

##%% 分类建模
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

x = data.drop(['id',"cardio"] ,axis = 1)
y = data["cardio"]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.2,
                                                    random_state = 2)

#%%  随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, max_depth = 20,
        max_features = 10, random_state = 20).fit(x_train,y_train)
y_pred = rf.predict(x_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print("随机森林的准确率为 %.5f" %(accuracy_score(y_test,y_pred)))

#%% ## 梯度提升树
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                  max_depth=4, random_state=10).fit(x_train, y_train)
y_pred = gbdt.predict(x_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))
print( "GBDT的准确率为 % .5f"%(accuracy_score(y_test,y_pred)))

#%% 特征重要性
figure, [axl,ax2] = plt.subplots(1,2,figsize=(12,6))
rf_importance = rf.feature_importances_
index = data.drop(['id', 'cardio'], axis=1).columns
rf_feature_importance = pd. DataFrame(rf_importance.T, index=index,columns=['score']).sort_values(by='score',
                                              ascending=True)
# 水平条形图绘制
rf_feature_importance.plot(kind='barh', title='随机森林特征重要性',
                           legend=False, ax=axl)

gbdt_importance = gbdt.feature_importances_
index = data.drop(['id',"cardio"], axis=1).columns
gbdt_feature_importance = pd.DataFrame(gbdt_importance.T, index=index,columns=['score']).sort_values(by='score', ascending=True)

# 水平条形图绘制
gbdt_feature_importance.plot(kind='barh', title='GBDT特征重要性',legend=False, ax=ax2)
plt.show()