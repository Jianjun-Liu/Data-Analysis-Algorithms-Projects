# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:53:20 2019

@author: CUPBs
"""
import matplotlib.pyplot as plt #用于画图工具
plt.rcParams['font.sans-serif'] = ['SimHei']#SimHei是黑体的意思
plt.rcParams['axes.unicode_minus'] = False#avoid negtive symbol
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
import time


def img2vector(filename):
    """
    函数说明:将32x32的二进制图像转换为1x1024向量
    """
    #创建1x1024零向量
    returnVect = np.zeros((1, 1024))
    #打开文件
    fr = open(filename)
    #按行读取
    for i in range(32):
        #读一行数据
        lineStr = fr.readline()
        #每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect


def KNN_handwritingClassTest(digitsPath):
    '''
    函数说明:手写数字分类测试
    输入：训练和测试数据文件夹所在目录
    输出：对指定的测试集文件，指定的训练集数据进行K近邻分类，并打印结果信息
    '''
    start_time = time.time()
    #训练集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir('./'+digitsPath+'/trainingDigits/')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,训练集
    trainingMat = np.zeros((m, 1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumber)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector('./'+digitsPath+'/trainingDigits/%s' % (fileNameStr))
    #构建kNN分类器
    neigh =KNeighborsClassifier(n_neighbors = 3, algorithm = 'auto')
    #拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    #返回testDigits目录下的文件列表
    testFileList = listdir('./'+digitsPath+'/testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行 分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #(filenameStr去掉.txt后缀),获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector('./'+digitsPath+'/testDigits/%s' % (fileNameStr))
        #获得预测结果
        classifierResult = neigh.predict(vectorUnderTest)
        end_time = time.time()
        if(classifierResult != classNumber):
            errorCount += 1.0
            print('分错的为：' + fileNameStr+'，分类结果为 %d\t真实结果为 %d'% (classifierResult, classNumber)  )
    T=end_time - start_time
    print('整个识别过程共用时%2.6f s；共误识别了 %d 个数据；错误率为 %f' % (T, errorCount, errorCount/mTest * 100))

"""
执行分类
"""
KNN_handwritingClassTest('digits')

