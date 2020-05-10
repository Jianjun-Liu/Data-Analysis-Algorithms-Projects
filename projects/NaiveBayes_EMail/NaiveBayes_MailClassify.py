# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 13:54:30 2019

@author: CUP
"""
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']#SimHei是黑体的意思
plt.rcParams['axes.unicode_minus'] = False#avoid negtive symbol
import numpy as np

def createVocabList(dataSet):
    """
    获取所有单词的集合
    :param dataSet: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)  
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList)# [0,0......]
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def splitString(string):
    '''
    将一个长的字符串进行分词的操作
    :param string: 需要切分的字符串    
    :return: 切分后字符串中的单词列表
    '''
    import re # 正则表达式
    regEx = re.compile('\\W+')
    listOfTokens = regEx.split(string)
    return listOfTokens

def classifyNBayes(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    使用算法：
        # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C)-> log(P(F1|C))+log(P(F2|C))+
        ....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),
        log(P(F4|C0)),log(P(F5|C0))....] 列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),
        log(P(F4|C1)),log(P(F5|C1))....] 列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 大家可能会发现，上面的计算公式，没有除以贝叶斯准则的公式的分母，也就是 P(w) （P(w) 指的是此文档在所有的文档中出现的概率）就进行概率大小的比较了，
    # 因为 P(w) 针对的是包含侮辱和非侮辱的全部文档，所以 P(w) 是相同的.
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推.
    # 这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1) # P(w|c1) * P(c1) ，即贝叶斯准则的分子
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1) # P(w|c0) * P(c0) ，即贝叶斯准则的分子·
    if p1 > p0:
        return 1
    else:
        return 0

def trainNBayes(trainMatrix, trainCategory):
    """
    训练数据优化版本
    :param trainMatrix: 文件单词矩阵
    :param trainCategory: 文件对应的类别
    :return:
    """   
    numTrainDocs = len(trainMatrix) #总样本（文件）数
    numWords = len(trainMatrix[0])# 总特征（单词）数
    pAbusive = sum(trainCategory) / float(numTrainDocs)# 侮辱性文件的出现概率
    # 构造单词出现次数列表：p0Num 正常的统计；p1Num 侮辱的统计
    p0Num = np.ones(numWords)#[0,0......]->[1,1,1,1,1.....]
    p1Num = np.ones(numWords)
    # 整个数据集单词出现总数：p0Denom 正常的统计；p1Denom 侮辱的统计
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:            
            p1Num += trainMatrix[i]#累加辱骂词的频次
            
            p1Denom += sum(trainMatrix[i])#每篇文章的辱骂的频次进行统计汇总
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的    
    p1Vect = np.log(p1Num/p1Denom)
    # 类别0，即正常文档的   
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def spamTest():
    # spam email classify
    '''
    使用朴素贝叶斯进行交叉验证
    文件解析及完整的垃圾邮件测试函数
    '''
    #我们将得到一个没有重复单词的单词库（834个不重复的单词），
    #生成这个单词库的目的是为了对每一个样本都生成一个1*834的0矩阵，然后对样本里的单词在单词库中进行检索，
    #如果在单词库中，那么把对应位置的0置为1，即把单词样本转化为了一个代表单词样本的0，1矩阵。
    #我们还把所有的样本加载到了trainData中，其对应的标签加载到了trainLabelInput中。
    fullTest=[];docList=[];classList=[]
    for i in range(1,26): #it only 25 doc in every class
        wordList=splitString(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)#为生成库做准备
        classList.append(1)#垃圾邮件标志为1
        wordList=splitString(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullTest.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList) #生成了一个无重复单词的单词库
    trainSet=list(range(50));testSet=[]
    
    #采用交叉验证法，随机的挑选10个样本作为验证集，剩余的15个样本作为训练集。通过训练将得到类概率密度，和先验概率
    for i in range(10):
        randIndex=int(np.random.uniform(0,len(trainSet)))#num in 0-49
        testSet.append(trainSet[randIndex])#挑选出了测试集
        del(trainSet[randIndex])#删除测试集的索引号
        
    trainMat=[];trainClass=[]
    for docIndex in trainSet:#生成了训练矩阵和训练标签
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0,p1,pSpam=trainNBayes(np.array(trainMat),np.array(trainClass))#得到训练数据
    
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNBayes(np.array(wordVector),p0,p1,pSpam)!= classList[docIndex]:
            errorCount += 1
            print(("分类错误邮件内容："), docList[docIndex])  
            
    print('识别错误数: ', errorCount)    
    print('测试集数目 :', len(testSet))
    print('错误率 :',float(errorCount)/len(testSet))
    
if __name__ == '__main__':
    #下面的为了演示分词的，可注释
    path='email/ham/1.txt'
    print('对'+path+'进行分词：')
    listWord=splitString(open(path).read())
    print(listWord)
    
    spamTest()