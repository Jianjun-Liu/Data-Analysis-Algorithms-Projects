# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:30:18 2020

@author: CUP
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pathlib import Path
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage import io,transform


def load_image_files(container_path, dimension=(32, 32)):
    """
    Load image files with categories as subfolder names
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = io.imread(file)
            img_resized = transform.resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    '''在Python开发中，经常将配置文件以 json 的形式写在文件中Bunch可以将配置文件转换
    为配置类和配置字典。
    '''
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

if __name__ == '__main__':
    #%% Image files used are from https://github.com/Abhishek-Arora/Image-Classification-Using-SVM
    image_dataset = load_image_files("images/")

    X_train, X_test, y_train, y_test = train_test_split(
        image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

    param_grid = [
      {'C': [1, 5,10, 100], 'kernel': ['linear']},
      {'C': [1, 5,10, 100], 'gamma': [0.01, 0.001], 'kernel': ['rbf']},
     ]
    svc = SVC()
    #%% GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出相对最优化的结果和参数。
        ### 但是这个方法适合于小数据集，一旦数据的量级上去了，很难得出结果。
    clf = GridSearchCV(svc, param_grid)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification report for - \n{}:\n{}\n".format(
        clf, metrics.classification_report(y_test, y_pred)))
    print("Confision Matrix is - \n{}:".format(metrics.confusion_matrix(y_test, y_pred),
          label=['dog','dollar','pizza','soccer','sunflower']))
    # 显示分类错误图像
    err=[y_pred!=y_test]
    t=0
    for k in range(len(y_pred)):
        if err[0][k]==True:
            t+=1;
#            plt.figure()
#            io.imshow(image_dataset.images[k,:,:])
            print('第 %s 个图像 %s 被错误识别为 %s . '%(k,image_dataset.target_names[y_test[k]],image_dataset.target_names[y_pred[k]]))
    print('\n共 %s 个图像错误识别'%t)
