# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:44:12 2020

@author: CUP
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
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



if __name__ == '__main__':

    image_dir = Path("E:/360Cloud/Latex_Book/EXPERIMENT/Fig/SVM_IMAGES/")
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    dimension=(64, 64)

    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = io.imread(file)
            img_resized = transform.resize(img, dimension, anti_aliasing=True, mode='reflect')
            images.append(img_resized)
            plt.figure()
            io.imshow(img)



