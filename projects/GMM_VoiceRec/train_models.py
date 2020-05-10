#train_models.py

import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture 
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")


#训练数据路径
source   = "development_set\\"   

#训练演讲人的存储路径
dest = "speaker_models\\"

train_file = "development_set_enroll.txt"        

file_paths = open(train_file,'r')

count = 1
# 提取每个演讲人的特征（每人5段音频）
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print(path)
    
    # 读取音频
    sr,audio = read(source + path)
    
    # 提取40维的 MFCC & delta MFCC特征
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # 当演讲人的5个文件的特征被合并后，开始训练模型
    if count == 5:    
        gmm = GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        # 将对象训练后的高斯模型保存到文件中
        picklefile = path.split("-")[0]+".gmm"
        pickle.dump(gmm,open(dest + picklefile,'wb')) # 将gmm对象的pickle编码表示写入到文件对象中
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)    
        features = np.asarray(())
        count = 0
    count = count + 1
    