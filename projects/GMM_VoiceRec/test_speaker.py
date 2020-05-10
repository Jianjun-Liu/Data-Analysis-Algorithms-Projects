#test_gender.py
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")
import time


# 训练数据路径
source   = "development_set\\"   
modelpath = "speaker_models\\"
test_file = "development_set_test.txt"        
file_paths = open(test_file,'r')

gmm_files = [os.path.join(modelpath,fname) for fname in 
              os.listdir(modelpath) if fname.endswith('.gmm')]

# 加载高斯模型
models    = [pickle.load(open(fname,'rb+')) for fname in gmm_files]

speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname 
              in gmm_files]

# 读取测试目路径获得测试音频文件列表
for path in file_paths:     
    path = path.strip()   
    print(path)
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)
    
    log_likelihood = np.zeros(len(models)) 
    
    for i in range(len(models)):
        gmm    = models[i]  #逐一检测每个模型
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
    
    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])
    time.sleep(1.0)


