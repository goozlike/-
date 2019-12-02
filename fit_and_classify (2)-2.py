#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from scipy.signal import convolve2d

a1 = np.matrix([1, 2, 1])
a2 = np.matrix([-1, 0, 1])
Kx = a1.T * a2
Ky = a2.T * a1

def extract_hog(image, binNumber = 16):
    #Приводиим все изображения к одному размеру (30,30)
    image = image[:,:,0]
    image = resize(image, (30,30))
    
    #Свертка с оператором Собеля
    gx = convolve2d(image, Kx, boundary='symm', mode='same')/255.0 
    gy = convolve2d(image, Ky, boundary='symm', mode='same')/255.0
    
    #Находим модуль градиента и угол 
    mag = np.hypot(gx,gy)
    ang = np.radians((np.arctan2(gy,gx)*180/np.pi)%360)
    
    bins = np.int32(binNumber*ang/(2*np.pi))
    
    bin_cells = bins[:10,:10], bins[10:20,:10], bins[:10,10:20], bins[10:20,10:20], bins[20:,10:20], bins[20:,:10], bins[:10, 20:], bins[10:20, 20:], bins[20:,20:]
    mag_cells = mag[:10,:10], mag[10:20,:10], mag[:10,10:20], mag[10:20,10:20], mag[20:,10:20], mag[20:,:10], mag[:10, 20:], mag[10:20, 20:], mag[20:,20:]
    hists = [np.bincount(b.ravel(), m.ravel(), binNumber) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)    
    hist = np.array(hist,dtype=np.float32)
    hist = hist/(np.linalg.norm(hist))
    
    if (len(hist) > 144):
        hist = hist[:144]
        
    return hist

def fit_and_classify(train_features, train_labels, test_features):
    #наилучшие гиперпараметры получились у нелинейного SVC 
    svc = SVC(C=200, gamma=0.01, kernel='rbf', random_state=17)
    
    #Преобразуем обучающую и тестовую выборку
    #Это значиттельно улучшает качесвто алгоритма на кросс-валидации
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    svc.fit(train_features_scaled, train_labels) #обучаемся на публичной выборке
    y_pred = svc.predict(test_features_scaled) #делаем предсказание 
    return y_pred

