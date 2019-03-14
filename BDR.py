# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:47:56 2019

@author: SzuHaoWu
"""

import numpy as np
import cv2
import math
from scipy.stats import multivariate_normal
def pdf(x,mean,cov):

    (m,n,l) = x.shape
    out = np.zeros([m,n])
    cov_inv = np.linalg.pinv(cov)
    cov_det = np.linalg.det(cov)
    for i in range(m):
        for j in range(n):
            diff = x[i][j]-mean
            sc = 1/math.sqrt(2*math.pi*cov_det)
            ex = np.exp((-0.5)*np.dot(np.dot(diff,cov_inv),(np.array([diff]).T)))
            out[i][j] = sc*ex
    return out
    
#%%
meanblue = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/meanblue.npy')
meannotblue = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/meannotblue.npy')
meanbar = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/meanbar.npy')
meanother = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/meanother.npy')
covblue = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/covblue.npy')
covnotblue = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/covnotblue.npy')
covbar = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/covbar.npy')
covother = np.load('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/covother.npy')

prior_blue = 0.25
prior_notblue = 0.75
prior_bar = 0.5
prior_other = 0.5

result_mask = np.zeros((800,1200))

#%%

img = cv2.imread('C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/trainset/18.png')
(m,n,l) = img.shape
pro_blue = multivariate_normal.pdf(img,meanblue,covblue)
pro_notblue = multivariate_normal.pdf(img,meannotblue,covnotblue)
pro_bar = multivariate_normal.pdf(img,meanbar,covbar)
pro_other = multivariate_normal.pdf(img,meanother,covother)
 
conditionA = prior_blue*pro_blue > prior_notblue*pro_notblue
conditionB = prior_bar*pro_bar > prior_other*pro_other 

for i in range(m):
    for j in range(n):
        if conditionA[i][j] == True:
            if conditionB[i][j] == True:
                result_mask[i][j] = 1
            else:
                result_mask[i][j] = 0
        else:
            result_mask[i][j] = 0
#        
        
#%%

np.save('result_mask',result_mask)            
cv2.imshow("1",result_mask)
cv2.waitKey(0)