# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:13:06 2019

@author: SzuHaoWu
"""
import numpy as np
import cv2
blue = []
notblue = []
barblue = []
otherblue = []
num_T,num_F = 0,0
for i in range(1,47):
    x = 'C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/trainset/mask_{}.npy'.format(i)
    y = 'C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/trainset/{}.png'.format(i)
    z = 'C:/Users/SzuHaoWu/Desktop/UCSD/ECE276A/ECE276A_HW1/label1/otherblue{}.npy'.format(i)
    maskbar = np.load(x)
    maskother = np.load(z)
    img = cv2.imread(y)
    (m,n) = maskbar.shape
    for j in range(m):
        for k in range(n):
            if maskbar[j,k] == True:
                barblue.append(img[j,k])
                blue.append(img[j,k])
                num_T = num_T+1
            elif maskother[j,k] == True:
                otherblue.append(img[j,k])
                blue.append(img[j,k])
                num_T = num_T+1
            else:
                notblue.append(img[j,k])
                num_F = num_F+1

#%%
meanbar = np.mean(barblue[:],axis=0)
np.save('meanbar',meanbar)
meanother = np.mean(otherblue[:],axis=0)
np.save('meanother',meanother)
meanblue = np.mean(blue[:],axis=0)
np.save('meanblue',meanblue)
meannotblue = np.mean(notblue[:],axis=0)
np.save('meannotblue',meannotblue)
varbar = np.var(barblue[:],axis=0)
np.save('covbar',np.diag(varbar))
varother = np.var(otherblue[:],axis=0)
np.save('covother',np.diag(varother))
varblue = np.var(blue[:],axis=0)
np.save('covblue',np.diag(varblue))
varnotblue = np.var(notblue[:],axis=0)
np.save('covnotblue',np.diag(varnotblue))

prior_blue = num_T/(num_T+num_F)
prior_notblue = num_F/(num_T+num_F)

