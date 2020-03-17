# -*- coding: utf-8 -*-
"""

Runs single-class SVM on LiMA frames of different sized objects
Created on Sun Mar 15 15:32:58 2020

@author: vayze
"""


from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import random

from skimage import io
import deepdish as dd

   
exp = ['Exp1', 'Exp2']

stim = [['23_Skel', '23_Bulge', '31_Skel', '31_Bulge','26_Skel', '26_Bulge'], \
        ['31_0_Skel', '31_0_Bulge','31_50_Skel', '31_50_Bulge']]
modelType = ['R_SN']
IMsize = str(19)


frames= 300
labels = [np.repeat(1, frames).tolist(), np.repeat(2, frames).tolist()]
#labels = list(chain(*labels))

folK = 10

#For single class SVM
#Nu value is the proportion of outliers you expect (i.e., upper-bound on training data)
#Gamma parameter determines smoothing of the edges of data (i.e., the )


for ee in range(0,len(exp)):
    n = 0
    CNN_Acc = np.empty(((len(stim[ee]) * len(stim[ee])*len(modelType))*10,8), dtype = object)
    
    for mm in range(0, len(modelType)):      
        
        
        allActs = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5')
        allActsSize = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts_Size' + IMsize +'.h5')
        for sTR in range(0,len(stim[ee])):
            for sTE in range(0,len(stim[ee])):
                trainAcc = 0
                testAcc = 0
                for fl in range(0,folK):
                    #instantiate SVM everytime
                    clf = svm.OneClassSVM(nu=.01)
                    
                    rN = np.random.choice(frames, frames, replace=False) 
                    
                    X_train = allActs['Figure_' + stim[ee][sTR]][rN[0:int(frames/2)],:]
                    
                    clf.fit(X_train)
                    tempAcc_train = clf.predict(X_train)
                    trainAcc = ((frames/2) - tempAcc_train[tempAcc_train == -1].size)/(frames/2)
                
                    #Test on object, but left out frames
                    X_test = allActsSize['Figure_' + stim[ee][sTE]][rN[int(frames/2):frames],:]
                    tempAcc_test = clf.predict(X_test)
                    testAcc = ((frames/2) - tempAcc_test[tempAcc_test == -1].size)/(frames/2)
                    
                    CNN_Acc[n,0] = exp[ee]
                    CNN_Acc[n,1] = modelType[mm]
                    CNN_Acc[n,2] = 'Figure_' + stim[ee][sTR]
                    CNN_Acc[n,3] = 'Figure_' + stim[ee][sTE]
                    
                    #Check if first 2 characters are same 
                    #to determine whether skel is the same
                    
                    if exp[ee] == 'Exp1':
                            
                        if stim[ee][sTR][0:2] == stim[ee][sTE][0:2]:
                            skel = 'Same'
                        else:
                            skel = 'Diff'
                        
                            
                    elif exp[ee] == 'Exp2':
                        if stim[ee][sTR][4:5] == stim[ee][sTE][4:5]:
                            skel = 'Same'
                        else:
                            skel = 'Diff'
                        
                    #check if surface forms are the same
                    if stim[ee][sTR][-4:] == stim[ee][sTE][-4:]:
                        SF = 'Same'
                    else:
                        SF = 'Diff'
                                
                    CNN_Acc[n,4] = skel
                    CNN_Acc[n,5] = SF
                    CNN_Acc[n,6] = trainAcc
                    CNN_Acc[n,7] = testAcc
                    
                    print(exp[ee], modelType[mm], skel, SF, CNN_Acc[n,6], CNN_Acc[n,7])
                    
                    n = n +1
                 
                
        np.savetxt('Results/LiMA_' + exp[ee] + '_allModels_OneClassSVM_Size' + IMsize + '.csv', CNN_Acc, delimiter=',', fmt= '%s')
            
