# -*- coding: utf-8 -*-
"""
Runs single-class SVM on LiMA frames
Tests categorization for: left out frames, same MA, same SF, all diff
Created on Sun Feb 16 15:16:04 2020

@author: VAYZENB
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

stim = [['23_Skel', '23_Bulge', '31_Skel', '31_Bulge','266_Skel', '266_Bulge'],['31_Skel_0', '31_Bulge_0','31_Skel_50', '31_Bulge_50']]
modelType = ['FF_IN', 'R_IN']



frames= 300
labels = [np.repeat(1, frames).tolist(), np.repeat(2, frames).tolist()]
#labels = list(chain(*labels))

folK = 10

#For single class SVM
#Nu value is the proportion of outliers you expect (i.e., upper-bound on training data)
#Gamma parameter determines smoothing of the edges of data (i.e., the )
clf = svm.OneClassSVM(gamma = 'scale', nu=.01)

for ee in range(0,len(exp)):
    n = 0
    CNN_Acc = np.empty((len(stim[ee]) * (len(stim[ee]))*4,8), dtype = object)
    
    for mm in range(0, len(modelType)):      
        
        
        allActs = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5')
        for sTR in range(0,len(stim[ee])):
            for sTE in range(0,len(stim[ee])):
                trainAcc = 0
                testAcc = 0
                for fl in range(0,folK):
                    rN = np.random.choice(frames, frames, replace=False) 
                    
                    X_train = allActs['Figure_' + stim[ee][sTR]][rN[0:int(frames/2)],:]
                    
                    clf.fit(X_train)
                    tempAcc_train = clf.predict(X_train)
                    trainAcc = trainAcc + ((frames/2) - tempAcc_train[tempAcc_train == -1].size)/(frames/2)
                
                    #Test on object, but left out frames
                    X_test = allActs['Figure_' + stim[ee][sTE]][rN[int(frames/2):frames],:]
                    tempAcc_test = clf.predict(X_test)
                    testAcc = testAcc + ((frames/2) - tempAcc_test[tempAcc_test == -1].size)/(frames/2)
                    
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
                    
                    #check if surface forms are the same
                    if stim[ee][sTR][-4:] == stim[ee][sTE][-4:]:
                        SF = 'Same'
                    else:
                        SF = 'Diff'
                        
                elif exp[ee] == 'Exp2':
                    if stim[ee][sTR][-2:] == stim[ee][sTE][-2:]:
                        skel = 'Same'
                    else:
                        skel = 'Diff'
                    
                    #check if surface forms are the same
                    if stim[ee][sTR][0:5] == stim[ee][sTE][0:5]:
                        SF = 'Same'
                    else:
                        SF = 'Diff'
                            
                CNN_Acc[n,4] = skel
                CNN_Acc[n,5] = SF
                CNN_Acc[n,6] = trainAcc/folK
                CNN_Acc[n,7] = testAcc/folK
                
                print(exp[ee], modelType[mm], skel, SF, CNN_Acc[n,7])
                
                n = n +1
                
                
                
    np.savetxt('Results/LiMA_' + exp[ee] + '_allModels_OneClassSVM.csv', CNN_Acc, delimiter=',', fmt= '%s')
            
#
#
#skelActs = np.vstack([allActs['31_Skel_0'],allActs ['31_Skel_50']])
#bulgeActs = np.vstack([allActs['31_Bulge_0'],allActs ['31_Bulge_50']])
#
#
#clf = svm.SVC(kernel='linear', C=1).fit(skelActs, labels)
##Add current score to existing
#tempScore = tempScore + clf.score(bulgeActs, labels)
#
#
#clf = svm.SVC(kernel='linear', C=1).fit(bulgeActs, labels)
##Add current score to existing
#tempScore = tempScore + clf.score(skelActs, labels)
#
#randList = np.random.choice(300, 300, replace=False) 
#
#
#
##Train it on half the data
#clf.fit(allActs['31_Bulge_0'][randList[0:149],:])
#
##Check training
#y_pred_train =  clf.predict(allActs['31_Bulge_0'][randList[0:149],:])
##Count how many errors it makes from training
#n_error_train = (300 -y_pred_train[y_pred_train == -1].size)/300
#
##Test on other half of the same data
#y_pred_test = clf.predict(allActs['31_Bulge_0'][randList[150:299],:])
##Check error rate of left out data
#n_error_test = (300 - y_pred_test[y_pred_test == -1].size)/300
#
#y_pred_outliers = clf.predict(allActs ['31_Bulge_50'])
#n_error_outliers = (300- y_pred_outliers[y_pred_outliers == 1].size)/300
#
#
