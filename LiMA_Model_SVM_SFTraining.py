# -*- coding: utf-8 -*-
"""
Runs single-class SVM on LiMA frames
This version trains on both surface forms and tests on categorization for other skeletons

Created on Tue Feb 18 13:59:43 2020

@author: VAYZENB
"""

# -*- coding: utf-8 -*-
"""
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

skel= [['23', '31', '266'] ,['31_0', '31_50']]
SFtrain = ['Skel', 'Bulge', 'Balloon', 'Shrink', 'Wave'] #Train SFs to test on Bulge
SFtest = ['Skel', 'Bulge'] #Train SFs to test on skel

modelType = ['FF_SN', 'R_SN']



frames= 300
trainFrames = 150

#Train labels
#not sure if needed
labels = [np.repeat(1, frames).tolist(), np.repeat(2, frames).tolist(), \
          np.repeat(3, frames).tolist(), np.repeat(4, frames).tolist()]
#labels = list(chain(*labels))

folK = 2

#For single class SVM
#Nu value is the proportion of outliers you expect (i.e., upper-bound on training data)
#Gamma parameter determines smoothing of the edges of data (i.e., the )
clf = svm.OneClassSVM(gamma = 'scale', nu=.01)

for ee in range(0,len(exp)):
    n = 0
    CNN_Acc = np.empty((len(skel[ee]) * len(SFtest)*len(modelType),6), dtype = object)
    
    for mm in range(0, len(modelType)):      
        
        allActs = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5')
        
        for sTR in range(0,len(skel[ee])): #The training will now be grouped by skeleton type          
            
            for sTE in range(0,len(SFtest)):
                if SFtest[sTE] == 'Skel':
                    altSF = 'Bulge'
                elif SFtest[sTE] == 'Bulge':
                    altSF = 'Skel'
                    
                
                trainAcc = 0
                testAcc = 0
                for fl in range(0,folK):
                    rN = np.random.choice(frames, frames, replace=False) 
                    
                    trainFig = 'Figure_' + skel[ee][sTR]
                    
                    #Create train data by stacking surface forms
                    X_train =np.vstack([allActs[trainFig + '_' + altSF][rN[0:int(frames/2)],:],\
                                        allActs[trainFig + '_Balloon'][rN[0:int(frames/2)],:], \
                                        allActs[trainFig + '_Shrink'][rN[0:int(frames/2)],:],\
                                        allActs[trainFig + '_Wave'][rN[0:int(frames/2)],:]])
                    
                    clf.fit(X_train)
                    tempAcc_train = clf.predict(X_train)
                    trainAcc = trainAcc + ((frames/2) - tempAcc_train[tempAcc_train == -1].size)/(frames/2)
                
                    #Test on object, but left out surface form
                    X_test = allActs[trainFig + '_' + SFtest[sTE]][rN[int(frames/2):frames],:]
                    tempAcc_test = clf.predict(X_test)
                    testAcc = testAcc + ((frames/2) - tempAcc_test[tempAcc_test == -1].size)/(frames/2)
                    
                CNN_Acc[n,0] = exp[ee] #Exp
                CNN_Acc[n,1] = modelType[mm] #Model
                CNN_Acc[n,2] = trainFig #Skel
                CNN_Acc[n,3] = SFtest[sTE] #Tested SF
                
                CNN_Acc[n,4] = trainAcc/folK
                CNN_Acc[n,5] = testAcc/folK
                
                n = n +1
                
                print(exp[ee], modelType[mm], trainFig, SFtest[sTE])
                
    np.savetxt('Results/LiMA_' + exp[ee] + '_allModels_OneClassSVM_SFTraining.csv', CNN_Acc, delimiter=',', fmt= '%s')
            
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
