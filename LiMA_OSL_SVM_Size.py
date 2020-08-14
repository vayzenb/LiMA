# -*- coding: utf-8 -*-
"""

Runs single-class SVM on LiMA frames of different sized objects
Created on Sun Mar 15 15:32:58 2020

@author: vayze
"""


from sklearn import svm
from sklearn.ensemble import IsolationForest
import warnings
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



modelType = ['FF_SN','R_SN', 'FF_IN', 'R_IN', 'GBJ', 'GIST']
#modelType = ['FF_SN','R_SN', 'FF_IN', 'R_IN']


manip = ['Size10', 'Size20', 'Size30', 'Size40', 'Size50']

frames= 300
labels = [np.repeat(1, frames).tolist(), np.repeat(2, frames).tolist()]
#labels = list(chain(*labels))

folK = 10

#For single class SVM
#Nu value is the proportion of outliers you expect (i.e., upper-bound on training data)
#Gamma parameter determines smoothing of the edges of data (i.e., the )


for ee in range(0,len(exp)):


    for sz in manip:    
        n = 0
        CNN_Acc = np.empty(((len(stim[ee]) * len(stim[ee])*len(modelType))*10,10), dtype = object)
        for mm in range(0, len(modelType)):      
            
            
            allActs = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5')
            
            
            allActsTest = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts_' + sz +'.h5')
            for sTR in range(0,len(stim[ee])):
                for sTE in range(0,len(stim[ee])):
                    if stim[ee][sTR][-4] != stim[ee][sTE][-4]: continue
                
                    trainAcc = 0
                    testAcc = 0
                    
                    for fl in range(0,folK):
                        #instantiate SVM everytime
                        #instantiate SVM everytime
                        warnings.filterwarnings("ignore")
                        ocs = svm.OneClassSVM(nu=.01) #one class SVM
                        #cov= EllipticEnvelope(random_state=0, contamination=0.01) #Elliptic Envelope classifier
                        isof=IsolationForest(random_state=0, contamination=0.01) #Isolation forest classifier
                        #lof = LocalOutlierFactor(n_neighbors=30, contamination=0.01) #local outlier factor
                        
                        
                        rN = np.random.choice(frames, frames, replace=False) 
                        
                        X_train = allActs['Figure_' + stim[ee][sTR]][rN[0:int(frames/2)],:]
                        
                        #fit all classifiers
                        ocs.fit(X_train) #Fit one-class SVM
                        #cov.fit(X_train)
                        isof.fit(X_train)
                        #lof.fit(X_train)
                        
                        #Predict training data
                        ocs_train = ocs.predict(X_train)
                        #cov_Train = cov.predict(X_train)
                        isof_Train = isof.predict(X_train)
                        #lof_Train = lof.predict(X_train)
                        
                        #Compute training data scores
                        trainAcc_ocs = ((frames/2) - ocs_train[ocs_train == -1].size)/(frames/2)
                        #trainAcc_cov = ((frames/2) - cov_Train[cov_Train == -1].size)/(frames/2)
                        trainAcc_isof = ((frames/2) - isof_Train[isof_Train == -1].size)/(frames/2)
                        #trainAcc_lof = ((frames/2) - lof_Train[lof_Train == -1].size)/(frames/2)
                    
                        #Test on object, but left out frames
                        X_test = allActsTest['Figure_' + stim[ee][sTE]][rN[int(frames/2):frames],:]
                        
                        #Predict test data
                        ocs_test = ocs.predict(X_test)
                        #cov_test = cov.predict(X_test)
                        isof_test = isof.predict(X_test)
                        #lof_test = lof.predict(X_test)
                        
                        testAcc_ocs = ((frames/2) - ocs_test[ocs_test == -1].size)/(frames/2)
                        #testAcc_cov = ((frames/2) - cov_test[cov_test == -1].size)/(frames/2)
                        testAcc_isof = ((frames/2) - isof_test[isof_test == -1].size)/(frames/2)
                        #testAcc_lof = ((frames/2) - lof_test[lof_test == -1].size)/(frames/2)
                        
                        
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
                        CNN_Acc[n,6] = trainAcc_ocs
                        CNN_Acc[n,7] = testAcc_ocs
                        CNN_Acc[n,8] = trainAcc_isof
                        CNN_Acc[n,9] = testAcc_isof
    
    
                        
                        print(exp[ee], modelType[mm], skel, SF, CNN_Acc[n,7], CNN_Acc[n,9], sz)
                        
                        n = n +1
                     
                    
            np.savetxt('Results/LiMA_' + exp[ee] + '_allModels_AllClassifiers_' + sz + '.csv', CNN_Acc, delimiter=',', fmt= '%s')
            
