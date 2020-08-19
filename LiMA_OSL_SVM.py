# -*- coding: utf-8 -*-
"""
Runs single-class SVM on LiMA frames
Tests categorization for: left out frames, same MA, same SF, all diff
Created on Sun Feb 16 15:16:04 2020

@author: VAYZENB
"""


from sklearn import svm
from sklearn.ensemble import IsolationForest
import numpy as np
import deepdish as dd

   
exp = ['Exp1', 'Exp2']


stim = [['23_Skel', '23_Bulge', '31_Skel', '31_Bulge','26_Skel', '26_Bulge'], \
        ['31_0_Skel', '31_0_Bulge','31_50_Skel', '31_50_Bulge']]

modelType = ['FF_SN','R_SN', 'FF_IN', 'R_IN', 'GBJ', 'GIST']


total_frames= 300
train_frames = 150
test_frames = total_frames - train_frames


folK = 10

#For single class SVM
#Nu value is the proportion of outliers you expect (i.e., upper-bound on training data)
#Gamma parameter determines smoothing of the edges of data (i.e., the )


for ee in range(0,len(exp)):
    n = 0
    CNN_Acc = np.empty(((len(stim[ee]) * len(stim[ee])*len(modelType))*10,10), dtype = object)
    
    for mm in range(0, len(modelType)):      
        
        
        allActs = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5')
        for sTR in range(0,len(stim[ee])):
            for sTE in range(0,len(stim[ee])):
                trainAcc = 0
                testAcc = 0
                for fl in range(0,folK):
                    #instantiate SVM everytime
                    ocs = svm.OneClassSVM(nu=.01) #one class SVM
                    #cov= EllipticEnvelope(random_state=0, contamination=0.01) #Elliptic Envelope classifier
                    isof=IsolationForest(random_state=0, contamination=0.01) #Isolation forest classifier
                    #lof = LocalOutlierFactor(n_neighbors=30, contamination=0.01) #local outlier factor
                    
                    #Shuffle order of frames
                    rN = np.random.choice(total_frames, total_frames, replace=False) 
                    
                    #select training frames
                    X_train = allActs['Figure_' + stim[ee][sTR]][rN[0:int(train_frames)],:]
                    
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
                    trainAcc_ocs = ((train_frames) - ocs_train[ocs_train == -1].size)/(train_frames)
                    #trainAcc_cov = ((frames/2) - cov_Train[cov_Train == -1].size)/(frames/2)
                    trainAcc_isof = ((train_frames) - isof_Train[isof_Train == -1].size)/(train_frames)
                    #trainAcc_lof = ((frames/2) - lof_Train[lof_Train == -1].size)/(frames/2)
                
                    #Test on object, but left out frames
                    X_test = allActs['Figure_' + stim[ee][sTE]][rN[int(train_frames):total_frames],:]
                    #X_test = allActs['Figure_' + stim[ee][sTE]][rN[0:total_frames],:]
                    
                    #Predict test data
                    ocs_test = ocs.predict(X_test)
                    #cov_test = cov.predict(X_test)
                    isof_test = isof.predict(X_test)
                    #lof_test = lof.predict(X_test)
                    
                    testAcc_ocs = ((test_frames) - ocs_test[ocs_test == -1].size)/(test_frames)
                    #testAcc_cov = ((frames/2) - cov_test[cov_test == -1].size)/(frames/2)
                    testAcc_isof = ((test_frames) - isof_test[isof_test == -1].size)/(test_frames)
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
                    #CNN_Acc[n,8] = trainAcc_cov
                    #CNN_Acc[n,9] = testAcc_cov
                    CNN_Acc[n,8] = trainAcc_isof
                    CNN_Acc[n,9] = testAcc_isof
                    #CNN_Acc[n,12] = trainAcc_lof
                    #CNN_Acc[n,13] = testAcc_lof

                    
                    print(exp[ee], modelType[mm], skel, SF, CNN_Acc[n,7], CNN_Acc[n,9])
                    
                    n = n +1
                
                
  
        np.savetxt('Results/LiMA_' + exp[ee] + '_allModels_OSL.csv', CNN_Acc, delimiter=',', fmt= '%s')
            
