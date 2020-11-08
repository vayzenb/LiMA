# -*- coding: utf-8 -*-
"""
Runs single-class SVM on LiMA frames
Tests categorization for: left out frames, same MA, same SF, all diff
This version trains on multiple SFs and tests on the left out one
Created on 8.1.20

@author: VAYZENB
"""


from sklearn import svm
from sklearn.ensemble import IsolationForest
import numpy as np
import deepdish as dd

   
exp = ['Exp1', 'Exp2']

modelType = ['FF_SN','R_SN', 'FF_IN', 'R_IN', 'GBJ', 'GIST']
modelType = ['CorNet_Z', 'CorNet_S']

skel = [['23', '31', '26'], ['31_0', '31_50']]
SF = ['Skel', 'Bulge', 'Balloon','Shrink', 'Wave']
testSF = ['Skel', 'Bulge']

total_frames= 300
train_frames = 148
test_frames = total_frames - train_frames

folK = 10

#For single class SVM
#Nu value is the proportion of outliers you expect (i.e., upper-bound on training data)
#Gamma parameter determines smoothing of the edges of data (i.e., the )


for ee in range(0,len(exp)):
    n = 0
    CNN_Acc = np.empty(((len(skel[ee]) * len(skel[ee])*len(modelType))*len(testSF)*folK,10), dtype = object)
    
    for mm in range(0, len(modelType)):      
        
        
        allActs = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5')
        for sTR in range(0,len(skel[ee])):
            for sTE in range(0,len(skel[ee])):
                for tSF in testSF:
                    trainSF = SF[:]
                    trainSF.remove(tSF)

                    trainAcc = 0
                    testAcc = 0
                    for fl in range(0,folK):
                        #Shuffle order of frames
                        rN = np.random.choice(total_frames, total_frames, replace=False)

                        #Test on object, but left out frames
                        X_test = allActs['Figure_' + skel[ee][sTE]+ '_' + tSF][rN[int(train_frames):total_frames],:]

                        
                        #Create empty train array with the same columns as the test set
                        X_train = np.array([], dtype=np.int64).reshape(0,X_test.shape[1])
                        for trSF in trainSF:
                            trFrames = np.random.choice(rN[0:train_frames], int(train_frames/len(trainSF)), replace=False)
                            X_train = np.vstack((X_train, allActs['Figure_' + skel[ee][sTR]+ '_' + trSF][trFrames,:]))


                        #instantiate SVM everytime
                        ocs = svm.OneClassSVM(nu=.01) #one class SVM
                        isof=IsolationForest(random_state=0, contamination=0.01) #Isolation forest classifier                       

                        
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
                        CNN_Acc[n,2] = 'Figure_' + skel[ee][sTR]
                        CNN_Acc[n,3] = 'Figure_' + skel[ee][sTE]
                        
                                                
                        #Check if skels are the same        
                        if skel[ee][sTR] == skel[ee][sTE]:
                            skelType = 'Same'
                        else:
                            skelType = 'Diff'
                                    
                        CNN_Acc[n,4] = skelType
                        CNN_Acc[n,5] = tSF
                        CNN_Acc[n,6] = trainAcc_ocs
                        CNN_Acc[n,7] = testAcc_ocs
                        CNN_Acc[n,8] = trainAcc_isof
                        CNN_Acc[n,9] = testAcc_isof


                        
                        print(exp[ee], modelType[mm], skel[ee][sTR],skel[ee][sTE], skelType,tSF, CNN_Acc[n,7], CNN_Acc[n,9])
                        
                        n = n +1
                    
                    
    
            np.savetxt('Results/LiMA_' + exp[ee] + '_CorNet_MSL.csv', CNN_Acc, delimiter=',', fmt= '%s')
            
