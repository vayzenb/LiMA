# -*- coding: utf-8 -*-
"""
Runs single-class SVM on LiMA frames
Tests categorization for: left out frames, same MA, same SF, all diff
This version trains on multiple SFs and tests on the left out one
Created on 8.1.20

@author: VAYZENB
"""


from sklearn import svm
import numpy as np
import deepdish as dd
import itertools

   
exp = ['Exp1', 'Exp2']


modelType = ['FF_SN','R_SN', 'FF_IN', 'R_IN', 'GBJ', 'GIST']


skel = [['23', '31', '26'], ['31_0', '31_50']]
SF = ['Skel', 'Bulge']



total_frames= 300
train_frames = 240
test_frames = total_frames - train_frames


trainLabels = [0]*train_frames + [1]*train_frames
testLabels = [0]*test_frames + [1]*test_frames

folK = 10

#For single class SVM
#Nu value is the proportion of outliers you expect (i.e., upper-bound on training data)
#Gamma parameter determines smoothing of the edges of data (i.e., the )


for ee in range(0,len(exp)):
    n = 0
    CNN_Acc = np.empty(((len(skel[ee]) * len(skel[ee])*len(modelType))*len(SF)*folK,10), dtype = object)
    
    for mm in range(0, len(modelType)):      
        
        
        allActs = dd.io.load('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5')
        for sTR in range(0,len(skel[ee])-1):
            for sTE in range(sTR+1,len(skel[ee])):
                for tSF in SF:
                    if tSF == 'Bulge':
                        trSF = 'Skel'
                    else:
                        trSF = 'Bulge'

                    trainAcc = 0
                    testAcc = 0
                    for fl in range(0,folK):
                        #Shuffle order of frames
                        rN = np.random.choice(total_frames, total_frames, replace=False)
                        

                        #Test on object, but left out frames
                        X_train = np.vstack((allActs['Figure_' + skel[ee][sTE]+ '_' + trSF][rN[0:int(train_frames)],:],
                        allActs['Figure_' + skel[ee][sTR]+ '_' + trSF][rN[0:int(train_frames)],:]))

                        X_test = np.vstack((allActs['Figure_' + skel[ee][sTE]+ '_' + tSF][rN[int(train_frames):total_frames],:],
                        allActs['Figure_' + skel[ee][sTR]+ '_' + tSF][rN[int(train_frames):total_frames],:]))


                        #instantiate SVM everytime
                        clf = svm.SVC(kernel='linear', C=1)

                        clf.fit(X_train, trainLabels)
                        
                        currScore = clf.score(X_test, testLabels)

                        CNN_Acc[n,0] = exp[ee]
                        CNN_Acc[n,1] = modelType[mm]
                        CNN_Acc[n,2] = 'Figure_' + skel[ee][sTR]
                        CNN_Acc[n,3] = 'Figure_' + skel[ee][sTE]
                        CNN_Acc[n,4] = trSF
                        CNN_Acc[n,5] = tSF
                        CNN_Acc[n,6] = currScore

                        
                        print(exp[ee], modelType[mm], skel[ee][sTR],skel[ee][sTE], trSF,tSF, currScore)
                        
                        n = n +1
                    
                    
    
            np.savetxt('Results/LiMA_' + exp[ee] + '_allModels_Contrasts.csv', CNN_Acc, delimiter=',', fmt= '%s')
            
