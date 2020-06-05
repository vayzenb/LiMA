# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:09:24 2020

@author: vayze
"""

import scipy.io as spio
import deepdish as dd

exp = ['Exp1', 'Exp2']

skel = [['23','31', '26'],['31_0', '31_50']]
SF = ['Skel', 'Bulge']
cond = ['', '_Size20']
#cond = ['_Side']

#mat = spio.loadmat('GBJ_Acts/Figure_23_Bulge_GBJActs.mat', squeeze_me=True)

IMsize = str(20)

modelType = ['GBJ', 'GIST']

frames = 300

for cc in cond:
    
    for mm in range(0, len(modelType)):
        if modelType[mm] == 'GBJ':
            actNum = 5760
            
        elif modelType[mm] == 'GIST':
            actNum = 512
            
        
        for ee in range(0,len(exp)):
            allActs = {}
        
            for ss in range(0,len(skel[ee])):
                for sf in SF:
                    
                    
                    mat = spio.loadmat('Activations/EV_Acts/Figure_' + skel[ee][ss] + '_' + sf + '_' + modelType[mm] + '_Acts' + cc + '.mat', squeeze_me=True)
                    
                    if cc == '_Size20':
                        matName = 'sizeActs_' + modelType[mm]
                    else:
                        matName = 'stimActs_' + modelType[mm]
                        
                    allActs['Figure_' + skel[ee][ss] +'_' + sf] = mat[matName]
                

            print(modelType[mm])
            dd.io.save('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts' + cc + '.h5', allActs)
            
                    

