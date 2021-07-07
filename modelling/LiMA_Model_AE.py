# -*- coding: utf-8 -*-
"""
Uses an autoencoder to approximate the habituation/dishabituation process
An autoencoder is afixed ontop of the pre-trained model 
LiMA stim are run through it

@author: VAYZENB
"""

import os
#os.chdir('C:/Users/vayze/Desktop/GitHub Repos/LiMA/')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
from itertools import chain
import deepdish as dd
import cornet

exp = ['Exp1', 'Exp2']

skel = [['23','31', '26'],['31_0', '31_50']]
SF = ['Skel', 'Bulge', 'Balloon', 'Shrink', 'Wave']
modelType = ['SayCam','CorNet_Z', 'CorNet_S','AlexNet_SN', 'ResNet_SN', 'AlexNet_IN', 'ResNet_IN']
modelType = ['ResNet_IN']


frames = 300

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()
#Set image loader for model
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert("RGB")
    image = Variable(normalize(to_tensor(scaler(image))).unsqueeze(0))
    
    return image     


#Use MSEerror loss

for mm in range(0, len(modelType)):
        #select model to run
    if modelType[mm] == 'AlexNet_IN':
        model = torchvision.models.alexnet(pretrained=True)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier #replace model classifier with stripped version
        layer = "fc7"
        actNum = 4096
        
    elif modelType[mm] == 'ResNet_IN':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = 
        layer = "avgpool"
        actNum = 2048

    #Decoder
    decode = nn.Sequential(nn.ReLU(),nn.ConvTranspose2d(actNum, 3, 224))

    #model.cuda()
    model.eval() #Set model into evaluation mode
        
    with torch.no_grad():
        #Loop through the experimental conditions
        for ee in range(0,len(exp)):
            allActs = {}
            
            for ss in range(0,len(skel[ee])):
                for sf in SF:
                    allActs['Figure_' + skel[ee][ss] +'_' + sf] = np.zeros((frames, actNum))
                    for ff in range(0, frames):
                        IM = image_loader('Frames/Figure_' + skel[ee][ss] +'_' + sf + '/Figure_' + skel[ee][ss] +'_' + sf + '_' + str(ff+1) +'.jpg')
                        IM = IM.cuda()
                        if modelType[mm] == 'CorNet_Z' or modelType[mm] == 'CorNet_S':
                            _model_feats = []
                            model(IM)
                            vec = _model_feats[0][0]
                        else:
                            vec = model(IM).cpu().detach().numpy() #Extract image vector
                            vec = list(chain.from_iterable(vec))

                        allActs['Figure_' + skel[ee][ss] +'_' + sf][ff] = vec
                        
                    print(modelType[mm], exp[ee], skel[ee][ss] +'_' + sf)
                        
                    dd.io.save('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5', allActs)
            
    