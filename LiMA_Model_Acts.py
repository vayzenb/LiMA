# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:42:11 2020

@author: VAYZENB
"""

import os
#os.chdir('C:/Users/vayzenb/Desktop/GitHub Repos/LiMA/')

import numpy as np
import pandas as pd
import itertools
import glob
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.models
import torchvision.transforms as T
from torch.autograd import Variable
from PIL import Image
from itertools import chain
import deepdish as dd

exp = ['Exp1', 'Exp2']

stim = [['23_Skel', '23_Bulge', '31_Skel', '31_Bulge','266_Skel', '266_Bulge'],['31_Skel_0', '31_Bulge_0','31_Skel_50', '31_Bulge_50']]
modelType = ['FF_IN', 'R_IN', 'FF_SN', 'R_SN']
modelType = ['FF_SN', 'R_SN']

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


for mm in range(0, len(modelType)):
        #select model to run
    if modelType[mm] == 'FF_IN':
        model = torchvision.models.alexnet(pretrained=True)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        model.classifier = new_classifier #replace model classifier with stripped version
        layer = "fc7"
        actNum = 4096
        
    elif modelType[mm] == 'R_IN':
        model = torchvision.models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        layer = "avgpool"
        actNum = 2048
                
    elif modelType[mm] == 'FF_SN':
        model = torchvision.models.alexnet(pretrained=False)
        #model.features = torch.nn.DataParallel(model.features)
        checkpoint = torch.load('ShapeNet_AlexNet_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        model.classifier = new_classifier #replace model classifier with stripped version
        #model.to(device)
        layer = "fc7"
        actNum = 4096
        
    elif modelType[mm] == 'R_SN':
        model = torchvision.models.resnet50(pretrained=False)
        #model = torch.nn.DataParallel(model.features)
        checkpoint = torch.load('ShapeNet_ResNet50_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        model = nn.Sequential(*list(model.children())[:-1])
        #model.to(device)
        layer = "avgpool"
        actNum = 2048
        
    #Loop through the experimental conditions
    for ee in range(0,len(exp)):
        allActs = {}
        
        for ss in range(0,len(stim[ee])):
            allActs['Figure_' + stim[ee][ss]] = np.zeros((frames, actNum))
            for ff in range(0, frames):
                IM = image_loader('Frames/Figure_' + stim[ee][ss] + '_' + str(ff+1) +'.jpg')
                vec = model(IM).detach().numpy() #Extract image vector
                allActs['Figure_' + stim[ee][ss]][ff] = list(chain.from_iterable(vec))
                
            print(modelType[mm], exp[ee], stim[ee][ss])
                
        dd.io.save('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5', allActs)
        
    