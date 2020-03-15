# -*- coding: utf-8 -*-
"""
Resizes frames and extracts feature activations for LiMA objects (exp1 + exp2)
Does for feedforward (AlexNet) and Recurrent (ResNet) models trained on imageNet (IN) or stylized IN (SN)
Created on Sun Mar 15 10:28:07 2020

@author: vayze
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


IMscale = 1+.25

exp = ['Exp1', 'Exp2']

skel = [['23','31', '26'],['31_0', '31_50']]
SF = ['Skel', 'Balloon', 'Bulge','Shrink','Wave']
modelType = ['FF_SN', 'R_SN']

frames = 300

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#Set image loader for model
def image_loader(image_name):
    """load image, returns cuda tensor"""
    ogIM = Image.open(image_name).convert("RGB")
    #Create gray background frame
    #scale background to X% of original image
    newIM = Image.new('RGBA', (int(ogIM.size[0]*IMscale),int(ogIM.size[1]*IMscale)), (119, 119, 119))
    
    #Overlay image on new background
    newIM.paste(ogIM,((newIM.width - ogIM.width) // 2, (newIM.height - ogIM.height) // 2))
    
    #Resize newIM to ogIM size
    newIM.resize(ogIM.size)
    
    newIM = Variable(normalize(to_tensor(scaler(newIM))).unsqueeze(0))
    return newIM     


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
        
        model.eval() #Set model into evaluation mode
        
    #Loop through the experimental conditions
    for ee in range(0,len(exp)):
        allActs = {}
        
        for ss in range(0,len(skel[ee])):
            for sf in SF:
                allActs['Figure_' + skel[ee][ss] +'_' + sf] = np.zeros((frames, actNum))
                for ff in range(0, frames):
                    IM = image_loader('Frames/Figure_' + skel[ee][ss] +'_' + sf + '_' + str(ff+1) +'.jpg')
                    vec = model(IM).detach().numpy() #Extract image vector
                    allActs['Figure_' + skel[ee][ss] +'_' + sf][ff] = list(chain.from_iterable(vec))
                    
                print(modelType[mm], exp[ee], skel[ee][ss] +'_' + sf)
                    
                dd.io.save('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts_Size' + str((IMscale-1)*1000)+ '.h5', allActs)
