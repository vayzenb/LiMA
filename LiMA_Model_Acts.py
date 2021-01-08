# -*- coding: utf-8 -*-
"""
Extracts feature activations for frames of LiMA objects (exp1 + exp2)
Does for feedforward (AlexNet) and Recurrent (ResNet) models trained on imageNet (IN) or stylized IN (SN)
Created on Sun Feb 16 14:42:11 2020

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
modelType = ['ResNext-TC-SAY','CorNet_Z', 'CorNet_S','ResNet_IN', 'ResNet_SN']
#modelType = ['SayCam']


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

#Gets feats for CorNet models
def _store_feats(layer, inp, output):
    """An ugly but effective way of accessing intermediate model features
    """   
    output = output.cpu().detach().numpy()
    _model_feats.append(np.reshape(output, (len(output), -1)))

def load_model(modelType_):
    #select model to run
    if modelType_ == 'AlexNet_IN':
        model = torchvision.models.alexnet(pretrained=True)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        model.classifier = new_classifier #replace model classifier with stripped version
        layer = "fc7"
        actNum = 4096
        
    elif modelType_ == 'ResNet_IN':
        model = torchvision.models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        layer = "avgpool"
        actNum = 2048
                
    elif modelType_ == 'AlexNet_SN':
        model = torchvision.models.alexnet(pretrained=False)
        checkpoint = torch.load('Weights/ShapeNet_AlexNet_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        model.classifier = new_classifier #replace model classifier with stripped version
        layer = "fc7"
        actNum = 4096
        
    elif modelType_ == 'ResNet_SN':
        model = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load('Weights/ShapeNet_ResNet50_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        model = nn.Sequential(*list(model.children())[:-1])
        
        layer = "avgpool"
        actNum = 2048
    
    elif modelType_ == 'CorNet_Z':
        model = getattr(cornet, 'cornet_z')
        model = model(pretrained=False, map_location='gpu')
        checkpoint = torch.load('Weights/cornet_z.pth')
        model.load_state_dict(checkpoint['state_dict'])
        layer = "avgpool"
        actNum = 512
            
        decode_layer = nn.Sequential(*list(model.children())[0][4][:-3])
        model = nn.Sequential(*list(model.children())[0][:-1])
        model.add_module('4', decode_layer)
        
        
        #try:
        #    m = model.module
        #except:
        #    m = model
        #model_layer = getattr(getattr(m, 'decoder'), layer)
        #model_layer.register_forward_hook(_store_feats)

    elif modelType_ == 'CorNet_S':
        model = getattr(cornet, 'cornet_s')
        model = model(pretrained=False, map_location='gpu')
        checkpoint = torch.load('Weights/cornet_s.pth')
        model.load_state_dict(checkpoint['state_dict'])
        layer = "avgpool"
        actNum = 512        

        decode_layer = nn.Sequential(*list(model.children())[0][4][:-3])
        model = nn.Sequential(*list(model.children())[0][:-1])
        model.add_module('4', decode_layer)
        #try:
        #    m = model.module
        #except:
        #    m = model
        
        #model_layer = getattr(getattr(m, 'decoder'), layer)
        #model_layer.register_forward_hook(_store_feats)

    elif modelType_ == 'ResNext-TC-SAY':
        model = torchvision.models.resnext50_32x4d(pretrained=False)
        #model = torch.nn.DataParallel(model)
        #model.fc = torch.nn.Linear(in_features=2048, out_features=n_out, bias=True)
        checkpoint = torch.load('Weights/SayCam_ResNext_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        
        actNum = 2048

        model = nn.Sequential(*list(model.children())[:-1])
        
    return model, actNum


for mm in range(0, len(modelType)):
        #select model to run
    model, actNum = load_model(modelType[mm])
    

    model.cuda()
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
                        
                        vec = model(IM).cpu().detach().numpy() #Extract image vector
                        vec = list(chain.from_iterable(vec))

                        allActs['Figure_' + skel[ee][ss] +'_' + sf][ff] = vec
                        
                    print(modelType[mm], exp[ee], skel[ee][ss] +'_' + sf)
                        
                    dd.io.save('Activations/LiMA_' + exp[ee] + '_' + modelType[mm] + '_Acts.h5', allActs)
            
    