curr_dir = '/user_data/vayzenbe/GitHub_Repos/LiMA'

import sys
sys.path.insert(1, f'{curr_dir}')
import os, argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import datasets
import torchvision.models as models
import numpy as np
from LoadFrames import LoadFrames
from statistics import mean
from PIL import Image
import matplotlib.pyplot as plt
import gc
import pandas as pd
import pdb
from itertools import chain
import deepdish as dd

'''
set up steps
'''

exp = ['Exp1', 'Exp2']

skel = [['23','31','26'],['31_0', '31_50']]
SF = ['Skel', 'Bulge']
modelType = ['pixel', 'flownet']
modelType = ['flownet']
#modelType = ['AlexNet_SN', 'ResNet_SN', 'AlexNet_IN', 'ResNet_IN', 'CorNet_Z', 'CorNet_S','SayCam']

batch_num = 10
n_frames = 300
#hab_min = 4 #minimum number of habituation trials to 
#batch_num = 10 #how many frames to use at a time
#exp = ['Exp1']
#skel=[['23']]
#SF = ['Skel']
#modelType = ['ResNet_SN',  'ResNet_IN', 'CorNet_Z', 'CorNet_S','SayCam']

#Transformations for ImageNet
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
# specify loss function
criterion = nn.MSELoss()


epochs = 100
hab_min = 4 #minimum number of habituation trials

def save_recon(out_, model, stim):
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')

    # Show the image
    #ax.imshow(im)

    out_ = out_.squeeze(0)
    out_ = inv_normalize(out_)
    out_ = out_.cpu().detach()
    ax.imshow(out_.permute(1, 2, 0))
    plt.axis('off')
    #print(f'Results/AE/recon/{model}_{fig_fig_}.png')
    plt.savefig(f'{curr_dir}/Results/AE/recon/{model}_{stim}.png', bbox_inches='tight', pad_inches = 0, dpi=150)
    plt.close()
    

def define_decoder(act_num):
    decoder = nn.Sequential(nn.Conv2d(3,act_num,kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.AdaptiveAvgPool2d(1),nn.ReLU(), nn.ConvTranspose2d(act_num, 3, 224))
    decoder = decoder.cuda()
    
    return decoder

def save_model(model, epoch, optimizer, loss, file_path):

    print('Saving model ...')
    #torch.save(model.state_dict(), f'{weights_dir}/cornet_classify_{cond}_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, file_path)


def load_model(modelType_):
    global stim_dir
    #select model to run
    if modelType_ == 'pixel':
        #model = nn.Sequential(nn.Conv2d(3,1024,kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.AdaptiveAvgPool2d(1))
        act_num = 1024
        stim_dir = f'{curr_dir}/Frames'
        

    elif modelType_ == 'flownet':
        #model = nn.Sequential(nn.Conv2d(3,4096,kernel_size=3, stride=2), nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.AdaptiveAvgPool2d(1))
        act_num = 1024
        stim_dir = f'{curr_dir}/modelling/flow_data'

    elif modelType_ == 'gbj':
        #use the gabor wavelets as they are (in a tranposed vector form)
        #they are already in a low-d space
        #the loss is still calcualted from the image though (but maybe grayscale it)

        act_num = 8000
    
    
    decoder = define_decoder(act_num)
    return decoder, act_num

'''
habituate
'''

def habituate(exp, model_type):
    """
    Habituates an autoencoder on one object
    """

    print('Habituating model...')
    #encoder, in_feat = load_model(model_type)
    hn = 0
    for sk in range(0,len(skel[exp[0]])):
        for sf in SF:
            
            torch.cuda.empty_cache() #clear GPU memory
            #Reset decoder for every object (i.e., make it like a fresh hab session)
            #Create decoder
            decoder, act_num = load_model(model_type)
            hab_dataset = LoadFrames(f'{stim_dir}/Figure_{skel[exp[0]][sk]}_{sf}', transform=transform)
            trainloader = torch.utils.data.DataLoader(hab_dataset, batch_size=batch_num, shuffle=True, num_workers = 2, pin_memory=True)

            early_hab = 0.0
            late_hab = []

            
            
            decoder.train()

            #set up optimzer
            optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
            for ep in range(0,epochs):
                train_loss = 0.0 
                total_loss =0.0
                n = 0
                for frames in trainloader:
                    frames = frames.cuda()


                    optimizer.zero_grad() #zero out gradients from previous epoch

                    decode_out = decoder(frames) #Run features through decoder

                    loss = criterion(decode_out, frames) #Calculate loss

                    # backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # perform a single optimization step (parameter update)
                    optimizer.step()

                    train_loss += (loss.item()*frames.size(0))
                    n = n +1
                    #print(train_loss, loss.item()*frames.size(0), n)

                total_loss = train_loss/n

                if ep < hab_min:
                    early_hab += total_loss #track loss for the first 4 trials
                    print(ep, total_loss)
                elif ep >= hab_min:
                    hab_start = early_hab / hab_min #Determine habituation criterion
                    late_hab.append(total_loss) #add current loss to habituation
                    hab_end = mean(late_hab[(len(late_hab)-4):len(late_hab)]) #calcualte mean of last 4 hab trials

                    print(ep, total_loss, hab_start, hab_end)
                    if hab_end < (hab_start/2) and ep >= int(hab_min *2): #test if habituated
                        break
                
                

            hab_data[hn,0] =  model_type
            hab_data[hn,1] =  skel[exp[0]][sk]
            hab_data[hn,2] =  sf
            hab_data[hn,3] =  ep
            hab_data[hn,4] =  hab_start
            hab_data[hn,5] =  hab_end

            print('Saving model', f'Figure_{skel[exp[0]][sk]}_{sf}', ep, hab_start, hab_end)  
            
            save_model(decoder, ep, optimizer, loss, f'{curr_dir}/Weights/decoder/{exp[1]}_{model_type}_Figure_{skel[exp[0]][sk]}_{sf}.pt')
            np.save(f'{curr_dir}/Weights/decoder/{exp[1]}_Summary_{model_type}.npy', hab_data)
            hn = hn + 1
            del decoder
            gc.collect()
            


def dishabituate(exp, model_type):

    '''
    Load each habituated model for one object and tests dishabituation for every other object
    
    '''
    print('Testing dishabitation...')
    
        
    hab_data = np.load(f'{curr_dir}/Weights/decoder/{exp[1]}_Summary_{model_type}.npy',allow_pickle=True)
        
    #for mm in range(0,1):

    #test_data = np.empty(((len(skel[ee]) * len(SF)),hab_data.shape[1]), dtype = object)
    
    exp_result = []
    #habituation index
    hn = 0
    #load habituated stim
    for sk_hab in skel[exp[0]]:
        for sf_hab in SF:
            trial_result = []
            torch.cuda.empty_cache() #clear GPU memory
            
            #Load model
            decoder, act_num = load_model(model_type)

            #Load checkpoint from fully trained decoder
            checkpoint = torch.load(f'{curr_dir}/Weights/decoder/{exp[1]}_{model_type}_Figure_{sk_hab}_{sf_hab}.pt')
            #print(f'Weights/decoder/{exp[ee]}_{hab_data[mm,0]}_Figure_{hab_data[mm,1]}_{hab_data[mm,2]}.pt')
            decoder.load_state_dict(checkpoint['model_state_dict'])
            decoder.eval()
            
            #load data for dishabituation
            for sk_dis in skel[exp[0]]:
                for sf_dis in SF:
                    #load stim for dishab object
                    hab_dataset = LoadFrames(f'{stim_dir}/Figure_{sk_dis}_{sf_dis}', transform=transform)
                    trainloader = torch.utils.data.DataLoader(hab_dataset, batch_size=batch_num, shuffle=True, num_workers = 2, pin_memory=False)
            

                    if sk_hab == sk_dis:
                        skel_cat = "same"
                    else:
                        skel_cat = "diff"

                    if sf_hab == sf_dis:
                        sf_cat = "same"
                    else:
                        sf_cat = "diff"

                    #total_loss = []
                    train_loss = 0.0 
                    total_loss =0.0
                    n =0

                    for frames in trainloader:
                        frames = frames.cuda()

                        decode_out = decoder(frames) #Run features through decoder

                        loss = criterion(decode_out, frames) #Calculate loss
                        train_loss += (loss.item()*frames.size(0))
                        n = n +1

                        # backward pass: compute gradient of the loss with respect to model parameters
                        #loss.backward()
                        # perform a single optimization step (parameter update)
                        #optimizer.step()                        

                        #print(train_loss, loss.item()*frames.size(0), n)
                    total_loss = train_loss/n
                    dishab_trial = hab_data[hn,:6].tolist() + [sk_dis, sf_dis, skel_cat, sf_cat, total_loss]
                    print(dishab_trial)
                    trial_result.append(dishab_trial)
                    exp_result.append(dishab_trial)

                    #print(ep, total_loss)

                    '''
                    if skel_cat == 'same' and sf_cat == 'same':
                        save_recon(decode_out, skel_cat, sf_cat, hab_data[hn,0],f'Figure_{hab_data[hn,1]}_{hab_data[hn,2]}')
                    '''

            #print(trial_result)
            
            np.savetxt(f'{curr_dir}/Results/AE/{exp[1]}_{hab_data[hn,0]}_Figure_{hab_data[hn,1]}_{hab_data[hn,2]}_Result.csv', np.array(trial_result), delimiter=',', fmt= '%s')
            hn = hn +1
            del decoder
            gc.collect()
            
    
    
    np.savetxt(f'{curr_dir}/Results/AE/{exp[1]}_Skel_Result.csv', np.array(exp_result), delimiter=',', fmt= '%s')

    df = pd.DataFrame(np.array(exp_result), columns = ['model','skel_hab','sf_hab', 'trials', 'hab_start', 'hab_end', 'skel_dishab', 'sf_dishab', 'skel_cat', 'sf_cat', 'loss'])
    df['loss'] = pd.to_numeric(df['loss'])

    print(df.groupby(['skel_cat', 'sf_cat'])['loss'].mean())

 
#ee = [1,'Exp2']
#habituate(ee,mm)

for ee in enumerate(exp):
    hab_data = np.empty(((len(skel[ee[0]]) * len(SF) *len(modelType)),11), dtype = object)
    hn = 0
    for mm in modelType:
        habituate(ee,mm)
        dishabituate(ee,mm)
 
