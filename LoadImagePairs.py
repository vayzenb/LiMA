import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np
import imageio
from glob import glob
import pdb
from natsort import natsorted

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imageio.imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im

    return []

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]


class LoadImagePairs(data.Dataset):
  def __init__(self, root = '/path/to/frames/only/folder', iext = 'jpg', replicates = 1):
    #self.args = args
    self.is_cropped = False
    self.crop_size = [256, 256]
    self.render_size = [-1,-1]
    self.replicates = 1

    images = natsorted( glob( join(root, '*.' + iext) ) )
    self.image_list = []
    for i in range(len(images)-1):
        im1 = images[i]
        im2 = images[i+1]
        self.image_list += [ [ im1, im2 ] ]

    self.size = len(self.image_list)
    
    self.frame_size = read_gen(self.image_list[0][0]).shape


    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    #args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = read_gen(self.image_list[index][0])
    img2 = read_gen(self.image_list[index][1])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = list(map(cropper, images))
    
    images = np.array(images).transpose(3,0,1,2)
    images = torch.from_numpy(images.astype(np.float32))

    return [images], [torch.zeros(images.size()[0:1] + (2,) + images.size()[-2:])]

  def __len__(self):
    return self.size * self.replicates

'''
curr_dir =  '/user_data/vayzenbe/GitHub_Repos/LiMA'
hab_dataset = LoadImagePairs(f'{curr_dir}/Frames/Figure_23_Bulge')
trainloader = torch.utils.data.DataLoader(hab_dataset, batch_size=1, shuffle=True, num_workers = 2, pin_memory=True)

for data in trainloader:
    pdb.set_trace()
'''