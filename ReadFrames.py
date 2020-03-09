# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:04:18 2019

@author: VAYZENB
"""

import os
import cv2
from PIL import Image


#os.chdir("C:/Users/vayzenb\Desktop\GitHub Repos\LiMA")
vids = os.listdir("Videos/")


for ii in range(0,len(vids)):
    vidcap = cv2.VideoCapture('Videos/' + vids[ii])
    vidFile = vids[ii][:-4]
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
      cv2.imwrite("Frames/" + vidFile + "_" + str(count) + ".jpg", image)     # save frame as JPEG file
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1
      
frames = os.listdir("Frames/")

for ii in range(0,len(frames)):
    IM = Image.open("Frames/" + frames[ii]).convert("RGB")
    IM = IM.crop((370, 100, 1070, 800)) 
    IM.save("Frames/" + frames[ii])
    