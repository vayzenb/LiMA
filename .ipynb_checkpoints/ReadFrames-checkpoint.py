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
    print(vids[ii])
    vidcap = cv2.VideoCapture('Videos/' + vids[ii])
    vidFile = vids[ii][:-4]
    os.mkdir("Frames/"+ vidFile)
    success,image = vidcap.read()
    count = 1
    success = True
    while success:    
      cv2.imwrite("Frames/" + vidFile + "/" + vidFile + "_" + str(count) + ".jpg", image)     # save frame as JPEG file
      success,image = vidcap.read()
      count += 1
      
    frames = os.listdir("Frames/" + vidFile + "/")

    for ii in range(0,len(frames)):
        IM = Image.open("Frames/" + vidFile + "/" + frames[ii]).convert("RGB")
        IM = IM.crop((370, 100, 1070, 800)) 
        IM.save("Frames/" + vidFile + "/"  + frames[ii])
    
      