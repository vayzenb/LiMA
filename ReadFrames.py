# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:04:18 2019

@author: VAYZENB
"""

import os
import cv2



os.chdir("R:/LourencoLab/Adult Studies/Shapes (All Experiments)/Scripts/LiMA")
vids = os.listdir("R:/LourencoLab/Adult Studies/Shapes (All Experiments)/Scripts/LiMA/Videos")

for ii in range(0,len(vids)):
    vidcap = cv2.VideoCapture('Videos/' + vids[ii])
    vidFile = vids[ii][:-4]
    success,image = vidcap.read()
    count = 1
    success = True
    while success:
      cv2.imwrite("frames/" + vidFile + "_" + str(count) + ".jpg", image)     # save frame as JPEG file
      success,image = vidcap.read()
      print 'Read a new frame: ', success
      count += 1