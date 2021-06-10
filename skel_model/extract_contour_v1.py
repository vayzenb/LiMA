# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:10:51 2021

@author: vayze
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import cv2 as cv

stim_folder = "C:/Users/vayze/Desktop/GitHub_Repos/LiMA/Frames"
skel = "23"
sf = 'Bulge'

filename = f'{stim_folder}/Figure_{skel}_{sf}/Figure_{skel}_{sf}_12.jpg'

im = Image.open(filename).convert('L') 
img = cv.imread(filename,0)
img = cv.blur(img, (4,4))

npim = np.array(im)

im2 = im.filter(ImageFilter.FIND_EDGES)

im3 = im2.convert('1')
#im3[im3 >0] = 1
edges = cv.Canny(img,100,200)
plt.imshow(edges)

thresh = cv.adaptiveThreshold(edges,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
    
    
_, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#contours = contours[0] if len(contours) == 2 else contours[1]
#big_contour = max(contours, key=cv.contourArea)

# draw white filled contour on black background
#result = np.zeros_like(img)
#cv.drawContours(result, [big_contour], 0, (255,255,255), cv.FILLED)

#plt.imshow(result)

'''
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
hh, ww = img.shape[:2]

# threshold
thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]

# get the (largest) contour
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)

# draw white filled contour on black background
result = np.zeros_like(img)
cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)

# save results
cv2.imwrite('knife_edge_result.jpg', result)

cv.imshow('result', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
'''