# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:10:51 2021

@author: vayze
"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
#import cv2 as cv
from skimage import io
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.feature import canny
from skimage.filters import threshold_otsu, threshold_local
from scipy import ndimage as ndi
from skimage.segmentation import flood, flood_fill
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.filters import sobel, gaussian

stim_folder = "C:/Users/vayze/Desktop/GitHub_Repos/LiMA/Frames"
skel = "23"
sf = 'Bulge'

filename = f'{stim_folder}/Figure_{skel}_{sf}/Figure_{skel}_{sf}_12.jpg'

im = io.imread(filename)
im = rgb2gray(im)
#thresh = threshold_otsu(im)
#binary = im > thresh


img = img_as_float(im)
filtered_img = gaussian(img, sigma=3)
# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(filtered_img, mu=0.25, lambda1=.5, lambda2=1, tol=1e-3, max_iter=200,
               dt=0.5, init_level_set='checkerboard', extended_output=True)

silh = cv[0].astype(int)
plt.imshow(silh)


'''

hist, hist_centers = histogram(im)

edges = canny(im,3)

plt.imshow(edges)
edges = edges.astype(int)

#This fills in all holes, even between the legs
#filled = ndi.binary_fill_holes(edges)
edges[edges ==1] = 2
#try using flood fill
light_coat = flood_fill(edges, (200, 130),1)

plt.imshow(light_coat, cmap=plt.cm.gray)

from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

image = img_as_float(io.imread(filename))
gimage = inverse_gaussian_gradient(image)

# Initial level set
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1
# List with intermediate results for plotting the evolution
evolution = []
#callback = store_evolution_in(evolution)
ls = morphological_geodesic_active_contour(gimage, 230, init_ls,
                                           smoothing=1, balloon=-1,
                                           threshold=0.69)

'''