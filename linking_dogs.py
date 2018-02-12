# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:47:51 2018

@author: Isaac Pang
"""

from skimage.feature import ORB, match_descriptors as md, plot_matches as pm
from skimage.color import rgb2gray
import os
import imageio as imio
import matplotlib.pyplot as plt
import numpy as np

custom_loc = r'\Documents\HPCC WI18\test_images'
default_path = os.path.expanduser('~') + custom_loc

# converted to grayscale so that the images could be better analyzed
original_im = rgb2gray(imio.imread(default_path + r'\original_husky.jpg'))
rotated_im = rgb2gray(imio.imread(default_path + r'\rotated_husky.jpg'))
flipped_im = rgb2gray(imio.imread(default_path + r'\flipped_husky.jpg'))

standard_size = original_im.shape

rotated_im_shape_diff = np.subtract(rotated_im.shape, original_im.shape)
rotated_im = rotated_im[int(rotated_im_shape_diff[0]/2):333+int(rotated_im_shape_diff[0]/2), 
                        int(rotated_im_shape_diff[1]/2):500+int(rotated_im_shape_diff[1]/2)]

flip_im_shape_diff = np.subtract(flipped_im.shape, original_im.shape)
flipped_im = flipped_im[int(flip_im_shape_diff[0]/2):333+int(flip_im_shape_diff[0]/2), 
                        int(flip_im_shape_diff[1]/2):500+int(flip_im_shape_diff[1]/2)]

# without any params, this ORB for these sepcific images go up to 500
for i in range(50, 501, 50):
    finder_orb = ORB(n_keypoints=i)
    
    finder_orb.detect_and_extract(original_im)
    keypoints1 = finder_orb.keypoints
    descriptors1 = finder_orb.descriptors
    
    finder_orb.detect_and_extract(rotated_im)
    keypoints2 = finder_orb.keypoints
    descriptors2 = finder_orb.descriptors
    
    finder_orb.detect_and_extract(flipped_im)
    keypoints3 = finder_orb.keypoints
    descriptors3 = finder_orb.descriptors
   
    match_or = md(descriptors1, descriptors2)
    match_of = md(descriptors1, descriptors3)
    
    fig, ax = plt.subplots(nrows=2, ncols=1)
        
    pm(ax[0], original_im, rotated_im, keypoints1, keypoints2, match_or)
    pm(ax[1], original_im, flipped_im, keypoints1, keypoints3, match_of)
        
    plt.show()
    
    