# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:52:08 2018

@author: Isaac Pang
"""
# to run this, requires dog datasets (the training and testing .mat files)
# from http://vision.stanford.edu/aditya86/ImageNetDogs/

import os
from scipy import io as scio
import numpy as np
import imageio as imio
from sklearn.ensemble import RandomForestClassifier as rfc

##### NOTES TO SELF: #####
# As reference, the keys in the train_data.mat
# dict_keys(['__header__', '__version__', '__globals__',
# 'train_info', 'train_fg_data', 'train_data'])

# loads data neccessary for training and testing
def load_data(fast_path, name1, name2):
    custom_loc = r'\Documents\HPCC WI18\datasets'
    default_path = os.path.expanduser('~') + custom_loc
    
    data_train_name = 'train_info'
    data_test_name = 'test_info'
    key_name1 = name1
    key_name2 = name2
    
    ### NON-OPTIMAL TESTING -- PIXEL BASED TESTING ###
    # when cropping images, the remaining pixels should be split 1:2, with
    # left:right and top:bottom. Most dog images have their heads in the upper 
    # left, sometimes centered, and occassionally in the lower-right or 
    # upper-right
    
    # this is the smallest dimensions without scaling any of given images
    # set up to save unnecessary calculations
    min_image_size = (97,100)
    
    dtrain = {key_name1: None, key_name2: None}
    dtest = {key_name1: None, key_name2: None}
    shrink_coeff = 0.001
    minipath_train = default_path + r'\small_train_data.mat'
    fullpath_train = default_path + r'\train_data.mat'
    minipath_test = default_path + r'\small_test_data.mat'
    fullpath_test = default_path + r'\test_data.mat'
    if os.path.isfile(minipath_train) and os.path.isfile(minipath_test) and fast_path:
        dtrain[key_name1] = scio.loadmat(minipath_train)[key_name1]
        dtrain[key_name2] = scio.loadmat(minipath_train)[key_name2]
        dtest[key_name1] = scio.loadmat(minipath_test)[key_name1]
        dtest[key_name2] = scio.loadmat(minipath_test)[key_name2]
    elif os.path.isfile(fullpath_train) and os.path.isfile(fullpath_test):
        # [0][0][0] contains arrays of .jpgs
        # [0][0][1] contains the arrays of the names of the jpgs?
        # [0][0][2] contains the labels in the data
        # [0][0][3] contains the arrays of .mat of the .jpgs      
        # original .mat files
        temp_train = scio.loadmat(fullpath_train)
        temp_test = scio.loadmat(fullpath_test)
        # original data
        x_train = temp_train[data_train_name][0][0][0]
        # original results, used for fitting
        y_train = temp_train[data_train_name][0][0][2]
        x_test =  temp_test[data_test_name][0][0][0]
        y_test = temp_test[data_test_name][0][0][2]
        # needs to cast to int because numpy array
        # first dimension in x_* is dimensions, and other is the data
        if fast_path:
            train_size = int(np.floor(x_train[:,0].shape[0]*shrink_coeff))
            test_size = int(np.floor(x_test[:,0].shape[0]*shrink_coeff))
        else:
            train_size = x_train[:,0].shape[0]
            test_size = x_test[:,0].shape[0]
        
        pm_train = np.random.permutation(train_size)
        pm_test = np.random.permutation(test_size)
        # TODO: this is a rough implementation created just to get it working
        # TODO: will need get actual ORB stuff working in this part
        
        # TODO: main priority: get the fine-pixel comparison working
        dtrain[key_name1] = np.array([imio.imread(default_path + r'\Images\n' +
              str(x_train[i,:][0])[3:-2])[nico][nico].ravel() for i in pm_train[:train_size]])
        dtrain[key_name2] = np.array([y_train[i,:][0] 
            for i in pm_train[:train_size]]).ravel()
        dtest[key_name1] = np.array([imio.imread(default_path + r'\Images\n' + 
             str(x_test[i,:][0])[3:-2])[nico][nico].ravel() for i in pm_test[:test_size]])
        dtest[key_name2] = np.array([y_test[i,:][0] 
            for i in pm_train[:test_size]]).ravel()  
        scio.savemat(default_path + 'small_train_data', mdict=dtrain, appendmat=True)
        scio.savemat(default_path + 'small_test_data', mdict=dtest, appendmat=True)
    return (dtrain, dtest, key_name1, key_name2)
        
def test_train(dtrain, dtest, key_name1, key_name2):
    # TODO: play around with the random forest settings
    randFrstClsf = rfc(max_depth=2)      
    randFrstClsf.fit(dtrain[key_name1], dtrain[key_name2])
    
    randFrstClsf.predict(dtest[key_name1])

        
        
if __name__ == '__main__':
    # true means that it will not go through all data; for local machine purposes
    data = load_data(True, 'data', 'labels')
    test_train(data[0], data[1], data[2], data[3])
    