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

    ### NON-OPTIMAL TESTING -- FINE-GRAIN PIXEL-BY-PIXEL TESTING ###
    # when cropping images, the remaining pixels should be split 1:2, with
    # left:right and top:bottom. Most dog images have their heads in the upper 
    # left, sometimes centered, and occassionally in the lower-right or 
    # upper-right

    # this is the smallest dimensions without scaling any of given images
    # set up to save unnecessary calculations
    min_image_size = (97, 100)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    dtrain = {key_name1: [], key_name2: []}
    dtest = {key_name1: [], key_name2: []}
    minipath_train = default_path + r'\small_train_data.mat'
    fullpath_train = default_path + r'\train_data.mat'
    minipath_test = default_path + r'\small_test_data.mat'
    fullpath_test = default_path + r'\test_data.mat'
    if os.path.isfile(minipath_train) and os.path.isfile(minipath_test) and fast_path:
        x_train = scio.loadmat(minipath_train)[key_name1]
        y_train = scio.loadmat(minipath_train)[key_name2]
        x_test = scio.loadmat(minipath_test)[key_name1]
        y_test = scio.loadmat(minipath_test)[key_name2]
    elif os.path.isfile(fullpath_train) and os.path.isfile(fullpath_test):
        # This is if the smaller sets for testing have not been created yet
        # or the whole data is to be tested on

        # [0][0][0] contains arrays of names with .jpgs
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
        x_test = temp_test[data_test_name][0][0][0]
        y_test = temp_test[data_test_name][0][0][2]
        # needs to cast to int because numpy array
        # first dimension in x_* is dimensions, and other is the data
        if fast_path:
            # This is a smaller set classifying Chihuahuas and Japanese spaniels
            small_dtrain = {key_name1: x_train[0:200], key_name2: y_train[0:200]}
            small_dtest = {key_name1: x_test[0:137], key_name2: y_test[0:137]}
            scio.savemat(default_path + r'\small_train_data', mdict=small_dtrain, appendmat=True)
            scio.savemat(default_path + r'\small_test_data', mdict=small_dtest, appendmat=True)
            x_train = small_dtrain[key_name1]
            y_train = small_dtrain[key_name2]
            x_test = small_dtest[key_name1]
            y_test = small_dtest[key_name2]

            # train_size = x_train[:,0].shape[0]
            # test_size = x_test[:,0].shape[0]

        # if this doesn't work for both cases, then uncomment the thing above and make a if else case

    pm_train = np.random.permutation(x_train.shape[0])
    pm_test = np.random.permutation(x_test.shape[0])
    # TODO: this is a rough implementation created just to get it working
    # TODO: will need get actual ORB stuff working in this part

    # TODO: main priority: get the fine-grained comparison working
    train_img = np.zeros((x_train.shape[0], 29100))
    for i in pm_train[:x_train.shape[0]]:
        curr_img = imio.imread(default_path + r'\Images\n'
                               + str(x_train[i, :][0])[3:-2])
        shape_diff = np.subtract(curr_img.shape[0:2], min_image_size)
        train_img[i] = curr_img[int(shape_diff[0] / 3):min_image_size[0]
                    + int(shape_diff[0] / 3), int(shape_diff[1] / 3):
                    min_image_size[1] + int(shape_diff[1] / 3), :].ravel()

    # quick change; took it out of the numpy cast	
    dtrain[key_name1] = train_img
    dtrain[key_name2] = [y_train[i, :][0] for i in pm_train[:x_train.shape[0]]]
    test_img = np.zeros((x_test.shape[0], 29100))
    for i in pm_test[:x_test.shape[0]]:
        curr_img = imio.imread(default_path + r'\Images\n'
                               + str(x_test[i, :][0])[3:-2])
        shape_diff = np.subtract(curr_img.shape[0:2], min_image_size)
        test_img[i] = curr_img[int(shape_diff[0] / 3):min_image_size[0]
                    + int(shape_diff[0] / 3), int(shape_diff[1] / 3):min_image_size[1]
                    + int(shape_diff[1] / 3), :].ravel()

    # quick change; took it out of the numpy cast	
    dtest[key_name1] = test_img
    dtest[key_name2] = [y_test[i, :][0] for i in pm_test[:x_test.shape[0]]]

    return dtrain, dtest, key_name1, key_name2


def test_train(dtrain, dtest, key_name1, key_name2, currAvg):
    # TODO: play around with the random forest settings
    randFrstClsf = rfc(max_depth=2)
    randFrstClsf.fit(dtrain[key_name1], dtrain[key_name2])

    result = randFrstClsf.predict(dtest[key_name1])
    corr = np.sum([dtest[key_name2][j] == result[j]
                   for j in range(0, len(dtest[key_name2]))])/len(dtest[key_name2])
    print(corr * 100)
    currAvg += corr * 100
    return currAvg


if __name__ == '__main__':
    avgacc = 0
    totalReps = 10
    # true means that it will not go through all data; for local machine purposes
    data = load_data(True, 'data', 'labels')
    avgacc = test_train(data[0], data[1], data[2], data[3], 0)
    for i in range(2, totalReps + 1):
        # true means that it will not go through all data; for local machine purposes
        data = load_data(True, 'data', 'labels')
        avgacc = test_train(data[0], data[1], data[2], data[3], avgacc)
    print("Average Accuracy: " + str(avgacc/totalReps))


