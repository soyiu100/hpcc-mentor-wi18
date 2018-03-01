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
import PIL
from PIL import Image as img
from sklearn.ensemble import RandomForestClassifier as rfc


def img_debug(paths, rnd_order):
    sizes = np.zeros((paths.shape[0], 2))

    custom_loc = r'\Documents\HPCC WI18\datasets'
    default_path = os.path.expanduser('~') + custom_loc

    # path for hyak
    default_path = os.getcwd()

    img_path = r'/Images/n'
    img_path = r'/datasets/Images/n'
    for j in rnd_order[:paths.shape[0]]:
        curr_img = img.open(default_path + img_path
                            + str(paths[j, :][0])[3:-2])
        sizes[j, :] = np.array(curr_img).shape[0:2]
    sizes[:, 0].sort()
    sizes[:, 1].sort()
    print(sizes[:, 0])
    print(sizes[:, 1])
    return sizes


##### NOTES TO SELF: #####
# As reference, the keys in the train_data.mat
# dict_keys(['__header__', '__version__', '__globals__',
# 'train_info', 'train_fg_data', 'train_data'])

# loads data neccessary for training and testing
def load_data(fast_path, name1, name2):
    # this custom_loc is for loacl machine purposes
    custom_loc = r'\Documents\HPCC WI18\datasets'

    default_path = os.path.expanduser('~') + custom_loc

    # path for hyak
    default_path = os.getcwd()

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
    min_image_size = (100, 97)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    dtrain = {key_name1: [], key_name2: []}
    dtest = {key_name1: [], key_name2: []}

    # these paths are for loacl machine purposes
    minipath_train = default_path + r'\small_train_data.mat'
    fullpath_train = default_path + r'\train_data.mat'
    minipath_test = default_path + r'\small_test_data.mat'
    fullpath_test = default_path + r'\test_data.mat'

    fullpath_train = os.path.abspath('train_data.mat')
    fullpath_test = os.path.abspath('test_data.mat')

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
            train_img_num = 400
            test_img_num = 338
            # This is a smaller set classifying 4 breeds of dogs, including Chihuahuas and Japanese spaniels
            small_dtrain = {key_name1: x_train[0:train_img_num], key_name2: y_train[0:train_img_num]}
            small_dtest = {key_name1: x_test[0:test_img_num], key_name2: y_test[0:test_img_num]}
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
    img_path = r'/Images/n'
    img_path = r'/datasets/Images/n'

    # img_debug(x_train, pm_train)
    # img_debug(x_test, pm_test)

    train_img = np.zeros((x_train.shape[0], min_image_size[0] * min_image_size[1] * 3))
    for j in pm_train[:x_train.shape[0]]:
        curr_img = img.open(default_path + img_path
                            + str(x_train[j, :][0])[3:-2])
        resized_img = np.array(curr_img.resize(min_image_size, PIL.Image.ANTIALIAS)).ravel()
        if resized_img.shape[0] == min_image_size[0] * min_image_size[1] * 3:
            train_img[j, :] = resized_img
        else:
            faulty_path = img_path + str(x_train[j, :][0])[3:-2]
            print("Erroneous image:", faulty_path)

    dtrain[key_name1] = train_img
    dtrain[key_name2] = [y_train[j, :][0] for j in pm_train[:x_train.shape[0]]]

    test_img = np.zeros((x_test.shape[0], min_image_size[0] * min_image_size[1] * 3))
    for j in pm_test[:x_test.shape[0]]:
        curr_img = img.open(default_path + img_path
                            + str(x_test[j, :][0])[3:-2])
        resized_img = np.array(curr_img.resize(min_image_size, PIL.Image.ANTIALIAS)).ravel()
        if resized_img.shape[0] == min_image_size[0] * min_image_size[1] * 3:
            test_img[j, :] = resized_img
        else:
            faulty_path = img_path + str(x_test[j, :][0])[3:-2]
            print("Erroneous image:", faulty_path)

    dtest[key_name1] = test_img
    dtest[key_name2] = [y_test[j, :][0] for j in pm_test[:x_test.shape[0]]]

    return dtrain, dtest, key_name1, key_name2


def test_train(dtrain, dtest, key_name1, key_name2, currAvg):
    # results testing around with small data sets showed that
    # accuracy was higher if max_depth was the number of classes
    randFrstClsf = rfc(max_depth=120)
    randFrstClsf.fit(dtrain[key_name1], dtrain[key_name2])

    result = randFrstClsf.predict(dtest[key_name1])
    corr = np.sum([dtest[key_name2][j] == result[j]
                   for j in range(0, len(dtest[key_name2]))]) / len(dtest[key_name2])
    print(corr * 100)
    currAvg += corr * 100
    return currAvg


if __name__ == '__main__':
    avgacc = 0
    totalReps = 20
    # true means that it will not go through all data; for local machine purposes
    data = load_data(False, 'data', 'labels')
    avgacc = test_train(data[0], data[1], data[2], data[3], 0)
    for i in range(2, totalReps + 1):
        avgacc = test_train(data[0], data[1], data[2], data[3], avgacc)
    print("Average Accuracy: " + str(avgacc / totalReps))
