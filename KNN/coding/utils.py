#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: utils.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""

import numpy as np


def train_test_spilt(feature, labels, train_size=0.7):

    train_num = int(len(feature) * train_size)

    train_set, train_label = feature[:train_num, :], labels[:train_num]
    test_set, test_label = feature[train_num:len(feature), :], labels[train_num:len(feature)]

    print(train_set.shape, train_label.shape, test_set.shape, test_label.shape)

    return train_set, train_label, test_set, test_label


def minMax_Normalization(feature):
    X_pre = None

    data_max = np.max(feature, axis=0)
    data_min = np.min(feature, axis=0)
    X_pre = (feature - data_min) / (data_max - data_min)

    return X_pre


def zScore_Standardization(feature):
    X_pre = None

    mean = np.mean(feature, axis=0)
    std = np.std(feature, axis=0)
    X_pre = (feature - mean) / std

    return X_pre

def feature_preprocessing(feature, Normalized=True):
    X_pre = None
    if Normalized:
        data_max = np.max(feature, axis=0)
        data_min = np.min(feature, axis=0)

        X_pre = (feature - data_min) / (data_max - data_min)
    else:
        mean = np.mean(feature, axis=0)
        std = np.std(feature, axis=0)
        X_pre = (feature - mean) / std

    return X_pre


def Kfold_cross_validation(feature, labels, k_fold=4, random_state=True):
    nums = len(labels)
    temp_label = labels.reshape(-1, 1)
    dataSets = np.concatenate((feature, temp_label), axis=1)

    if random_state:
        dataSets = np.random.permutation(dataSets)

    size = nums // k_fold
    start_index = 0
    end_index = size

    for i in range(k_fold):
        if end_index > nums:
            end_index = nums

        train = np.concatenate((dataSets[0:start_index], dataSets[end_index:]))
        test = dataSets[start_index:end_index]

        start_index += size
        end_index += size

        train_set, train_label = train[:, :-1].astype(float), train[:, -1]
        test_set, test_label = test[:, :-1].astype(float), test[:, -1]

        # print(train_set.shape, train_label.shape, test_set.shape, test_label.shape)

        yield train_set, train_label, test_set, test_label
