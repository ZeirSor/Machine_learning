#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: knn-handwriting digit recognition.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
from decorators import calc_runTime
from loading import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from knn import KNearestNeighbor
import os


@calc_runTime
def run():
    feature = []
    labels = []
    path_list = []
    for i in range(10):
        for j in range(1, 501):
            img_path = os.path.join(r'..\datasets', "digist", str(i), str(i) + "_" + str(j) + ".bmp")
            img_arr = plt.imread(img_path)
            feature.append(img_arr)
            labels.append(i)
            path_list.append(img_path)
    print(path_list)

    feature = np.array(feature)
    feature = feature.reshape((5000, 784))
    labels = np.array(labels).reshape(-1, 1)

    dataSets = np.concatenate((feature, labels), axis=1)
    dataSets = np.random.permutation(dataSets)

    feature = dataSets[:, :-1]
    labels = dataSets[:, -1]

    train_set, train_label, test_set, test_label = train_test_spilt(feature, labels, train_size=0.8)

    knn = KNearestNeighbor(train_set, train_label, test_set, test_label, k=3)
    ks = np.arange(3, 100, 2)
    best_k = knn.find_k_and_plot(ks)
    print("best k:", best_k)

    knn.predict(train_set, train_label, test_set, test_label)
    knn.score()

if __name__ == '__main__':
    run()