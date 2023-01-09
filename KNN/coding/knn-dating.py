#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: knn-dating.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
from decorators import calc_runTime
from loading import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from knn import KNearestNeighbor

@calc_runTime
def run():
    with open('../datasets/datingTestSet.txt') as file:
        dataSets_list = []
        for line in file:
            dataSets_list.append(line.strip().split())
    dataSets = np.array(shuffle(dataSets_list))
    feature = dataSets[:, :-1].astype(float)
    labels = np.array(dataSets[:, -1])
    feature = feature_preprocessing(feature)

    knn = KNearestNeighbor(feature, labels, train_size=0.7, k=10)
    # # knn.fit(knn.test_set[0])
    # knn.predict()
    # print("accuracy: {}".format(knn.score()))
    #
    _, _, best_k = knn.find_k_and_plot([2 * i + 1 for i in range(1, 20)])
    print("best k:", best_k)

    # for train_set, train_label, test_set, test_label in Kfold_cross_validation(feature, labels):
    #     print("train_set:", train_set.shape)
    #     print("train_label", train_label.shape)
    #     print("test_set", test_set.shape)
    #     print("test_label", test_label.shape)
if __name__ == '__main__':
    run()