#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: knn-films.py
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
    datasets = load_datasets_excel('../datasets/knn_films.xlsx')
    feature = np.array(datasets[['Action Lens', 'Love Lens']])
    labels = np.array(datasets['target'])
    feature = feature_preprocessing(feature)

    knn = KNearestNeighbor(feature, labels, train_size=0.7)
    # # knn.fit(knn.test_set[0])
    # knn.predict()
    # print("accuracy: {}%".format(knn.score()))

    _, _, best_k = knn.find_k_and_plot([2 * i + 1 for i in range(1, 4)])
    print("best k:", best_k)
    # for train_set, train_label, test_set, test_label in Kfold_cross_validation(feature, labels):
    #     print("train_set:", train_set)
    #     print("train_label", train_label)
    #     print("test_set", test_set)
    #     print("test_label", test_label)

if __name__ == '__main__':
    run()