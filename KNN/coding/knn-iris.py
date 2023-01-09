#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: knn-iris.py
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
from sklearn.datasets import load_iris

@calc_runTime
def run():
    from sklearn.utils import shuffle
    datasets = load_iris()
    feature = np.array(datasets['data'])
    labels = np.array(datasets['target']).reshape(-1, 1)

    res = np.concatenate((feature, labels), axis=1)
    res = shuffle(res)

    feature = res[:, :-1]
    labels = res[:, -1]

    feature = feature_preprocessing(feature, True)

    knn = KNearestNeighbor(feature, labels, train_size=0.7, k=4)
    # knn.fit(knn.test_set[0])
    # knn.predict()
    # print("accuracy: {}".format(knn.score()))

    _, _, best_k = knn.find_k_and_plot([2 * i + 1 for i in range(1, 20)])
    print("best k:", best_k)

if __name__ == '__main__':
    run()