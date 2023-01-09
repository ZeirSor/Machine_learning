#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: knn-adults.py
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
import pandas as pd

@calc_runTime
def run(onehot=False):
    from sklearn.utils import shuffle
    if onehot:
        datasets = pd.read_csv('../datasets/adults.txt')
        shuffle(datasets)
        datasets = datasets[['age', 'education_num', 'occupation', 'hours_per_week', 'salary']]

        col_names = datasets.columns[datasets.columns != 'salary']
        feature = datasets[col_names]
        labels = np.array(datasets['salary'])

        datasets_oh = pd.get_dummies(feature['occupation'])
        feature = pd.concat((feature, datasets_oh), axis=1).drop(labels='occupation', axis=1)

        feature = np.array(feature)
        feature = feature_preprocessing(feature)
        print(type(feature), type(labels))

        knn = KNearestNeighbor(feature, labels, train_size=0.7, k=20)
        # knn.fit(knn.test_set[0])
        knn.predict()
        print("accuracy: {}".format(knn.score()))

        _, _ , best_k = knn.find_k_and_plot([2 * i + 1 for i in range(1, 20)])
        print("best k:", best_k)

    else:
        datasets = pd.read_csv('../datasets/adults.txt')
        shuffle(datasets)
        datasets = datasets[['age', 'education_num', 'occupation', 'hours_per_week', 'salary']]

        col_names = datasets.columns[datasets.columns != 'salary']
        feature = datasets[col_names]

        occupation_count = feature['occupation'].unique()
        count = 1
        occu_dic = {}
        for occ in occupation_count:
            occu_dic[occ] = count
            count += 1
        feature['occupation'] = feature['occupation'].map(occu_dic)
        feature = np.array(feature)

        labels = np.array(datasets['salary'])
        feature = feature_preprocessing(feature)
        print(type(feature), type(labels))

        knn = KNearestNeighbor(feature, labels, train_size=0.7, k=20)
        # knn.fit(knn.test_set[0])
        # knn.predict()
        # print("accuracy: {}".format(knn.score()))

        _, _ , best_k = knn.find_k_and_plot([2 * i + 1 for i in range(1, 40)])
        print("best k:", best_k)

if __name__ == '__main__':
    run(True)