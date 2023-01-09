#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: knn.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
import statistics

import numpy as np
from utils import train_test_spilt, Kfold_cross_validation
import matplotlib.pyplot as plt

class KNearestNeighbor:
    def __init__(self, train_set, train_label, test_set, test_label, train_size=0.8, k=3):
        self.train_set = train_set
        self.train_label = train_label
        self.test_set = test_set
        self.test_label = test_label

        self.temp_train_set = train_set
        self.temp_train_label = train_label
        self.temp_test_set = test_set
        self.temp_test_label = test_label

        self.train_size = train_size
        self.k = k

        self.error = 0
        self.res = None

    def calc_dist(self, x):
        return np.sum((x - self.train_set) ** 2, axis=1) ** 0.5

    def fit(self, x):
        dist = self.calc_dist(x)
        sort_index = np.argsort(dist)

        count_dict = {}
        for i in range(self.k):
            # print(len(self.train_label), i, len(sort_index))
            count_dict[self.train_label[sort_index[i]]] = count_dict.get(self.train_label[sort_index[i]], 0) + 1
        count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)

        return count_dict[0][0]

    def predict(self, train_set, train_label, test_set, test_label):
        self.train_set = train_set
        self.train_label = train_label
        self.test_set = test_set
        self.test_label = test_label

        # print(train_label)
        # print(len(test_label))

        res_list = []
        for i in range(len(self.test_set)):
            res = self.fit(self.test_set[i])
            if res != self.test_label[i]:
                self.error += 1
            res_list.append(res)
        print('predict result:', res_list)
        print('standard result:', list(self.test_label))
        # print(res_list)
        self.res = np.array(res_list)
        return res_list


    def score(self):
        # print(self.error, len(self.test_label))
        # accuracy = (1 - (self.error / len(self.test_label))) * 100
        accu_num = np.sum(self.res == self.test_label)
        accuracy = accu_num / len(self.test_label) * 100
        print("accuracy: {}".format(accuracy))
        return accuracy

    def find_k_and_plot(self, ks):
        scores = []
        for k in ks:
            print("k = {}".format(k))
            self.k = k
            score = self.cross_val_score(cv=4, k=k).mean()
            scores.append(score)
            print('-----------------------------------')
        scores = np.array(scores)
        print('scores: ', scores)
        plt.plot(ks, scores)
        plt.xlabel('k')
        plt.ylabel('score%')
        plt.show()
        self.k = ks[np.argmax(scores)]
        return self.k

    def cross_val_score(self, cv=4, k=3):
        scores = []
        for train_set, train_label, test_set, test_label in Kfold_cross_validation(self.temp_train_set, self.temp_train_label, k_fold=cv):
            self.train_set = train_set
            self.train_label = train_label
            self.test_set = test_set
            self.test_label = test_label
            self.k = k
            self.predict(train_set, train_label, test_set, test_label)
            scores.append(self.score())
        print("average score:", np.array(scores).mean())
        return np.array(scores)