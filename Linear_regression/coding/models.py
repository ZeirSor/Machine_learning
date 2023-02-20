#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: models.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

import numpy as np

class LinearRegression:

    def __init__(self, w=None):
        # print('class LinearRegression init!')
        ...

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        ones = np.ones(shape=(len(X_train), 1))
        X_train = np.concatenate((X_train, ones), axis=1)
        self.X_train = X_train
        self.y_train = y_train
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        self.w = np.linalg.pinv(X_train) @ y_train
        self.y_train_pred = X_train @ self.w

        # self.MSE = np.mean(np.square((y_train - self.y_train_pred)))
        self.intercept = self.w[-1]
        self.MSE = MSE(y_train, self.y_train_pred)
        self.R2 = r2_score(y_train, self.y_train_pred)

        self.show_train_res()


    def predict(self, X_test: np.ndarray, y_test: np.ndarray):
        ones = np.ones(shape=(len(X_test), 1))
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        X_test = np.concatenate((X_test, ones), axis=1)
        self.y_test = y_test
        self.y_pred = X_test @ self.w

        self.show_pred_res(y_test)

    def show_train_res(self):
        print("train has completed:")
        print('\tw: {}'.format(list(self.w[:-1].reshape(-1))))
        print('\tintercept: {}'.format(self.intercept[0]))
        print('\tMSE: {}'.format(self.MSE))
        print('\tR2: {}'.format(self.R2))

    def show_pred_res(self, y_test: np.ndarray):
        print("pred has completed:")
        print('\tMSE: {}'.format(MSE(y_test, self.y_pred)))
        print('\tR2: {}'.format(r2_score(y_test, self.y_pred)))

if __name__ == '__main__':
    ...