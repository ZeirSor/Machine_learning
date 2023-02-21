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

from Regresion import Regression

class LinearRegression(Regression):

    def __init__(self, w=None):
        # print('class LinearRegression init!')
        ...

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):

        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        self.x_train = x_train
        self.y_train = y_train

        self.w = np.linalg.pinv(x_train) @ y_train
        self.y_train_pred = x_train @ self.w

        # self.MSE = np.mean(np.square((y_train - self.y_train_pred)))
        self.show_train_res()


    def predict(self, x_test: np.ndarray, y_test: np.ndarray):
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        self.x_test = x_test
        self.y_test = y_test

        self.y_test_pred = x_test @ self.w
        self.show_pred_res(y_test)
        return self.y_test_pred

class RidgeRegression(Regression):

    def __init__(self, alpha=1.0):
        ...
        self.alpha = alpha

    def fit(self, x_train, y_train):
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        self.x_train = x_train
        self.y_train = y_train

        self.w = np.linalg.inv(self.alpha * np.eye(x_train.shape[1]) + x_train.T @ x_train) @ x_train.T @ y_train
        self.y_train_pred = x_train @ self.w

        self.show_train_res()


    def predict(self, x_test, y_test):
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        self.x_test = x_test
        self.y_test = y_test

        self.y_test_pred = x_test @ self.w
        self.show_pred_res(y_test)
        return self.y_test_pred



if __name__ == '__main__':
    ...