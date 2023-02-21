#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: Regresion.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

class Regression:

    def __init__(self):
        ...


    def fit(self):
        ...

    def predict(self):
        ...

    def show_train_res(self):
        self.intercept = self.w[-1]
        self.MSE = MSE(self.y_train, self.y_train_pred)
        self.R2 = r2_score(self.y_train, self.y_train_pred)
        print("train has completed:")
        print('\tw: {}'.format(list(self.w[:-1].reshape(-1))))
        print('\tintercept: {}'.format(self.intercept[0]))
        print('\tMSE: {}'.format(self.MSE))
        print('\tR2: {}'.format(self.R2))

    def show_pred_res(self, y_test):
        print("pred has completed:")
        print('\tMSE: {}'.format(MSE(y_test, self.y_test_pred)))
        print('\tR2: {}'.format(r2_score(y_test, self.y_test_pred)))