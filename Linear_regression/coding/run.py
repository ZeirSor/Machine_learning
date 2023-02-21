#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: run.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
from utils import add_ones_and_spilt

import numpy as np
import sklearn.datasets as ds
from models import LinearRegression, RidgeRegression

def run():
    dataset = ds.load_boston()
    feature = dataset['data']
    target = dataset['target']
    # feature_names = dataset['feature_names']
    x_train, x_test, y_train, y_test = add_ones_and_spilt(feature, target, test_size=0.2, random_state=2023)

    print('*' * 30 + ' Linear Regression ' + '*' * 30)
    model_linear = LinearRegression()
    model_linear.fit(x_train, y_train)
    model_linear.predict(x_test, y_test)

    print('*' * 30 + ' Ridge Regression ' + '*' * 30)
    model_Ridge = RidgeRegression(alpha=1)
    model_Ridge.fit(x_train, y_train)
    model_Ridge.predict(x_test, y_test)


if __name__ == '__main__':
    run()

