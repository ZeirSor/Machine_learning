#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: run.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from models import LinearRegression

def run():
    dataset = ds.load_boston()
    feature = dataset['data']
    target = dataset['target']
    # feature_names = dataset['feature_names']
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=2023)
    model = LinearRegression()
    model.fit(X_train, y_train)
    model.predict(X_test, y_test)



if __name__ == '__main__':
    run()

