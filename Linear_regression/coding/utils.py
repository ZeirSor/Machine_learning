#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: utils.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
from sklearn.model_selection import train_test_split

def add_ones_and_spilt(feature, target, test_size=0.2, random_state=2023):
    import numpy as np
    ones = np.ones(shape=(len(feature), 1))
    feature = np.concatenate((feature, ones), axis=1)
    return train_test_split(feature, target, test_size=0.2, random_state=2023)
