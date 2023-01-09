#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: loading.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
from sklearn.utils import shuffle
import pandas as pd

def load_datasets_excel(path='../datasets/knn_films.xlsx', random_state=True):
    datasets = pd.read_excel(path)

    if random_state:
        datasets = shuffle(datasets)
    return datasets



def load_datasets_csv(path='../datasets/adults.txt', random_state=True):
    datasets = pd.read_csv(path)

    if random_state:
        datasets = shuffle(datasets)

    return datasets