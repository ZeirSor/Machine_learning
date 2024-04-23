#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: mkdir_.py
@Software: PyCharm
@Github  : https://github.com/ZeirSor
@Description: 
    Create subdirectories 'notebook', 'coding', 'datasets' for all folders in the current directory
"""
import os.path

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

print(ABS_PATH)

dir_list = list(filter(lambda x: os.path.isdir(x) and x[0] != '.', os.listdir(ABS_PATH)))

print(dir_list)


def mkdir_():
    for dir in dir_list:
        name = ['notebook', 'coding', 'datasets']
        for dir_name in name:
            path = os.path.join(ABS_PATH, dir, dir_name).replace("\\", '/')
            print(path)
            if not os.path.exists(path):
                os.mkdir(path)

def run():
    mkdir_()

if __name__ == '__main__':
    run()