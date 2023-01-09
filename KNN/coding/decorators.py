#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : ZeirSor
@FileName: decorators.py
@Software: PyCharm
@Github    ：https://github.com/ZeirSor
@Description: 
"""
import functools
import time

def calc_runTime(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print("------ Function \"{}\" runs for {} seconds. ------".format(func.__name__, end_time - start_time))
        return res
    return inner