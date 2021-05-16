# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


import numpy as np


def check_y_true_and_prob(y_true, y_prob):
    y_true = np.squeeze(np.array(y_true))
    y_prob = np.squeeze(np.array(y_prob))

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError('Expected y_true and y_prob is 1D array, but got %sD and %sD.' % (y_true.ndim, y_prob.ndim))

    if np.size(y_true) != np.size(y_prob):
        raise ValueError('Found input y_true and y_prob with inconsistent numbers of values.')

    if np.min(y_prob) < 0 or np.max(y_prob) > 1:
        raise ValueError('Expected y_prob contains values between 0 and 1.')

    y_label = np.unique(y_true)
    if np.size(y_label) != 2 or y_label[1] != 1:
        raise ValueError('Input y_true should only have 0 and 1, but got %s.' % str(y_label))

    return y_true, y_prob


def check_column_or_1d(x, y):
    x_array = np.squeeze(np.array(x))
    y_array = np.squeeze(np.array(y))

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError('Expected x and y is 1D array, but got %sD and %sD.' % (x_array.ndim, y_array.ndim))

    if np.size(x_array) != np.size(y_array):
        raise ValueError('Found input x and y with inconsistent numbers of values.')

    return x_array, y_array

