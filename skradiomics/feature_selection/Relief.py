# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


import numpy as np
from .utils import check_X_y


def Relief(X, y):
    X, y = check_X_y(X, y, ensure_min_samples=4)

    if np.sum(y == 0) < 2 or np.sum(y == 1) < 2:
        raise ValueError('Input y should have at least 2 samples for each class')

    n_sample, n_feature = X.shape

    # initial weight all is zero
    weight = np.zeros((n_feature,))

    # feature range
    max_min = np.max(X, axis=0) - np.min(X, axis=0) + np.spacing(1)

    # standard X
    X = (X - np.min(X, axis=0)) / max_min

    # sample index array
    all_index = np.arange(n_sample)

    for i in range(n_sample):
        label = y[i]
        sample = X[i, :]

        left_index = all_index[all_index != i]

        left_X = X[left_index, :]
        left_y = y[left_index]

        # calculate difference
        difference = np.abs(left_X - sample)

        # find the near hit and near miss sample
        candidate_index = np.arange(n_sample - 1)
        near_index = candidate_index[left_y == label]
        hit = np.nanargmin(np.sum(difference[near_index, :], axis=1))
        miss_index = candidate_index[left_y != label]
        miss = np.nanargmin(np.sum(difference[miss_index, :], axis=1))

        # update weight
        weight = weight + (np.abs(X[i, :] - X[miss_index[miss], :]) - np.abs(X[i, :] - X[near_index[hit], :]))

    return weight

