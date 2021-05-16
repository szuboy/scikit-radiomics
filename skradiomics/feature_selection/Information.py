# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


import numpy as np

from .utils import check_X_y, feature_binary_partition_discretize


def elog(x):
    if x <= 0 or x >= 1:
        return 0
    else:
        return x * np.log(x)


def hist(x_list):
    d = dict()
    for s in x_list:
        d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z) / len(x_list), d.values())


def entropy(prob_list, base=2):
    return -sum(map(elog, prob_list)) / np.log(base)


def mutual_info(X, y):
    X, y = check_X_y(X, y)

    n_samples, n_features = X.shape

    score_func = lambda x: -entropy(hist(list(zip(x, y)))) + entropy(hist(x)) + entropy(hist(y))

    scores = []
    for i in range(n_features):
        x_i = X[:, i]
        discreteize_list = feature_binary_partition_discretize(x_i)
        score = np.nanmax(list(map(score_func, discreteize_list)))
        scores.append(score)

    return np.array(scores)


def info_gain(X, y):
    X, y = check_X_y(X, y)

    n_samples, n_features = X.shape

    score_func = lambda x: -entropy(hist(list(zip(x, y)))) + entropy(hist(x)) + entropy(hist(y))

    scores = []
    for i in range(n_features):
        x_i = X[:, i]
        discreteize_list = feature_binary_partition_discretize(x_i)

        score = np.nanmax(list(map(score_func, discreteize_list)))
        scores.append(score)

    return np.array(scores)


def info_gain_ratio(X, y):
    X, y = check_X_y(X, y)

    n_samples, n_features = X.shape

    score_func = lambda x: (-entropy(hist(list(zip(x, y)))) + entropy(hist(x)) + entropy(hist(y))) / (entropy(hist(x)) + 1e-5)

    scores = []
    for i in range(n_features):
        x_i = X[:, i]
        discreteize_list = feature_binary_partition_discretize(x_i)
        score = np.nanmax(list(map(score_func, discreteize_list)))
        scores.append(score)

    return np.array(scores)


