# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


import numpy as np
from scipy import special

from .utils import check_x_y, check_X_y


def F_test(x, y):
    """
    x: one feature list or 1d-array
    y: classification label corresponding to x, type: list or 1d-array
    link: https://en.wikipedia.org/wiki/One-way_analysis_of_variance
    """
    x, y = check_x_y(x, y, accept_sparse=False)

    # class numbers
    y_class = np.unique(y)
    assert np.size(y_class) != 1, 'Found y only has one label.'

    # x - np.mean(x), that makes mean(x) equal to zero
    x = x - np.mean(x)

    # sum of squares decomposition formula
    ss_total = np.sum(np.square(x)) - np.square(np.sum(x)) / np.size(x)

    # difference between groups
    ss_bn = 0
    for label in y_class:
        ss_bn = ss_bn + np.square(np.sum(x[y == label])) / np.size(x[y == label])
    ss_wn = ss_total - ss_bn

    # epsilon
    eps = np.spacing(1)

    # statistic
    n_label = np.size(y_class)
    df_bn = n_label - 1
    df_wn = np.size(x) - n_label
    ms_b = ss_bn / df_bn
    ms_w = ss_wn / df_wn
    f = ms_b / (ms_w + eps)
    prob = special.fdtrc(df_bn, df_wn, f)

    return f, prob


def ANOVA(X, y):

    X, y = check_X_y(X, y, ensure_min_samples=3)
    n_sample, n_feature = X.shape

    scores, p_values = [], []
    for i in range(n_feature):
        score, p_value = F_test(X[:, i], y)
        scores.append(score)
        p_values.append(p_value)

    return np.array(scores), np.array(p_values)

