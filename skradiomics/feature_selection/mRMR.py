# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .utils import check_X_y
from .Information import mutual_info


def correlation(x1, x2):

    x1_2 = np.square(x1)
    x2_2 = np.square(x2)
    x1_x2 = x1 * x2

    u_x1 = np.mean(x1)
    u_x2 = np.mean(x2)

    s_x = np.sqrt(np.mean(x1_2) - np.square(u_x1))
    s_y = np.sqrt(np.mean(x2_2) - np.square(u_x2))
    s_xy = np.mean(x1_x2) - u_x1 * u_x2

    return s_xy / (s_x * s_y + 1e-8)


def mRMR(X, y):
    """
    Minimum Redundancy Maximum Relevance (mRMR) feature selection algorithm.
    paper: An Improved Minimum Redundancy Maximum Relevance Approach for Feature Selection in Gene Expression Data
    paper: Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy
    """
    X, y = check_X_y(X, y)

    n_samples, n_features = X.shape

    all_index = np.arange(n_features)
    selected_index = []

    feature_relevance = mutual_info(X, y)

    index = np.nanargmax(feature_relevance)
    selected_index.append(index)

    scores = np.zeros((n_features,))
    scores[index] = feature_relevance[index]

    for i in range(1, n_features):
        left_index = np.setdiff1d(all_index, selected_index)

        relevance = feature_relevance[left_index]
        redundancy = [np.mean([correlation(X[:, i], X[:, s]) for s in selected_index]) for i in left_index]

        metric = relevance - np.abs(redundancy)
        index = np.nanargmax(metric)

        scores[left_index[index]] = metric[index]
        selected_index.append(left_index[index])

    return scores


