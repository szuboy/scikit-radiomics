# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import numpy as np


def check_X_y(X, y, accept_sparse=False, ensure_min_samples=1, ensure_min_features=1):
    """
    Input validation for standard estimators.
    Checks X and y for consistent length, enforces X 2d and y 1d.
    Standard input checks are only applied to y, such as checking that y does not have np.nan or np.inf targets.
    """
    X_array = np.array(X)

    if X_array.ndim == 1:
        raise ValueError('Expected X is 2D array, but got 1D array.')

    X_array = np.atleast_2d(X_array)
    if X_array.ndim >= 3:
        raise ValueError('Found array with dim %d. expected dim=2.' % X_array.ndim)

    n_samples, n_features = X_array.shape

    if ensure_min_samples > 0:
        if n_samples < ensure_min_samples:
            raise ValueError('Found X with %d sample(s), but minimum %d is required.' % (n_samples, ensure_min_samples))

    if ensure_min_features > 0:
        if n_features < ensure_min_features:
            raise ValueError('Found X with %d feature(s), but minimum %d is required.' % (n_samples, ensure_min_features))

    y_array = np.array(y)
    y_array = np.squeeze(y_array)
    if y_array.ndim != 1:
        raise ValueError('Expected y is 1D array, but got %sD array' % y_array.ndim)

    if not np.isfinite(np.sum(X_array)) or np.isnan(X_array).any() or not np.isfinite(X_array).all():
        raise ValueError('Input X contains NaN, infinity or a value too large')
    if not np.isfinite(np.sum(y_array)) or np.isnan(y_array).any() or not np.isfinite(y_array).all():
        raise ValueError('Input y contains NaN, infinity or a value too large')

    if n_samples != np.size(y_array):
        raise ValueError('Found input X and y with inconsistent numbers of samples')

    y_label = np.unique(y_array)
    if np.size(y_label) != 2 or y_label[1] != 1:
        raise ValueError('Input y should only have 0 and 1, but got %s.' % str(y_label))

    return X_array, y_array


def check_x_y(x, y, accept_sparse=False):
    """
    Input validation for feature to feature or feature to label
    Checks X and y for consistent length, enforces x 1d and y 1d.
    """
    x_array = np.array(x)
    x_array = np.squeeze(x_array)

    if x_array.ndim != 1:
        raise ValueError('Expected x is 1D array, but got %sD array' % x_array.ndim)

    y_array = np.array(y)
    y_array = np.squeeze(y_array)

    if y_array.ndim != 1:
        raise ValueError('Expected y is 1D array, but got %sD array' % y_array.ndim)

    if np.size(x_array) != np.size(y_array):
        raise ValueError('Found input x and y with inconsistent numbers of values')

    y_label = np.unique(y_array)
    if np.size(y_label) != 2 or y_label[1] != 1:
        raise ValueError('Input y should only have 0 and 1, but got %s.' % str(y_label))

    return x_array, y_array


def feature_binary_partition_discretize(x):
    x = np.squeeze(x)

    if x.ndim != 1:
        raise ValueError('Expected x is 1D array, but got %sD array' % x.ndim)

    unique_feature = np.unique(x)

    candidates = (unique_feature[:-1] + unique_feature[1:]) / 2

    if np.size(candidates) == 0:
        return [np.zeros_like(x)]

    discretize_feature_list = []
    for t in candidates:
        one_hot_feature = np.zeros_like(x)
        one_hot_feature[x > t] = 1
        discretize_feature_list.append(one_hot_feature)

    return discretize_feature_list


