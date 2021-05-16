# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import numpy as np

from .utils import check_X_y


def fisher_score(X, y):
    X, y = check_X_y(X, y, ensure_min_samples=3)

    n_sample, n_feature = X.shape

    scores = []
    for i in range(n_feature):
        x_i = X[:, i]
        u_i = np.mean(x_i)

        m, d = 0, 0
        for c in np.unique(y):
            d = d + np.size(y == c) * np.square(np.mean(x_i[y == c]) - u_i)
            m = m + np.size(y == c) * np.var(x_i[y == c])
        scores.append(d / (m + 1e-5))

    return np.asarray(scores)

