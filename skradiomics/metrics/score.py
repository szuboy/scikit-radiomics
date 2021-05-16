# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import numpy as np
from .utils import check_y_true_and_prob


def brier_score(y_true, y_prob):
    """
    reference: https://en.wikipedia.org/wiki/Brier_score
    """
    y_true, y_prob = check_y_true_and_prob(y_true, y_prob)

    return np.average((y_true - y_prob) ** 2)


