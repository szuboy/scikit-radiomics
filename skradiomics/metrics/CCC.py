
import numpy as np
from .utils import check_column_or_1d


def CCC(x, y):
    x, y = check_column_or_1d(x, y)

    ux, sigma_x, uy, sigma_y = np.mean(x), np.var(x), np.mean(y), np.var(y)

    mean_square = np.mean(np.square(x - y))
    expected_square = sigma_x + sigma_y + np.square(ux - uy)
    ccc = 1 - mean_square / expected_square

    return ccc

