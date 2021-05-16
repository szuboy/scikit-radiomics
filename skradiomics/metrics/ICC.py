
import numpy as np
from .utils import check_column_or_1d


def ICC(x, y, model='one_way_random', definition='absolute', type='single'):
    """
    model: one_way_random, two_way_random, two_way_mixed
    type: single, mean
    definition: absolute, consistency
    """
    x, y = check_column_or_1d(x, y)

    xy = np.concatenate((x, y))
    x, y = x - np.mean(xy), y - np.mean(xy)

    # sum of squares decomposition formula
    ss_total = np.sum(np.square(x)) + np.sum(np.square(y)) - np.square(np.sum(x) + np.sum(y)) / (np.size(xy))

    # difference between groups, residual, in-groups, random
    ss_bn = (np.square(np.sum(x)) + np.square(np.sum(y))) / np.size(x)
    ss_wn = np.sum(np.square((x + y) / 2)) * 2
    ss_residual = ss_total - ss_bn
    ss_error = ss_total - ss_bn - ss_wn

    # df
    n_group, n_object = 2, np.size(x)
    df_bn, df_wn, df_error, df_residual = n_group - 1, np.size(x) - 1, (n_group - 1) * (n_object - 1), n_object - n_group

    # statistic
    ms_b, ms_w, ms_error, ms_residual = ss_bn / df_bn, ss_wn / df_wn, ss_error / df_error, ss_residual / df_residual

    if model == 'one_way_random':
        if definition == 'absolute':
            if type == 'single':
                icc = (ms_w - ms_residual) / (ms_w + (n_group + 1) * ms_residual)
            else:
                icc = (ms_w - ms_residual) / ms_w
        else:
            raise ValueError('one_way_random model matches with absolute definition, but get %s' % definition)

    elif model == 'two_way_random':
        if definition == 'absolute':
            if type == 'single':
                icc = (ms_w - ms_error) / (ms_w + (n_group - 1) * ms_error + n_group / n_object * (ms_b - ms_error))
            else:
                icc = (ms_w - ms_error) / (ms_w - (ms_b - ms_error) / n_object)
        elif definition == 'consistency':
            if type == 'single':
                icc = (ms_w - ms_error) / (ms_w + (n_group - 1) * ms_error)
            else:
                icc = (ms_w - ms_error) / ms_w
        else:
            raise ValueError('model matches with absolute or consistency definition, but get %s' % definition)

    elif model == 'two_way_mixed':
        if definition == 'absolute':
            if type == 'single':
                icc = (ms_w - ms_error) / (ms_w + (n_group - 1) * ms_error + n_group / n_object * (ms_b - ms_error))
            else:
                icc = (ms_w - ms_error) / (ms_w - (ms_b - ms_error) / n_object)
        elif definition == 'consistency':
            if type == 'single':
                icc = (ms_w - ms_error) / (ms_w + (n_group - 1) * ms_error)
            else:
                icc = (ms_w - ms_error) / ms_w
        else:
            raise ValueError('model matches with absolute or consistency definition, but get %s' % definition)

    else:
        icc = 0

    return icc



