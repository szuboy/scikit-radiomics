# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


import collections
import numpy as np
import SimpleITK as sitk

from .base import ngtdm_matrix
from skradiomics.utils.modules import FEATURE_EXTRACTORS


# @FEATURE_EXTRACTORS.register_module(name='ngtdm')
# class RadiomicsNGTDM(RadiomicsBase):
#     def __init__(self, **kwargs):
#         super(RadiomicsNGTDM, self).__init__(**kwargs)
#         pass


@FEATURE_EXTRACTORS.register_module(name='ngtdm')
def radiomics_ngtdm(image, mask, settings, **kwargs):
    feature_vector = collections.OrderedDict()

    epsilon = np.spacing(1)
    distances = settings.get('NGTDMDistance', [1])

    for distance in distances:
        kwargs['distance'] = distance

        p_ngtdm = ngtdm_matrix(image, mask, settings, **kwargs)

        # delete empty grey levels
        empty_grey_levels = np.where(p_ngtdm[:, 0] == 0)
        p_ngtdm = np.delete(p_ngtdm, empty_grey_levels, 0)

        # coefficient
        Nvp = np.nansum(p_ngtdm[:, 0])

        p_i = p_ngtdm[:, 0] / Nvp

        s_i = p_ngtdm[:, 1]

        i = p_ngtdm[:, 2]

        Ngp = np.sum(p_ngtdm[:, 0] > 0)

        p_zero = np.where(p_i[None, :] < 1e-5)

        # features
        sum_coarse = np.nansum(p_i * s_i)
        if sum_coarse != 0:
            sum_coarse = 1 / sum_coarse
        else:
            sum_coarse = 1e5
        feature_vector['Coarseness_Distance%s' % distance] = sum_coarse

        div = Ngp * (Ngp - 1)
        contrast = np.nansum(p_i[:, None] * p_i[None, :] * (i[:, None]- i[None, :]) ** 2) * np.sum(s_i) / Nvp
        if div != 0:
            contrast = contrast / div
        else:
            contrast = 0
        feature_vector['Contrast_Distance%s' % distance] = contrast

        i_pi = i * p_i
        abs_diff = np.abs(i_pi[:, None] - i_pi[None, :])

        abs_diff_sum = np.nansum(abs_diff)
        busyness = np.nansum(p_i * s_i)

        if abs_diff_sum != 0:
            busyness = busyness / abs_diff_sum
        else:
            busyness = 0
        feature_vector['Busyness_Distance%s' % distance] = busyness

        pi_si = p_i * s_i
        numerator = pi_si[:, None] + pi_si[None, :]

        divisor = p_i[:, None] + p_i[None, :]
        divisor[np.abs(divisor) < 1e-5] = 1

        complexity = np.nansum(np.abs(i[:, None] - i[None, :]) * numerator / divisor) / Nvp
        feature_vector['Complexity_Distance%s' % distance] = complexity

        sum_s_i = np.sum(s_i)
        strength = (p_i[:, None] + p_i[None, :]) * (i[:, None] - i[None, :]) ** 2
        strength = np.nansum(strength)
        if sum_s_i != 0:
            strength = strength / sum_s_i
        else:
            strength = 0
        feature_vector['Strength_Distance%s' % distance] = strength

    return feature_vector



