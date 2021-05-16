# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import collections
import numpy as np

from .base import glszm_matrix
from skradiomics.utils.modules import FEATURE_EXTRACTORS


# @FEATURE_EXTRACTORS.register_module(name='glszm')
# class RadiomicsGLSZM(RadiomicsBase):
#     def __init__(self, **kwargs):
#         super(RadiomicsGLSZM, self).__init__(**kwargs)
#         pass


@FEATURE_EXTRACTORS.register_module(name='glszm')
def radiomics_glszm(image, mask, settings, **kwargs):
    feature_vector = collections.OrderedDict()

    epsilon = np.spacing(1)

    p_glszm = glszm_matrix(image, mask, settings, **kwargs)
    Ng = p_glszm.shape[0]
    Ng_vector = np.arange(1, Ng + 1)

    ps = np.sum(p_glszm, 0)  # shape (Ns,)
    pg = np.sum(p_glszm, 1)  # shape (Ng,)

    i_vector = np.arange(1, Ng + 1)
    j_vector = np.arange(1, p_glszm.shape[1] + 1)

    Nz = max(np.sum(p_glszm), 1)

    Np = max(np.sum(ps * j_vector), 1)

    small_area_emphasis = np.nansum(ps / np.square(j_vector)) / Nz
    feature_vector['SmallAreaEmphasis'] = small_area_emphasis

    large_area_emphasis = np.nansum(ps * np.square(j_vector)) / Nz
    feature_vector['LargeAreaEmphasis'] = large_area_emphasis

    gray_level_non_uniformity = np.nansum(pg ** 2) / Nz
    feature_vector['GrayLevelNonUniformity'] = gray_level_non_uniformity

    gray_level_non_uniformity_normalized = np.nansum(pg ** 2) / Nz ** 2
    feature_vector['GrayLevelNonUniformityNormalized'] = gray_level_non_uniformity_normalized

    size_zone_non_uniformity = np.nansum(ps ** 2) / Nz
    feature_vector['SizeZoneNonUniformity'] = size_zone_non_uniformity

    size_zone_non_uniformity_normalized = np.sum(ps ** 2) / Nz ** 2
    feature_vector['SizeZoneNonUniformityNormalized'] = size_zone_non_uniformity_normalized

    zone_percentage = Nz / Np
    feature_vector['ZonePercentage'] = zone_percentage

    p_g = pg / Nz
    u_i = np.nansum(p_g * i_vector)
    gray_level_variance = np.nansum(p_g * (i_vector - u_i) ** 2)
    feature_vector['GrayLevelVariance'] = gray_level_variance

    p_s = ps / Nz
    u_j = np.nansum(p_s * j_vector)
    zone_variance = np.nansum(p_s * (j_vector - u_j) ** 2)
    feature_vector['ZoneVariance'] = zone_variance

    normal_p_glszm = p_glszm / Nz
    zone_entropy = - np.nansum(normal_p_glszm * np.log2(normal_p_glszm + epsilon))
    feature_vector['ZoneEntropy'] = zone_entropy

    low_gray_level_zone_emphasis = np.nansum(pg / np.square(i_vector)) / Nz
    feature_vector['LowGrayLevelZoneEmphasis'] = low_gray_level_zone_emphasis

    high_gray_level_zone_emphasis = np.nansum(pg * np.square(i_vector)) / Nz
    feature_vector['HighGrayLevelZoneEmphasis'] = high_gray_level_zone_emphasis

    small_area_low_gray_level_emphasis = np.nansum(p_glszm / (np.square(i_vector[:, None]) * np.square(j_vector[None, :]))) / Nz
    feature_vector['SmallAreaLowGrayLevelEmphasis'] = small_area_low_gray_level_emphasis

    small_area_high_gray_level_emphasis = np.nansum(p_glszm * (np.square(i_vector[:, None]) / np.square(j_vector[None, :]))) / Nz
    feature_vector['SmallAreaHighGrayLevelEmphasis'] = small_area_high_gray_level_emphasis

    large_area_low_gray_level_emphasis = np.nansum(p_glszm * (np.square(j_vector[None, :]) / np.square(i_vector[:, None]))) / Nz
    feature_vector['LargeAreaLowGrayLevelEmphasis'] = large_area_low_gray_level_emphasis

    large_area_high_gray_level_emphasis = np.nansum(p_glszm * (np.square(j_vector[None, :]) * np.square(i_vector[:, None]))) / Nz
    feature_vector['LargeAreaHighGrayLevelEmphasis'] = large_area_high_gray_level_emphasis

    return feature_vector



