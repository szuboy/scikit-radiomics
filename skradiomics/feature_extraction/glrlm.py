# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import collections
import numpy as np

from .base import glrlm_matrix
from skradiomics.utils.modules import FEATURE_EXTRACTORS


# @FEATURE_EXTRACTORS.register_module(name='glrlm')
# class RadiomicsGLRLM(RadiomicsBase):
#     def __init__(self, **kwargs):
#         super(RadiomicsGLRLM, self).__init__(**kwargs)
#         pass


@FEATURE_EXTRACTORS.register_module(name='glrlm')
def radiomics_glrlm(image, mask, settings, **kwargs):
    feature_vector = collections.OrderedDict()

    epsilon = np.spacing(1)
    weight_normal = settings.get('weightingNorm', None)

    p_glrlm, angles = glrlm_matrix(image, mask, settings, **kwargs)
    weights = np.ones((len(angles),))

    if weight_normal is not None:
        pixel_spacing = np.array(image.GetSpacing())
        for a_idx, a in enumerate(angles):
            if weight_normal == 'infinity':
                weights[a_idx] = np.exp(-max(np.abs(a) * pixel_spacing) ** 2)
            elif weight_normal == 'euclidean':
                weights[a_idx] = np.exp(-np.sum((np.abs(a) * pixel_spacing) ** 2))
            elif weight_normal == 'manhattan':
                weights[a_idx] = np.exp(-np.sum(np.abs(a) * pixel_spacing) ** 2)
            elif weight_normal == 'no_weighting':
                weights[a_idx] = 1
            else:
                weights[a_idx] = 1

    p_glrlm = np.sum(p_glrlm * weights, 2)

    Ng = p_glrlm.shape[0]
    Ng_vector = np.arange(1, Ng + 1)

    Nr = np.nansum(p_glrlm)

    pr = np.nansum(p_glrlm, 0)  # shape (Nr,)
    pg = np.nansum(p_glrlm, 1)  # shape (Ng,)

    i_vector = np.arange(1, Ng + 1)
    j_vector = np.arange(1, p_glrlm.shape[1] + 1)

    # delete columns that run lengths not present in the ROI
    empty_run_length = np.where(pr == 0)
    p_glrlm = np.delete(p_glrlm, empty_run_length, 1)
    j_vector = np.delete(j_vector, empty_run_length)
    pr = np.delete(pr, empty_run_length)

    short_run_emphasis = np.nansum(pr / np.square(j_vector)) / Nr
    feature_vector['ShortRunEmphasis'] = short_run_emphasis

    long_run_emphasis = np.nansum(pr * np.square(j_vector)) / Nr
    feature_vector['LongRunEmphasis'] = long_run_emphasis

    gray_level_non_uniformity = np.nansum(np.square(pg)) / Nr
    feature_vector['GrayLevelNonUniformity'] = gray_level_non_uniformity

    gray_level_non_uniformity_normalized = np.nansum(np.square(pg)) / np.square(Nr)
    feature_vector['GrayLevelNonUniformityNormalized'] = gray_level_non_uniformity_normalized

    run_length_non_uniformity = np.nansum(np.square(pr)) / Nr
    feature_vector['RunLengthNonUniformity'] = run_length_non_uniformity

    run_length_non_uniformity_normalized = np.nansum(np.square(pr)) / np.square(Nr)
    feature_vector['RunLengthNonUniformityNormalized'] = run_length_non_uniformity_normalized

    run_percentage = Nr / np.nansum(pr * j_vector)
    feature_vector['RunPercentage'] = run_percentage

    p_g = pg / Nr
    u_i = np.nansum(p_g * i_vector)
    gray_level_variance = np.nansum(p_g * np.square(i_vector - u_i))
    feature_vector['GrayLevelVariance'] = gray_level_variance

    p_r = pr / Nr
    u_j = np.nansum(p_r * j_vector)
    run_variance = np.nansum(p_r * np.square(j_vector - u_j))
    feature_vector['RunVariance'] = run_variance

    normal_p_glrlm = p_glrlm / Nr
    run_entropy = -1 * np.nansum(normal_p_glrlm * np.log2(normal_p_glrlm + epsilon))
    feature_vector['RunEntropy'] = run_entropy

    low_gray_level_run_emphasis = np.nansum(pg / np.square(i_vector)) / Nr
    feature_vector['LowGrayLevelRunEmphasis'] = low_gray_level_run_emphasis

    high_gray_level_run_emphasis = np.nansum(pg * np.square(i_vector)) / Nr
    feature_vector['HighGrayLevelRunEmphasis'] = high_gray_level_run_emphasis

    short_run_low_gray_level_emphasis = np.nansum(p_glrlm / (np.square(i_vector[:, None]) * np.square(j_vector[None, :]))) / Nr
    feature_vector['ShortRunLowGrayLevelEmphasis'] = short_run_low_gray_level_emphasis

    short_run_high_gray_level_emphasis = np.nansum(p_glrlm * (np.square(i_vector[:, None]) / np.square(j_vector[None, :]))) / Nr
    feature_vector['ShortRunHighGrayLevelEmphasis'] = short_run_high_gray_level_emphasis

    long_run_low_gray_level_emphasis = np.nansum(p_glrlm / (np.square(j_vector[None, :]) * np.square(i_vector[:, None]))) / Nr
    feature_vector['LongRunLowGrayLevelEmphasis'] = long_run_low_gray_level_emphasis

    long_run_high_gray_level_emphasis = np.nansum(p_glrlm * (np.square(j_vector[None, :]) * np.square(i_vector[:, None]))) / Nr
    feature_vector['LongRunHighGrayLevelEmphasis'] = long_run_high_gray_level_emphasis

    return feature_vector

















































