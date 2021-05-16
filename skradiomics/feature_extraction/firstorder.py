# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import copy
import collections
import numpy as np
import SimpleITK as sitk

from .base import RadiomicsBase, bin_image
from skradiomics.utils.modules import FEATURE_EXTRACTORS

#
# @FEATURE_EXTRACTORS.register_module(name='firstorder')
# class RadiomicsFirstorder(RadiomicsBase):
#     def __init__(self, **kwargs):
#         super(RadiomicsFirstorder, self).__init__(**kwargs)


@FEATURE_EXTRACTORS.register_module(name='firstorder')
def radiomics_firstorder(image, mask, settings, **kwargs):
    feature_vector = collections.OrderedDict()

    spacing = image.GetSpacing()
    np_image = sitk.GetArrayFromImage(image)
    np_mask = sitk.GetArrayFromImage(mask)

    np_discretized_image, _ = bin_image(image, mask, settings, **kwargs)
    _, p_i = np.unique(np_discretized_image[np_mask != 0], return_counts=True)
    p_i = p_i.reshape((1, -1))

    sum_bins = np.sum(p_i, axis=1, keepdims=True)
    sum_bins[sum_bins == 0] = 1
    p_i = p_i / sum_bins

    np_roi_array = np_image[np_mask != 0]
    voxel_shift = settings.get('voxelArrayShift', 0)

    energy = np.nansum((np_roi_array + voxel_shift)**2)
    feature_vector['Energy'] = energy

    total_energy = energy * np.prod(spacing)
    feature_vector['TotalEnergy'] = total_energy

    epsilon = np.spacing(1)
    entropy = -1 * np.sum(p_i * np.log2(p_i + epsilon))
    feature_vector['Entropy'] = entropy

    minimum = np.nanmin(np_roi_array)
    feature_vector['Minimum'] = minimum

    maximum = np.nanmax(np_roi_array)
    feature_vector['Maximum'] = maximum

    mean_value = np.nanmean(np_roi_array)
    feature_vector['Mean'] = mean_value

    median_value = np.nanmedian(np_roi_array)
    feature_vector['Median'] = median_value

    percentile10 = np.nanpercentile(np_roi_array, 10)
    feature_vector['Percentile10'] = percentile10

    percentile90 = np.nanpercentile(np_roi_array, 90)
    feature_vector['Percentile90'] = percentile90

    interquartile_range = np.nanpercentile(np_roi_array, 75) - np.nanpercentile(np_roi_array, 25)
    feature_vector['InterquartileRange'] = interquartile_range

    voxel_range = maximum - minimum
    feature_vector['Range'] = voxel_range

    mean_absolute_deviation = np.nanmean(np.absolute(np_roi_array - mean_value))
    feature_vector['MeanAbsoluteDeviation'] = mean_absolute_deviation

    percentile_array = copy.deepcopy(np_roi_array)
    percentile_mask = ~np.isnan(percentile_array)
    percentile_mask[percentile_mask] = ((percentile_array - percentile10)[percentile_mask] < 0) | ((percentile_array - percentile90)[percentile_mask] > 0)
    percentile_array[percentile_mask] = np.nan
    robust_mean_absolute_deviation = np.nanmean(np.absolute(percentile_array - np.nanmean(percentile_array)))
    feature_vector['RobustMeanAbsoluteDeviation'] = robust_mean_absolute_deviation

    n_voxel = np.sum(~np.isnan(np_roi_array))
    root_mean_squared = np.sqrt(energy / n_voxel)
    feature_vector['RootMeanSquared'] = root_mean_squared

    standard_deviation = np.nanstd(np_roi_array)
    feature_vector['StandardDeviation'] = standard_deviation

    m2 = np.nanmean(np.power(np_roi_array - mean_value, 2))
    m3 = np.nanmean(np.power(np_roi_array - mean_value, 3))
    skewness = m3 / (m2 + epsilon) ** 1.5
    feature_vector['Skewness'] = skewness

    m4 = np.nanmean(np.power(np_roi_array - mean_value, 4))
    kurtosis = m4 / (m2 + epsilon) ** 2
    feature_vector['Kurtosis'] = kurtosis

    variance_value = standard_deviation ** 2
    feature_vector['Variance'] = variance_value

    uniformity = np.nansum(p_i ** 2)
    feature_vector['Uniformity'] = uniformity

    return feature_vector


