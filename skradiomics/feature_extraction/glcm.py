# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import collections
import numpy as np

from .base import glcm_matrix
from skradiomics.utils.modules import FEATURE_EXTRACTORS


# from .base import RadiomicsBase
# @FEATURE_EXTRACTORS.register_module(name='glcm')
# class RadiomicsGLCM(RadiomicsBase):
#     def __init__(self, **kwargs):
#         super(RadiomicsGLCM, self).__init__(**kwargs)
#         pass


@FEATURE_EXTRACTORS.register_module(name='glcm')
def radiomics_glcm(image, mask, settings, **kwargs):
    feature_vector = collections.OrderedDict()

    epsilon = np.spacing(1)
    distances = settings.get('GLCMDistance', [1])
    weight_normal = settings.get('weightingNorm', None)

    for distance in distances:
        kwargs['distance'] = distance

        p_glcm, angles = glcm_matrix(image, mask, settings, **kwargs)
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

        p_glcm = np.sum(p_glcm * weights, 2)
        p_glcm = p_glcm / np.sum(p_glcm)

        Ng = p_glcm.shape[0]
        Ng_vector = np.arange(1, Ng + 1)

        i, j = np.meshgrid(Ng_vector, Ng_vector, indexing='ij', sparse=False)

        px = np.sum(p_glcm, axis=1)
        py = np.sum(p_glcm, axis=0)

        px_add_y = np.array([np.sum(p_glcm[i + j == k]) for k in range(2, 2 * Ng + 1)])
        px_sub_y = np.array([np.sum(p_glcm[np.abs(i - j) == k]) for k in range(0, Ng)])

        ux = np.sum(i * p_glcm)
        uy = np.sum(j * p_glcm)

        k_value_sum = np.arange(2, 2 * Ng + 1)
        k_value_diff = np.arange(0, Ng)

        Hxy = -1 * np.nansum(p_glcm * np.log2(p_glcm + epsilon))

        auto_correlation = np.nansum(p_glcm * (i * j))
        feature_vector['AutoCorrelation_Distance%s' % distance] = auto_correlation

        joint_average = ux
        feature_vector['JointAverage_Distance%s' % distance] = joint_average

        cluster_prominence = np.nansum(p_glcm * (i + j - ux - uy) ** 4)
        feature_vector['ClusterProminence_Distance%s' % distance] = cluster_prominence

        cluster_shade = np.nansum(p_glcm * (i + j - ux - uy) ** 3)
        feature_vector['ClusterShade_Distance%s' % distance] = cluster_shade

        cluster_tendency = np.nansum(p_glcm * (i + j - ux - uy) ** 2)
        feature_vector['ClusterTendency_Distance%s' % distance] = cluster_tendency

        contrast = np.nansum(p_glcm * np.square(i - j))
        feature_vector['Contrast_Distance%s' % distance] = contrast

        sx = np.nansum(p_glcm * (i - ux) ** 2)
        sy = np.nansum(p_glcm * (j - uy) ** 2)
        correlation = np.sum(p_glcm * (i - ux) * (j - uy)) / (sx + sy + epsilon)
        if sx * sy == 0:
            correlation = 1
        feature_vector['Correlation_Distance%s' % distance] = correlation

        difference_average = np.nansum(k_value_diff * px_sub_y)
        feature_vector['DifferenceAverage_Distance%s' % distance] = difference_average

        difference_entropy = -1 * np.nansum(px_sub_y * np.log2(px_sub_y + epsilon))
        feature_vector['DifferenceEntropy_Distance%s' % distance] = difference_entropy

        difference_variance = np.nansum(px_sub_y * (k_value_diff - difference_average) ** 2)
        feature_vector['DifferenceVariance_Distance%s' % distance] = np.nanmean(difference_variance)

        joint_energy = np.nanmean(np.sum(p_glcm ** 2))
        feature_vector['JointEnergy_Distance%s' % distance] = joint_energy

        joint_entropy = np.nanmean(Hxy)
        feature_vector['JointEntropy_Distance%s' % distance] = joint_entropy

        Hx = -1 * np.nansum(px * np.log2(px + epsilon))
        Hy = -1 * np.nansum(py * np.log2(py + epsilon))
        Hxy1 = -1 * np.sum(p_glcm * np.log2(px * py + epsilon))
        div = np.fmax(Hx, Hy)
        if div != 0:
            imc1 = (Hxy - Hxy1) / div
        else:
            imc1 = 0
        feature_vector['Imc1_Distance%s' % distance] = imc1

        idm = np.nansum(px_sub_y / (1 + k_value_diff ** 2))
        feature_vector['Idm_Distance%s' % distance] = idm

        idmn = np.nansum(px_sub_y / (1 + (np.square(k_value_diff) / np.square(Ng))))
        feature_vector['Idmn_Distance%s' % distance] = idmn

        inverse_difference = np.nansum(px_sub_y / (1 + k_value_diff))
        feature_vector['Id_Distance%s' % distance] = inverse_difference

        idn = np.nansum(px_sub_y / (1 + k_value_diff / Ng))
        feature_vector['Idn_Distance%s' % distance] = idn

        inverse_variance = np.nansum(px_sub_y[1:] / k_value_diff[1:] ** 2)
        feature_vector['InverseVariance_Distance%s' % distance] = inverse_variance

        max_prob = np.nanmax(p_glcm)
        feature_vector['MaximumProbability_Distance%s' % distance] = max_prob

        sum_average = np.nansum(k_value_sum * px_add_y)
        feature_vector['SumAverage_Distance%s' % distance] = sum_average

        sum_entropy = -1 * np.nansum(px_add_y * np.log2(px_add_y + epsilon))
        feature_vector['SumEntropy_Distance%s' % distance] = sum_entropy

        sum_square = np.nansum(p_glcm * ((i - ux) ** 2))
        feature_vector['SumSquares_Distance%s' % distance] = sum_square

    return feature_vector

