# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


import pywt
import six

from skradiomics.utils.modules import PREPROCESSORS, FILTER_CLASSES, FEATURE_EXTRACTORS


def checkPreprocessMethod(value, rule_obj, path):
    if value is None:
        return True
    for method in value:
        if method not in PREPROCESSORS:
            raise ValueError('Preprocess method %s is not recognized.' % method)
    return True


def checkWavelet(value, rule_obj, path):
    if not isinstance(value, six.string_types):
        raise TypeError('Wavelet not expected type (str)')
    wavelist = pywt.wavelist()
    if value not in wavelist:
        raise ValueError('Wavelet "%s" not available in pyWavelets %s' % (value, wavelist))
    return True


def checkInterpolator(value, rule_obj, path):
    if value is None:
        return True
    if isinstance(value, six.string_types):
        enum = {'sitkNearestNeighbor',
                'sitkLinear',
                'sitkBSpline',
                'sitkGaussian',
                'sitkLabelGaussian',
                'sitkHammingWindowedSinc',
                'sitkCosineWindowedSinc',
                'sitkWelchWindowedSinc',
                'sitkLanczosWindowedSinc',
                'sitkBlackmanWindowedSinc'}
        if value not in enum:
            raise ValueError('Interpolator value "%s" not valid, possible values: %s' % (value, enum))
    elif isinstance(value, int):
        if value < 1 or value > 10:
            raise ValueError('Intepolator value %i, must be in range of [1-10]' % value)
    else:
        raise TypeError('Interpolator not expected type (str or int)')
    return True


def checkWeighting(value, rule_obj, path):
    if value is None:
        return True
    elif isinstance(value, six.string_types):
        enum = ['euclidean', 'manhattan', 'infinity', 'no_weighting']
        if value not in enum:
            raise ValueError('WeightingNorm value "%s" not valid, possible values: %s' % (value, enum))
    else:
        raise TypeError('WeightingNorm not expected type (str or None)')
    return True


def checkFeatureClass(value, rule_obj, path):
    if value is None:
        raise TypeError('feature registry dictionary cannot be None value')
    for feature_name in value:
        if feature_name not in FEATURE_EXTRACTORS:
            raise ValueError('Feature function %s is not recognized.' % feature_name)
    return True


def checkImageType(value, rule_obj, path):
    if value is None:
        raise TypeError('imageType dictionary cannot be None value')
    for image_type in value:
        if image_type not in FILTER_CLASSES:
            raise ValueError('Image Type %s is not recognized.' % image_type)

    return True
