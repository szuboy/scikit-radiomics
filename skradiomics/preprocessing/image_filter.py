# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import numpy as np
import SimpleITK as sitk

from skradiomics.utils.modules import FILTER_CLASSES


@FILTER_CLASSES.register_module(name='original')
def original_image(image, mask, settings, **kwargs):
    yield image, mask


@FILTER_CLASSES.register_module(name='LoG')
def LoG_image(image, mask, settings, **kwargs):
    size = np.array(image.GetSize())
    spacing = np.array(image.GetSpacing())

    if np.min(size) < 4:
        raise ValueError('Image too small to apply LoG filter, size: %s' % size)

    sigma_values = settings.get('LoGSigma', [])

    for sigma in sigma_values:
        if np.all(size >= np.ceil(sigma / spacing) + 1):
            LoG_filter = sitk.LaplacianRecursiveGaussianImageFilter()
            LoG_filter.SetNormalizeAcrossScale(True)
            LoG_filter.SetSigma(sigma)
            image_name = ('log-sigma-%s-mm' % sigma).replace('.', '-')
            yield LoG_filter.Execute(image), mask


@FILTER_CLASSES.register_module(name='wavelet')
def wavelet_image(image, mask, settings, **kwargs):
    pass


@FILTER_CLASSES.register_module(name='square')
def square_image(image, mask, settings, **kwargs):
    np_image = sitk.GetArrayFromImage(image).astype(np.float64)
    coeff = 1 / np.sqrt(np.max(np.abs(np_image)))
    np_image = (coeff * np_image) ** 2
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.CopyInformation(image)
    yield sitk_image, mask


@FILTER_CLASSES.register_module(name='square_root')
def square_root_image(image, mask, settings, **kwargs):
    np_image = sitk.GetArrayFromImage(image).astype(np.float64)
    coeff = np.max(np.abs(np_image))
    np_image[np_image > 0] = np.sqrt(np_image[np_image > 0] * coeff)
    np_image[np_image < 0] = np.sqrt(-np_image[np_image < 0] * coeff)
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.CopyInformation(image)
    yield sitk_image, mask


@FILTER_CLASSES.register_module(name='logarithm')
def logarithm_image(image, mask, settings, **kwargs):
    np_image = sitk.GetArrayFromImage(image).astype(np.float64)
    max_value = np.max(np.abs(np_image))
    np_image[np_image > 0] = np.log(np_image[np_image > 0] + 1)
    np_image[np_image < 0] = - np.log(-np_image[np_image < 0] + 1)
    np_image = np_image * (max_value / np.max(np.abs(np_image)))
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.CopyInformation(image)
    yield sitk_image, mask


@FILTER_CLASSES.register_module(name='exponential')
def exponential_image(image, mask, settings, **kwargs):
    np_image = sitk.GetArrayFromImage(image).astype(np.float64)
    max_value = np.max(np.abs(np_image))
    coeff = np.log(max_value) / max_value
    np_image = np.exp(coeff * np_image)
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.CopyInformation(image)
    yield sitk_image, mask


@FILTER_CLASSES.register_module(name='gradient')
def gradient_image(image, mask, settings, **kwargs):
    gradient_filter = sitk.GradientMagnitudeImageFilter()
    gradient_filter.SetUseImageSpacing(settings.get('gradientUseSpacing', True))
    yield gradient_filter.Execute(image), mask


# @FILTER_CLASSES.register_module(name='gabor')
# def gabor_image(image, mask, settings, **kwargs):
#     pass

