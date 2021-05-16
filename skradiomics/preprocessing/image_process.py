# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import six
import numpy as np
import SimpleITK as sitk

from skradiomics.utils.modules import PREPROCESSORS


@PREPROCESSORS.register_module(name='adjust')
def adjust_image(image, mask, settings, **kwargs):
    width = settings.get('windowWidth', None)
    level = settings.get('windowLevel', None)

    if width is None or level is None:
        raise ValueError('Adjust process method needs windowWidth and windowLevel, but got %s, %s' % (width, level))

    np_image = sitk.GetArrayFromImage(image)
    lower, upper = level - width / 2, level + width / 2
    np_image = np.clip(np_image, lower, upper)
    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.CopyInformation(image)
    image = sitk_image

    return image, mask


@PREPROCESSORS.register_module(name='resample')
def resample_image(image, mask, settings, **kwargs):
    resampled_pixel_spacing = settings.get('resampledPixelSpacing', None)
    interpolator = settings.get('interpolator', sitk.sitkBSpline)

    if resampled_pixel_spacing is None:
        raise ValueError('Resample process method needs resampledPixelSpacing, but got %s' % resampled_pixel_spacing)

    if image is None or mask is None:
        raise ValueError('Requires both image and mask to resample')

    mask_spacing = np.array(mask.GetSpacing())
    image_spacing = np.array(image.GetSpacing())

    Nd_resampled = len(resampled_pixel_spacing)
    Nd_mask = len(mask_spacing)
    assert Nd_resampled == Nd_mask, 'Wrong dimensionality (%i-D) of resampledPixelSpacing!, %i-D required' % (Nd_resampled, Nd_mask)

    # If spacing for a direction is set to 0, use the original spacing (enables "only in-slice" resampling)
    resampled_pixel_spacing = np.array(resampled_pixel_spacing)
    resampled_pixel_spacing = np.where(resampled_pixel_spacing == 0, mask_spacing, resampled_pixel_spacing)

    # calc resample new size
    image_size = image.GetSize()
    rasampled_new_size = [int(size * rate) for size, rate in zip(image_size, image_spacing / resampled_pixel_spacing)]

    image_pixel_type = image.GetPixelID()
    mask_pixel_type = mask.GetPixelID()

    direction = np.array(mask.GetDirection())

    try:
        if isinstance(interpolator, six.string_types):
            interpolator = getattr(sitk, interpolator)
    except Exception:
        interpolator = sitk.sitkBSpline

    rif = sitk.ResampleImageFilter()

    rif.SetOutputSpacing(resampled_pixel_spacing)
    rif.SetOutputDirection(direction)
    rif.SetSize(rasampled_new_size)
    rif.SetOutputPixelType(image_pixel_type)
    rif.SetInterpolator(interpolator)
    resampled_image = rif.Execute(image)

    rif.SetOutputPixelType(mask_pixel_type)
    rif.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = rif.Execute(mask)

    return resampled_image, resampled_mask


@PREPROCESSORS.register_module(name='normalize')
def normalize_image(image, mask, settings, **kwargs):
    """
     Normalize an image by setting its mean to zero and variance to one.
     It is based on all gray values in the image, not just those inside the segmentation.
    """
    scale = settings.get('normalizeScale', 1)

    image = sitk.Normalize(image)

    image = image * scale

    return image, mask


@PREPROCESSORS.register_module(name='remove')
def remove_image(image, mask, settings, **kwargs):
    """
    Remove outlier of segmentation by IQR score, which define by `removeOutliers`
    """
    outliers = settings.get('removeOutliers', 5)

    if outliers is None:
        raise ValueError('Remove outliers method needs `removeOutliers`, but got %s' % outliers)

    np_image = sitk.GetArrayFromImage(image)
    np_mask = sitk.GetArrayFromImage(mask)

    lower = np.nanpercentile(np_image[np_mask != 0], outliers)
    upper = np.nanpercentile(np_image[np_mask != 0], 100 - outliers)

    np_image[np_mask != 0] = np.clip(np_image[np_mask != 0], lower, upper)

    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.CopyInformation(image)

    return sitk_image, mask


@PREPROCESSORS.register_module(name='crop')
def crop_image(image, mask, settings, **kwargs):
    if image is None or mask is None:
        raise ValueError('Requires both image and mask to crop')

    label_filter = sitk.LabelStatisticsImageFilter()
    label_filter.Execute(image, mask)

    label = 1
    if label not in label_filter.GetLabels():
        raise ValueError('Label (%g) not present in mask' % label)

    # (L_X, U_Y, L_Y, U_Y, L_Z, U_Z)
    bbox = np.array(label_filter.GetBoundingBox(label=label))

    pad_distance = 5
    size = np.array(mask.GetSize())

    lower_min_bounds = bbox[0::2] - pad_distance
    upper_max_bounds = size - bbox[1::2] - pad_distance - 1

    # ensure cropped area is not outside original image bounds
    lower_min_bounds = np.maximum(lower_min_bounds, 0)
    upper_max_bounds = np.maximum(upper_max_bounds, 0)

    # crop image
    crop_filter = sitk.CropImageFilter()
    try:
        crop_filter.SetLowerBoundaryCropSize(lower_min_bounds)
        crop_filter.SetUpperBoundaryCropSize(upper_max_bounds)
    except TypeError:
        crop_filter.SetLowerBoundaryCropSize(lower_min_bounds.tolist())
        crop_filter.SetUpperBoundaryCropSize(upper_max_bounds.tolist())
    cropped_image = crop_filter.Execute(image)
    cropped_mask = crop_filter.Execute(mask)

    return cropped_image, cropped_mask




