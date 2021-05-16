# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import os
import six
import numpy as np
import SimpleITK as sitk


def load_and_valid_image(image_file_path, mask_file_path, settings, **kwargs):
    if isinstance(image_file_path, six.string_types) and os.path.isfile(image_file_path):
        image = sitk.ReadImage(image_file_path)
    elif isinstance(image_file_path, sitk.SimpleITK.Image):
        image = image_file_path
    else:
        raise ValueError('Error reading image file path or SimpleITK object')

    if isinstance(mask_file_path, six.string_types) and os.path.isfile(mask_file_path):
        mask = sitk.ReadImage(mask_file_path)
    elif isinstance(mask_file_path, sitk.SimpleITK.Image):
        mask = mask_file_path
    else:
        raise ValueError('Error reading image file path or SimpleITK object')

    image_size = np.array(image.GetSize())
    mask_size = np.array(mask.GetSize())

    if not np.array_equal(image_size, mask_size):
        raise ValueError('Error image shape %s do not match mask shape %s' % (image_size, mask_size))

    image_spacing = np.array(image.GetSpacing())
    mask_spacing = np.array(mask.GetSpacing())

    if not np.allclose(image_spacing, mask_spacing, rtol=1e-3, atol=1e-3):
        raise ValueError('Error image spacing %s do not match mask spacing %s' % (image_spacing, mask_spacing))

    label = settings.get('label', 1)
    if label == 0:
        raise ValueError('Label should not be zero, zero means background, meaningless')

    mask = sitk.Cast(mask, sitk.sitkUInt32)
    labels = np.unique(sitk.GetArrayFromImage(mask))
    if len(labels) == 1:
        raise ValueError('No labels found in this mask (i.e. nothing is segmented)!')
    if label != -1 and label not in labels:
        raise ValueError('Label (%g) not present in mask. Choose from %s' % (label, labels[labels != 0]))

    min_roi_size = settings.get('minimumROISize', None)

    np_mask = sitk.GetArrayFromImage(mask)
    if label == -1:
        np_mask[np_mask != 0] = 1
    else:
        np_mask[np_mask != label] = 0
        np_mask[np_mask == label] = 1

    if min_roi_size is not None:
        if np.sum(np_mask) <= min_roi_size:
            raise ValueError('Size of the ROI is too small (minimum size: %g)' % min_roi_size)

    relabel_mask = sitk.GetImageFromArray(np_mask)
    relabel_mask.CopyInformation(mask)

    return image, relabel_mask

