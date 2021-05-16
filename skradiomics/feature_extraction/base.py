# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import inspect
import numpy as np
import skimage.measure
import SimpleITK as sitk
from functools import reduce


class RadiomicsBase(object):
    def __init__(self, **kwargs):
        pass

    @classmethod
    def getFeatureNames(cls):
        attributes = inspect.getmembers(cls)
        features = [a[0][3:-12] for a in attributes if a[0].startswith('get') and a[0].endswith('FeatureValue')]
        return features


def bin_image(image, mask, settings, **kwargs):
    bin_width = settings.get('binWidth', 25)

    np_image = sitk.GetArrayFromImage(image)
    np_mask = sitk.GetArrayFromImage(mask)

    parameter_values = np_image[np_mask != 0]

    minimum = np.min(parameter_values)
    maximum = np.max(parameter_values)

    lower_bound = minimum - (minimum % bin_width)
    upper_bound = maximum + 2 * bin_width

    bin_edges = np.arange(lower_bound, upper_bound, bin_width)

    if len(bin_edges) == 1:
        bin_edges = [bin_edges[0] - .5, bin_edges[0] + .5]

    np_image[np_mask != 0] = np.digitize(np_image[np_mask != 0], bin_edges)

    return np_image, bin_edges


def glcm_matrix(image, mask, settings, **kwargs):
    offsets = np.array([[1, 1, 1],
                        [1, 1, 0],
                        [1, 1, -1],
                        [1, 0, 1],
                        [1, 0, 0],
                        [1, 0, -1],
                        [1, -1, 1],
                        [1, -1, 0],
                        [1, -1, -1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [0, 1, -1],
                        [0, 0, 1]])

    np_mask = sitk.GetArrayFromImage(mask)

    np_discretized_image, bin_edges = bin_image(image, mask, settings, **kwargs)
    Ng = int(np.max(np.unique(np_discretized_image[np_mask != 0])))
    np_discretized_image[np_mask == 0] = 0

    shape = np_discretized_image.shape
    distance = kwargs.get('distance', 1)

    matrix = np.zeros((Ng + 1, Ng + 1, 13))
    for i, offset in enumerate(offsets * distance):
        x = np.ravel(np_discretized_image[max(offset[0], 0):shape[0] + min(offset[0], 0),
                                          max(offset[1], 0):shape[1] + min(offset[1], 0),
                                          max(offset[2], 0):shape[2] + min(offset[2], 0)])
        y = np.ravel(np_discretized_image[max(-offset[0], 0):shape[0] + min(-offset[0], 0),
                                          max(-offset[1], 0):shape[1] + min(-offset[1], 0),
                                          max(-offset[2], 0):shape[2] + min(-offset[2], 0)])
        matrix[:, :, i] = np.histogram2d(x, y, bins=Ng+1)[0]
    return matrix[1:, 1:, :], offsets * distance


def gldm_matrix(image, mask, settings, **kwargs):
    pass


def glrlm_matrix(image, mask, settings, **kwargs):
    # From https://github.com/vnarayan13/Slicer-OpenCAD/tree/master/HeterogeneityCAD
    offsets = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, -1, 0],
                        [-1, 0, 1],
                        [0, 1, -1],
                        [1, 1, 1],
                        [-1, 1, -1],
                        [1, 1, -1],
                        [-1, 1, 1]])

    np_mask = sitk.GetArrayFromImage(mask)

    np_discretized_image, bin_edges = bin_image(image, mask, settings, **kwargs)
    Ng = int(np.max(np.unique(np_discretized_image[np_mask != 0])))
    np_discretized_image[np_mask == 0] = 0

    diagonal_list = list()

    # (1,0,0), #(-1,0,0),
    aDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(np_discretized_image, (1, 2, 0))])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, aDiags))

    # (0,1,0), #(0,-1,0),
    bDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(np_discretized_image, (0, 2, 1))])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, bDiags))

    # (0,0,1), #(0,0,-1),
    cDiags = reduce(lambda x, y: x + y, [a.tolist() for a in np.transpose(np_discretized_image, (0, 1, 2))])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, cDiags))

    # (1,1,0),#(-1,-1,0),
    lower = -np_discretized_image.shape[0] + 1
    upper = np_discretized_image.shape[1]

    dDiags = reduce(lambda x, y: x + y, [np_discretized_image.diagonal(a, 0, 1).tolist() for a in range(lower, upper)])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, dDiags))

    # (1,0,1), #(-1,0-1),
    lower = -np_discretized_image.shape[0] + 1
    upper = np_discretized_image.shape[2]

    eDiags = reduce(lambda x, y: x + y, [np_discretized_image.diagonal(a, 0, 2).tolist() for a in range(lower, upper)])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, eDiags))

    # (0,1,1), #(0,-1,-1),
    lower = -np_discretized_image.shape[1] + 1
    upper = np_discretized_image.shape[2]

    fDiags = reduce(lambda x, y: x + y, [np_discretized_image.diagonal(a, 1, 2).tolist() for a in range(lower, upper)])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, fDiags))

    # (1,-1,0), #(-1,1,0),
    lower = -np_discretized_image.shape[0] + 1
    upper = np_discretized_image.shape[1]

    gDiags = reduce(lambda x, y: x + y,
                    [np_discretized_image[:, ::-1, :].diagonal(a, 0, 1).tolist() for a in range(lower, upper)])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, gDiags))

    # (-1,0,1), #(1,0,-1),
    lower = -np_discretized_image.shape[0] + 1
    upper = np_discretized_image.shape[2]

    hDiags = reduce(lambda x, y: x + y,
                    [np_discretized_image[:, :, ::-1].diagonal(a, 0, 2).tolist() for a in range(lower, upper)])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, hDiags))

    # (0,1,-1), #(0,-1,1),
    lower = -np_discretized_image.shape[1] + 1
    upper = np_discretized_image.shape[2]

    iDiags = reduce(lambda x, y: x + y,
                    [np_discretized_image[:, :, ::-1].diagonal(a, 1, 2).tolist() for a in range(lower, upper)])
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, iDiags))

    # (1,1,1), #(-1,-1,-1)
    lower = -np_discretized_image.shape[0] + 1
    upper = np_discretized_image.shape[1]

    jDiags = [np.diagonal(h, x, 0, 1).tolist()
              for h in [np_discretized_image.diagonal(a, 0, 1) for a in range(lower, upper)]
              for x in range(-h.shape[0] + 1, h.shape[1])]
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, jDiags))

    # (-1,1,-1), #(1,-1,1),
    lower = -np_discretized_image.shape[0] + 1
    upper = np_discretized_image.shape[1]

    kDiags = [np.diagonal(h, x, 0, 1).tolist()
              for h in [np_discretized_image[:, ::-1, :].diagonal(a, 0, 1) for a in range(lower, upper)]
              for x in range(-h.shape[0] + 1, h.shape[1])]
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, kDiags))

    # (1,1,-1), #(-1,-1,1),
    lower = -np_discretized_image.shape[0] + 1
    upper = np_discretized_image.shape[1]

    lDiags = [np.diagonal(h, x, 0, 1).tolist()
              for h in [np_discretized_image[:, :, ::-1].diagonal(a, 0, 1) for a in range(lower, upper)]
              for x in range(-h.shape[0] + 1, h.shape[1])]
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, lDiags))

    # (-1,1,1), #(1,-1,-1),
    lower = -np_discretized_image.shape[0] + 1
    upper = np_discretized_image.shape[1]

    mDiags = [np.diagonal(h, x, 0, 1).tolist()
              for h in [np_discretized_image[:, ::-1, ::-1].diagonal(a, 0, 1) for a in range(lower, upper)]
              for x in range(-h.shape[0] + 1, h.shape[1])]
    diagonal_list.append(filter(lambda x: np.nonzero(x)[0].size > 1, mDiags))

    matrix = np.zeros((Ng, np.max(np_mask.shape), 13))
    for angle in range(0, len(diagonal_list)):
        for diagonal in diagonal_list[angle]:
            diagonal = np.array(diagonal)

            pos, = np.where(np.diff(diagonal) != 0)
            pos = np.concatenate(([0], pos + 1, [len(diagonal)]))

            run_length_encoding = list(zip([n for n in diagonal[pos[:-1]]], pos[1:] - pos[:-1]))

            run_length_encoding = [[int(x - 1), int(y - 1)] for x, y in run_length_encoding if x != 0]

            x_index, y_index = list(zip(*run_length_encoding))

            matrix[x_index, y_index, angle] += 1

    return matrix, offsets


def glszm_matrix(image, mask, settings, **kwargs):
    np_mask = sitk.GetArrayFromImage(mask)

    np_discretized_image, bin_edges = bin_image(image, mask, settings, **kwargs)
    Ns = int(np.sum(np_mask))
    Ng = int(np.max(np.unique(np_discretized_image[np_mask != 0])))
    np_discretized_image[np_mask == 0] = 0

    matrix = np.zeros((Ng, Ns))
    zero_mask = np.zeros_like(np_mask)
    for level in range(1, Ng + 1):
        zero_mask[np_discretized_image == level] = 1
        zero_mask[np_discretized_image != level] = 0
        label_mask, n_labels = skimage.measure.label(zero_mask, return_num=True)
        for label_value in range(1, n_labels + 1):
            matrix[level - 1, np.sum(label_mask == label_value) - 1] += 1
    return matrix


def ngtdm_matrix(image, mask, settings, **kwargs):
    np_mask = sitk.GetArrayFromImage(mask)

    np_discretized_image, bin_edges = bin_image(image, mask, settings, **kwargs)

    Ng = int(np.max(np.unique(np_discretized_image[np_mask != 0])))
    Ng_vector = np.arange(1, Ng + 1).reshape((-1, 1))
    np_discretized_image[np_mask == 0] = 0

    d = kwargs.get('distance', 1)
    np_discretized_image = np.pad(np_discretized_image, ((d, d), (d, d), (d, d)), mode='constant', constant_values=0)

    count_vector, ngtdm_vector = np.zeros((Ng, 1)), np.zeros((Ng, 1))
    delete_i = (2 * d + 1) ** 3 // 2

    x_where, y_where, z_where = np.where(np_discretized_image != 0)
    for x, y, z in zip(x_where, y_where, z_where):
        around = np.delete(np_discretized_image[x - d:x + d + 1, y - d:y + d + 1, z - d:z + d + 1].flatten(), delete_i)
        center = int(np_discretized_image[x, y, z])
        difference = np.nansum(np.abs(center - np.mean(around[around != 0]))) if np.sum(around) else 0
        count_vector[center - 1] += 1
        ngtdm_vector[center - 1] += difference

    return np.hstack((count_vector, ngtdm_vector, Ng_vector))


