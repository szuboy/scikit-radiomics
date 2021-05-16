# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

import numpy as np
import collections
import skimage.measure
import SimpleITK as sitk
from scipy.spatial.distance import pdist


from skradiomics.utils.modules import FEATURE_EXTRACTORS

# from .base import RadiomicsBase
# @FEATURE_EXTRACTORS.register_module(name='shape')
# class RadiomicsShape(RadiomicsBase):
#     def __init__(self, **kwargs):
#         super(RadiomicsShape, self).__init__(**kwargs)
#         pass


@FEATURE_EXTRACTORS.register_module(name='shape')
def radiomics_shape(image, mask, settings, **kwargs):
    feature_vector = collections.OrderedDict()

    spacing = image.GetSpacing()

    np_mask = sitk.GetArrayFromImage(mask)

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(np_mask, 0.5, spacing=spacing)

    surface_area = skimage.measure.mesh_surface_area(verts, faces)
    feature_vector['SurfaceArea'] = surface_area

    voxel_volume = np.sum(np_mask) * np.prod(spacing)
    feature_vector['VoxelVolume'] = voxel_volume

    surface_volume_ratio = surface_area / voxel_volume
    feature_vector['SurfaceVolumeRatio'] = surface_volume_ratio

    sphericity = (36 * np.pi * voxel_volume ** 2) ** (1.0 / 3.0) / surface_area
    feature_vector['Sphericity'] = sphericity

    compactness1 = voxel_volume / (surface_area ** (3.0 / 2.0) * np.sqrt(np.pi))
    feature_vector['Compactness1'] = compactness1

    compactness2 = (36 * np.pi) * (voxel_volume ** 2.0) / (surface_area ** 3.0)
    feature_vector['Compactness2'] = compactness2

    spherical_disproportion = surface_area / (36 * np.pi * voxel_volume ** 2) ** (1.0 / 3.0)
    feature_vector['SphericalDisproportion'] = spherical_disproportion

    max_diameter_3d = np.max(pdist(verts))
    feature_vector['MaxDiameter3D'] = max_diameter_3d

    return feature_vector









































