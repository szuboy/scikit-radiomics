# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


import json
import collections
import pykwalify.core

from skradiomics.utils import schema_func, schema_yaml
from skradiomics.preprocessing.image_regular import load_and_valid_image
from skradiomics.utils.modules import PREPROCESSORS, FEATURE_EXTRACTORS, FILTER_CLASSES


class RadiomicsFeatureExtractor(object):
    def __init__(self, *args, **kwargs):
        self.settings = {}
        self.enable_image_types = []
        self.enable_feature_classes = []

    def _update_params(self, params_file=None, params_dict=None):
        """
        validates and updates a parameter dictionary.
        """
        core = pykwalify.core.Core(source_file=params_file, source_data=params_dict,
                                   schema_files=[schema_yaml], extensions=[schema_func])
        params = core.validate()
        self.settings = params.get('setting', {})

        enable_image_types = self.settings.get('imageType', [])
        if len(enable_image_types) == 0:
            self.enable_image_types = ['original']
        else:
            self.enable_image_types = enable_image_types

        enable_feature_classes = self.settings.get('featureClass', [])
        if len(enable_feature_classes) == 0:
            self.enable_feature_classes = []
        else:
            self.enable_feature_classes = enable_feature_classes

    def load_yaml_params(self, params_file):
        """
        parse specified parameters file and use it to update settings, enabled feature, preprocess and image types.
        For more information on the structure of the parameter file, see: link
        If supplied file does not match the requirements (i.e. unrecognized names or invalid values for a setting),
        a error is raised.
        """
        self._update_params(params_file)

    def load_json_params(self, json_configuration):
        """
        parse JSON structured configuration and use it to update settings, enabled feature, preprocess and image types.
        For more information on the structure of the parameter file, see: link
        If supplied file does not match the requirements (i.e. unrecognized names or invalid values for a setting),
        a error is raised.
        """
        parameter_data = json.loads(json_configuration)
        self._update_params(params_dict=parameter_data)

    def execute(self, image_file_path, mask_file_path, label=None):
        """
        Compute radiomics signature for provide image and mask combination. It comprises of the following steps:
        1. Image and mask are loaded and check if they are match and valid. (mask is processed to only with 0/1)
        2. Preprocessing image and mask and normalized/resampled if necessary, return cropped image and mask.
        3. Calculate features by yaml setting file or json configuration, which returned as ``collections.OrderedDict``.
        """
        if label is not None:
            self.settings['label'] = label

        # 1. load and valid image if it is match mask, such size, spacing, label
        image, mask = load_and_valid_image(image_file_path, mask_file_path, self.settings)

        # 2. preprocessing image
        preprocess_methods = self.settings.get('preprocessMethod', [])
        for method in preprocess_methods:
            image, mask = PREPROCESSORS.get(method)(image, mask, self.settings)

        # 3. calculate features by settings
        feature_vector = collections.OrderedDict()

        # calculate shape feature, which only for original image
        if 'shape' in self.enable_feature_classes:
            for key, value in FEATURE_EXTRACTORS.get('shape')(image, mask, self.settings).items():
                feature_vector['original_shape_%s' % key] = value
            self.enable_feature_classes.remove('shape')

        # calculate features for all filter images and feature class
        for filter_name in self.settings.get('imageType', []):
            for (filter_image, filter_mask) in FILTER_CLASSES.get(filter_name)(image, mask, self.settings):
                for actor in self.enable_feature_classes:
                    for key, value in FEATURE_EXTRACTORS.get(actor)(filter_image, filter_mask, self.settings).items():
                        feature_vector['%s_%s_%s' % (filter_name, actor, key)] = value

        return feature_vector

