# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)


from .base import RadiomicsBase
from skradiomics.utils.modules import FEATURE_EXTRACTORS


# @FEATURE_EXTRACTORS.register_module(name='gldm')
# class RadiomicsGLDM(RadiomicsBase):
#     def __init__(self, **kwargs):
#         super(RadiomicsGLDM, self).__init__(**kwargs)
#         pass


@FEATURE_EXTRACTORS.register_module(name='gldm')
def radiomics_gldm(image, mask, settings, **kwargs):
    pass
