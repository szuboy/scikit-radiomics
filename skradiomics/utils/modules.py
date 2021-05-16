# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: jeremy.zhang(szujeremy@gmail.com, Shenzhen University, China)

from .registry import Registry


PREPROCESSORS = Registry('pre_processor')
FILTER_CLASSES = Registry('filter_classes')
FEATURE_EXTRACTORS = Registry('feature_extractor')

