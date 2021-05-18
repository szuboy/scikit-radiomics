# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import setuptools
from setuptools import setup

with open('requirements.txt', 'r') as fp:
    requirements = list(filter(bool, (line.strip() for line in fp)))

with open('README.md', 'rb') as fp:
    long_description = fp.read().decode('utf-8')

package_data = ['%s/%s' % ('utils', filename) for filename in os.listdir('skradiomics/utils') if not filename.endswith('.py')]

setup(
    name='scikit-radiomics',

    url='https://github.com/szuboy/scikit-radiomics#readme',
    project_urls={
        'Documentation': 'https://scikit-radiomics.readthedocs.io/en/latest/index.html',
        'Github': 'https://github.com/szuboy/scikit-radiomics'
    },

    author='szuboy',
    author_email='scikit-radiomics@googlegroups.com',

    version='1.0.0',

    packages=setuptools.find_packages(),

    package_data={'skradiomics': package_data},

    description='Radiomics for medical image',
    long_description=long_description,
    long_description_content_type="text/markdown",

    license='Apache License',

    keywords='radiomics python workflow feature-selection metric preprocessing',

    install_requires=requirements,

    python_requires='>=3.0'
)