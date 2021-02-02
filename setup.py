#!/usr/bin/env python

import os
from setuptools import setup

with open('agglom_attention_flowws/version.py') as version_file:
    exec(version_file.read())

module_names = [
    'WikiText2',
    'Text8',
    'GPTModel',
    'Run',
]

flowws_modules = [
    '{0} = agglom_attention_flowws.{0}:{0}'.format(name) for name in module_names]

setup(name='agglom_attention_flowws',
      author='Matthew Spellings',
      author_email='matthew.p.spellings@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Stage-based scientific workflows for agglomerative attention experiments',
      entry_points={
          'agglom_attention_flowws': flowws_modules,
      },
      extras_require={},
      install_requires=[
          'flowws',
          'keras-gtar',
          'keras-transformer',
          'keras-tqdm',
          'subword-nmt',
      ],
      license='MIT',
      packages=[
          'agglom_attention_flowws',
          # extracted bits from keras-transformer repository example files
          'agglom_attention_flowws.kt_examples',
      ],
      include_package_data=True,
      python_requires='>=3',
      version=__version__
      )
