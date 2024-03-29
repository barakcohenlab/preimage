#!/usr/bin/env python
# -*- coding:utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension

import numpy as np
# from Cython.Distutils import build_ext
from Cython.Build import cythonize


ext_modules = [
    Extension("preimage.kernels._generic_string", ["preimage/kernels/_generic_string.pyx"],
              include_dirs=[np.get_include()], extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"]),
    Extension("preimage.inference.branch_and_bound", ["preimage/inference/branch_and_bound.pyx"],
              include_dirs=[np.get_include()]),
    Extension("preimage.inference.bound_calculator", ["preimage/inference/bound_calculator.pyx"],
              include_dirs=[np.get_include()]),
    Extension("preimage.inference.node", ["preimage/inference/node.pyx"],
              include_dirs=[np.get_include()]),
    Extension("preimage.inference.node_creator", ["preimage/inference/node_creator.pyx"],
              include_dirs=[np.get_include()]),
    Extension("preimage.features.gs_similarity_weights", ["preimage/features/gs_similarity_weights.pyx"],
              include_dirs=[np.get_include()])
]

setup(
    # cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules)
)