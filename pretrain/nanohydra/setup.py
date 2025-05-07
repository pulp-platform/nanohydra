# Copyright (C) 2024-2025 ETH Zurich
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
# 
# Author: Jose Fonseca, ETH Zurich (jcastro@student.ethz.ch)
# Author: Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)


from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import multiprocessing 

ext_modules=[Extension("conv1d_opt_orig", ["./nanohydra/optimized_fns/conv1d_opt_orig.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("conv1d_opt_x_f32_w_f32", ["./nanohydra/optimized_fns/conv1d_opt_x_f32_w_f32.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("conv1d_opt_x_f32_w_b1", ["./nanohydra/optimized_fns/conv1d_opt_x_f32_w_b1.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("conv1d_opt_x_int16_w_b1", ["./nanohydra/optimized_fns/conv1d_opt_x_int16_w_b1.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("hard_counting_opt", ["./nanohydra/optimized_fns/hard_counting_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("soft_counting_opt", ["./nanohydra/optimized_fns/soft_counting_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp']),
             Extension("combined_counting_opt", ["./nanohydra/optimized_fns/combined_counting_opt_fn.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'])
                ]


from Cython.Build import cythonize
setup(ext_modules=cythonize(ext_modules, 
                    compiler_directives={"language_level": "3"},
                    nthreads=24),
      include_dirs=[np.get_include()])