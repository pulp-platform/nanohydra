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



import numpy as np
import multiprocessing
from cython.parallel import prange
cimport numpy as cnp
cimport cython

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def conv1d_opt_x_f32_w_f32(cnp.ndarray[DTYPE_t, ndim=2] x, cnp.ndarray[DTYPE_t, ndim=3] w, unsigned int dilation):

    
    
    
    cdef unsigned int num_examples = x.shape[0]
    cdef unsigned int xlen   = x.shape[1]
    cdef unsigned int wlen   = w.shape[2]
    cdef unsigned int H = w.shape[0]
    cdef unsigned int K = w.shape[1]
    cdef unsigned int xpad_len = (9//2)*(dilation+1)+1
    cdef unsigned int h,k,xi,wi,ex

    cdef cnp.ndarray[DTYPE_t, ndim=4] Y     = np.zeros([num_examples, H, K, xlen], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=2] x_dil = np.zeros([num_examples, xlen+xpad_len*2], dtype=DTYPE)

    x_dil[:,xpad_len:xlen+xpad_len] = x[:,:]

    # Work-sharing construct must start here, since np.take uses gil.
    with nogil:
        for ex in prange(num_examples, schedule='static', num_threads=24):
            for h in range(H):
                for k in range(K):
                    for xi in range(xlen):
                        for wi in range(wlen):
                            Y[ex, h, k, xi] += x_dil[ex,xi+xpad_len+(wi-4)*(dilation+1)]*w[h,k,wi]

    return Y
