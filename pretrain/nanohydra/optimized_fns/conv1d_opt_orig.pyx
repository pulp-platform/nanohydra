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
def conv1d_opt_orig(cnp.ndarray[DTYPE_t, ndim=2] x, cnp.ndarray[DTYPE_t, ndim=3] w, unsigned int dilation):

    cdef unsigned int num_examples = x.shape[0]
    cdef unsigned int xlen   = x.shape[1]
    cdef unsigned int wlen   = w.shape[2]
    cdef unsigned int wlenD2 = int((w.shape[2]-1)/2)
    cdef unsigned int H = w.shape[0]
    cdef unsigned int K = w.shape[1]
    cdef unsigned int xdil_len = int(xlen/(dilation+1))

    cdef unsigned int h,k,xi,wi

    cdef cnp.ndarray[DTYPE_t, ndim=4] Y     = np.zeros([num_examples, H, K, xdil_len], dtype=DTYPE)
    cdef cnp.ndarray[DTYPE_t, ndim=1] x_dil = np.zeros([xlen+wlen], dtype=DTYPE)

    for ex in range(num_examples):
        # Calculate the current dilation for the given example
        x_dil[wlenD2:wlenD2+xdil_len] = np.take(x[ex,:], [(1+dilation)*i for i in range(xdil_len)])

        # Work-sharing construct must start here, since np.take uses gil.
        with nogil:
            for h in prange(H, schedule='static', num_threads=24):
                for k in range(K):
                    for xi in range(0, xdil_len, 1):
                        for wi in range(wlen):
                            Y[ex, h, k, xi] += x_dil[xi+wi]*w[h,k,wi]

    return Y
