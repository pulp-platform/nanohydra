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
from cython.parallel import prange
cimport numpy as cnp
cimport cython

cnp.import_array()

DTYPE_INT16 = np.int16
ctypedef cnp.int16_t DTYPE_INT16_t

DTYPE_INT32 = np.int32
ctypedef cnp.int32_t DTYPE_INT32_t

DTYPE_UINT32 = np.uint32
ctypedef cnp.uint32_t DTYPE_UINT32_t

@cython.boundscheck(False)
@cython.wraparound(False)
def combined_counting_opt(cnp.ndarray[DTYPE_UINT32_t, ndim=3] args_max,
                          cnp.ndarray[DTYPE_UINT32_t, ndim=3] args_min, 
                          cnp.ndarray[DTYPE_INT32_t,  ndim=3] optims_max, 
                          cnp.ndarray[DTYPE_INT32_t,  ndim=3] optims_min, 
                          unsigned int kernels_per_group,
                          unsigned int frac_bit_shift):

    cdef unsigned int num_examples = optims_max.shape[0]
    cdef unsigned int num_groups   = optims_max.shape[1]
    cdef unsigned int num_samples  = optims_max.shape[2]
    cdef unsigned int idx_max, idx_min, ex, gr, s
    cdef int optim_max

    cdef cnp.ndarray[DTYPE_INT16_t, ndim=3] feats = np.zeros([num_examples, num_groups, 2*kernels_per_group], dtype=DTYPE_INT16)

    with nogil:
        for ex in prange(num_examples, schedule='static', num_threads=22):
            for gr in range(num_groups):
                for s in range(num_samples):
                    optim_max = optims_max[ex,gr,s]
                    idx_max   = args_max[ex,gr,s]
                    idx_min   = args_min[ex,gr,s]

                    # Soft-Max
                    feats[ex,gr,idx_max*2+0] += optim_max >> frac_bit_shift

                    # Hard-Min
                    feats[ex,gr,idx_min*2+1] += 1 
    
    return feats