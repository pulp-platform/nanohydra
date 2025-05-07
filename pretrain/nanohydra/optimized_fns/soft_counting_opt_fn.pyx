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

DTYPE_FLOAT32 = np.float32
ctypedef cnp.float32_t DTYPE_FLOAT32_t

DTYPE_UINT32 = np.uint32
ctypedef cnp.uint32_t DTYPE_UINT32_t

@cython.boundscheck(False)
@cython.wraparound(False)
def soft_counting_opt(cnp.ndarray[DTYPE_UINT32_t, ndim=3] args, cnp.ndarray[DTYPE_FLOAT32_t, ndim=3] optims, unsigned int kernels_per_group):
    cdef unsigned int num_examples = optims.shape[0]
    cdef unsigned int num_groups   = optims.shape[1]
    cdef unsigned int num_samples  = optims.shape[2]
    cdef unsigned int idx , ex, gr, s
    cdef float optim

    cdef cnp.ndarray[DTYPE_FLOAT32_t, ndim=3] feats = np.zeros([num_examples, num_groups, kernels_per_group], dtype=DTYPE_FLOAT32)

    with nogil:
        for ex in prange(num_examples, schedule='static', num_threads=22):
            for gr in range(num_groups):
                for s in range(num_samples):
                    optim = optims[ex,gr,s]
                    idx   = args[ex,gr,s]
                    feats[ex,gr,idx] += optim
    
    return feats