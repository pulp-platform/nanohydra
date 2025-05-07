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


from nanohydra.optimized_fns.conv1d_opt_x_f32_w_f32         import conv1d_opt_x_f32_w_f32
from nanohydra.optimized_fns.conv1d_opt_x_f32_w_b1          import conv1d_opt_x_f32_w_b1
from nanohydra.optimized_fns.conv1d_opt_x_int16_w_b1        import conv1d_opt_x_int16_w_b1
import numpy as np
import time

# Input vector params
NUM_EXAMPLES  = 300
INPUT_VEC_LEN = 1600
DIL = 2

# Weight matrix params
DIVISOR    = 2
G          = 64
K          = 8
KERNEL_LEN = 9


if __name__ == '__main__':

    # Constants
    FUNCS = ['orig', 'x_f32_w_f32', 'x_f32_w_b1', 'x_int16_w_b1']

    # Test vars
    times  = {k:0 for k in FUNCS}
    errors = {k:0 for k in FUNCS}

    # Initialize RNG
    rng = np.random.default_rng(seed=42)

    # Test data
    X = rng.integers(low=-2**10, high=2**10, size=(NUM_EXAMPLES, INPUT_VEC_LEN)).astype(np.int16)
    W = rng.choice([-1, 1], size=(G // DIVISOR, K, KERNEL_LEN), p=[0.5, 0.5]).astype(np.int16)
    Y = {k:None for k in FUNCS}

    # Transform data
    start = time.perf_counter()
    Y['orig']     = conv1d_opt_x_f32_w_f32(X.astype(np.float32), W.astype(np.float32), DIL)
    times['orig'] = time.perf_counter()-start

    start = time.perf_counter()
    Y['x_f32_w_b1']     = conv1d_opt_x_f32_w_b1(X.astype(np.float32), W, DIL)
    times['x_f32_w_b1'] = time.perf_counter()-start
    errors['x_f32_w_b1'] = np.sum(np.abs(Y['x_f32_w_b1']-Y['orig']))

    start = time.perf_counter()
    Y['x_int16_w_b1']     = conv1d_opt_x_int16_w_b1(X, W, DIL)
    times['x_int16_w_b1'] = time.perf_counter()-start
    errors['x_int16_w_b1'] = np.sum(np.abs(Y['x_int16_w_b1']-Y['orig']))

    print(np.sum(np.abs(Y['orig'][0][0][0]-Y['x_int16_w_b1'][0][0][0])))

    # Print Results
    for k,v in errors.items():
        print(f"'{k}': {v}, executed in {times[k]:.3f} seconds")
