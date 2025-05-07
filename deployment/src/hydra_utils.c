// Copyright (C) 2024-2025 ETH Zurich
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// SPDX-License-Identifier: Apache-2.0
// ==============================================================================
// 
// Author: Jose Fonseca, ETH Zurich (jcastro@student.ethz.ch)
// Author: Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)


#include "../include/hydra.h"

uint16_t pow2(int d, int val) {
    uint16_t out=1;
    for(int i=0; i < val; i++)
        out *=2;
    return out;
}

uint16_t padding_len(uint16_t lenW, uint16_t dilation) {
    return (lenW / 2) * (dilation+1) + 1;
}

uint16_t generate_dilation_val(uint16_t dil_idx) {
    return dil_idx == 0 ? dil_idx : pow2(2, dil_idx-1);
}