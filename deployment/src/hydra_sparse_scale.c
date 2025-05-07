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

void hydra_sparse_scale(Hydra *hydra) {
    for(uint32_t f=0; f < hydra->len_feat_vec; f++) {

        // Under-clip to zero, skip normalization if feature is zero
        hydra->featVec[f] = (hydra->featVec[f] < 0 ? 0 : hydra->featVec[f]);
        
        if(hydra->featVec[f] > 0) {
            hydra->featVec[f] = hydra->featVec[f] - hydra->featMean[f];

            if(hydra->featVec[f] < 0) {
                // Most values are positive, but arithmetic shift of negative numbers 
                // is not equivalent to division by powers of two, since it does not round to zero.
                for(int s = 0; s < hydra->featStd[f]; s++) {
                    hydra->featVec[f] = (hydra->featVec[f]) / 2;
                }
            }
            else {
                hydra->featVec[f] = (hydra->featVec[f]) >> hydra->featStd[f];
            }
        }
    }
}