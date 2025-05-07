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

void hydra_reset(Hydra *hydra) {

    uint16_t i;

    // Reset the feature vector, parallelization point for OMP Parallel For
    for(i=0; i < hydra->len_feat_vec; i++) {
        hydra->featVec[i] = 0;
    }

    // Reset the classifier score accumulator
    for(i=0; i < hydra->N_classes; i++) {
        hydra->classf_scores[i] = 0;
    }
}