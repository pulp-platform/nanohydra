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

void hydra_classifier(Hydra* hydra) {
    #if defined (TARGET_GAP9) && defined (VECTORIZE)
    v4s featVec;
    v4s classf_weights;
    int8_t *pFeatVec = (int8_t*) hydra->featVec;
    #endif

    for(uint8_t c=0; c < hydra->N_classes; c++) {

        #if defined (VECTORIZE)
        for(int f=0; f < hydra->len_feat_vec; f+=4) {
        #else
        for(int f=0; f < hydra->len_feat_vec; f+=1) {
        #endif
            #if defined (TARGET_GAP9) && defined (VECTORIZE)
            featVec                  = __builtin_pulp_pack4(hydra->featVec[f], hydra->featVec[f+1], hydra->featVec[f+2], hydra->featVec[f+3]);
            classf_weights           = *((v4s*) &hydra->classf_weights[c][f]);
            hydra->classf_scores[c]  = __builtin_pulp_sdotsp4(featVec, classf_weights, hydra->classf_scores[c]);
            #else
            hydra->classf_scores[c] += hydra->featVec[f] * hydra->classf_weights[c][f];
            #endif
        }
        hydra->classf_scores[c] += hydra->classf_bias[c];
    }
}