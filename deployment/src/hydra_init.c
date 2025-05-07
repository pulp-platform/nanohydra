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

#ifndef TARGET_GAP9
    #define MALLOC malloc
#else
    #define MALLOC pi_l2_malloc
#endif


Hydra* hydra_init(
    uint16_t   lenX,
    uint16_t   lenW,
    uint16_t   K,
    uint16_t   G,
    uint8_t    N_dil,
    uint8_t    N_diff,
    uint8_t    N_chan, 
    uint8_t    N_feats,
    uint8_t    N_classes,
    uint8_t    conv_frac_bit_shift) {

    // Declare pointer to hydra struct
    Hydra *hydra;
    
    // Initialize the Hydra struct
    hydra            = (Hydra*) MALLOC(sizeof(Hydra));
    hydra->lenX      = lenX;
    hydra->lenW      = lenW;
    hydra->lenXpad   = padding_len(hydra->lenW,generate_dilation_val(hydra->N_dil))+100; // ToDO: For high dil, we need extra padding.
    hydra->K         = K;
    hydra->G         = G;
    hydra->N_dil     = N_dil;
    hydra->N_feats   = N_feats;
    hydra->N_diff    = N_diff;
    hydra->N_chan    = N_chan;
    hydra->N_classes = N_classes;
    hydra->conv_frac_bit_shift =conv_frac_bit_shift;

    // Calculated attributes
    hydra->H       = hydra->G/2;
    hydra->len_feat_vec = hydra->H*hydra->K*hydra->N_dil*hydra->N_diff*hydra->N_feats*hydra->N_chan;

    // Allocate input vector
    hydra->inX      = (RCKINT**) MALLOC(sizeof(RCKINT*)*hydra->N_chan);
    hydra->inX_diff = (RCKINT**) MALLOC(sizeof(RCKINT*)*hydra->N_chan);

    for(int c=0; c < hydra->N_chan; c++) {
        hydra->inX[c]      = (RCKINT*) MALLOC(sizeof(RCKINT)*(hydra->lenX + 2*hydra->lenXpad+1));
        hydra->inX_diff[c] = (RCKINT*) MALLOC(sizeof(RCKINT)*(hydra->lenX + 2*hydra->lenXpad+1));

        // Initialize to zeros input vector
        for(int i=0; i <= hydra->lenX + 2*hydra->lenXpad; i++) {
            hydra->inX[c][i]      = (RCKINT) (0);
            hydra->inX_diff[c][i] = (RCKINT) (0);
        }
    }    

    // Allocate weight vector
    hydra->inW = (RCKINT*) MALLOC(sizeof(RCKINT)*hydra->H*hydra->K*(hydra->lenW));

    // Allocate feature vector
    hydra->featVec = (int16_t*) MALLOC(sizeof(int16_t) * hydra->len_feat_vec); 

    // Allocate scaler attribute memory
    hydra->featMean = (int16_t*)  MALLOC(sizeof(int16_t) * hydra->len_feat_vec);
    hydra->featStd  = (uint8_t *) MALLOC(sizeof(uint8_t) * hydra->len_feat_vec);

    // Allocate classifier attribute structures
    hydra->classf_scores  = (int32_t*)  MALLOC(sizeof(int32_t)  * hydra->N_classes);
    hydra->classf_bias    = (int8_t*)  MALLOC(sizeof(int8_t)  * hydra->N_classes);
    hydra->classf_weights = (int8_t**) MALLOC(sizeof(int8_t*) * hydra->N_classes);

    for(int c=0; c < hydra->N_classes; c++) {
        hydra->classf_weights[c] = (int8_t*) MALLOC(sizeof(int8_t) * hydra->len_feat_vec);
    }

    return hydra;
}
