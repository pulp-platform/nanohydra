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


#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifdef TARGET_GAP9
#include "pmsis.h"
#include <bsp/bsp.h>
#else
#include <omp.h>
#endif

#include "hydra_defines.h"

#ifdef QUANT_8BIT
    #define RCKINT int8_t
#else
    #define RCKINT int16_t
#endif

typedef struct Hydra {
    // Memory allocations
    RCKINT   **inX;
    RCKINT   **inX_diff;
    RCKINT     *inW;
    int16_t    *featVec;
    int8_t   **classf_weights;
    int8_t    *classf_bias;
    int32_t    *classf_scores;

    // Attributes
    uint16_t lenX;  
    uint16_t lenW;
    uint16_t lenXpad;
    uint16_t H;     
    uint16_t K;
    uint16_t G;
    uint8_t  N_dil; 
    uint8_t  N_diff; 
    uint8_t  N_chan; 
    uint8_t  N_feats;
    uint16_t len_feat_vec;
    uint8_t  conv_frac_bit_shift;

    // Classifier Attributes
    uint8_t N_classes;

    // Scaler Attributes
    int16_t *featMean;
    uint8_t *featStd;

} Hydra;

#ifdef TARGET_GAP9
typedef struct {
    RCKINT   * inX;
    RCKINT   * inW;
    int16_t  * featVec;
    uint8_t    dil;
    Hydra*     hydra;
    uint8_t    diff_idx;
} TeamForkArgs_T;
#else
#endif


Hydra* hydra_init(
    uint16_t  lenX,
    uint16_t  lenW,
    uint16_t  H,     
    uint16_t  G,
    uint8_t   N_dil,
    uint8_t   N_diff,
    uint8_t   N_chan, 
    uint8_t   N_feats,
    uint8_t   N_classes,
    uint8_t   conv_frac_bit_shift);

void hydra_reset(Hydra *hydra);

#if defined (TARGET_GAP9) && defined (PARALLELIZE)
void hydra_convolve(void* args);
#else
void hydra_convolve(RCKINT   *inX, 
                    RCKINT   *inW, 
                    int16_t  *featVec, 
                    uint16_t   dil,
                    Hydra     *hydra,
                    uint8_t    diff_idx
                    );
#endif

void hydra_forward(Hydra *hydra);

void hydra_sparse_scale(Hydra *hydra); 

void hydra_classifier(Hydra* hydra);

uint16_t padding_len(uint16_t lenW, uint16_t dilation);

uint16_t generate_dilation_val(uint16_t dil_idx);