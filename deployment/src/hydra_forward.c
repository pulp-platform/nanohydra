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

#if defined (TARGET_GAP9) && defined (PARALLELIZE)
#else
void hydra_forward(Hydra *hydra) {

    uint8_t dil_idx;
    uint8_t diff_idx;
    uint8_t chan;
    uint16_t dil;

    // Iterate through the work chunks, for each dil/diff combination
    for (dil_idx = 0; dil_idx < hydra->N_dil; dil_idx++) {
        dil = generate_dilation_val(dil_idx);
        for (diff_idx = 0; diff_idx < hydra->N_diff; diff_idx++) {
            for (chan = 0; chan < hydra->N_chan; chan++) {
                if(diff_idx == 0) {
                    hydra_convolve(hydra->inX[chan],      
                                   hydra->inW, 
                                   &(hydra->featVec[chan*(hydra->K*hydra->H*hydra->N_feats) + diff_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats) + dil_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats*hydra->N_diff)]), 
                                   dil, 
                                   hydra,
                                   diff_idx);
                }
                else {
                    hydra_convolve(hydra->inX_diff[chan], 
                                   hydra->inW, 
                                   &(hydra->featVec[chan*(hydra->K*hydra->H*hydra->N_feats) + diff_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats) + dil_idx*(hydra->K*hydra->H*hydra->N_chan*hydra->N_feats*hydra->N_diff)]), 
                                   dil, 
                                   hydra,
                                   diff_idx);
                }
            }
        }
    }
}
#endif
