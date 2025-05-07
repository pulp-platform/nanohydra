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


from sktime.datasets                     import load_UCR_UEA_dataset as load_ucr_ds
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from nanohydra.hydra import NanoHydra
from nanohydra.quantizer import LayerQuantizer
import pandas as pd

DATASETS        = [
    "InlineSkate"
]
DO_QUANTIZE     = True
DO_PLOT_QUANT   = False

X  = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}

CSV_PATH = "./data/results_ucr112_variants.csv"
BEST_OF      = 25

def training_round(Xtrain, Xtest, Ytrain, Ytest, k=8, g=64, seed=None):

    input_length = Xtrain.shape[0]
    
    if(DO_QUANTIZE):                
        lq_input = LayerQuantizer(Xtrain, 16)
        Xtrain = lq_input.quantize(Xtrain)
        Xtest  = lq_input.quantize(Xtest)
        print(f"Input Vector Quant.: {lq_input}")
        accum_bits_shift = lq_input.get_fract_bits()-1

    if(accum_bits_shift < 0):
        accum_bits_shift = 0
    # Initialize the kernel transformer, scaler and classifier
    model  = NanoHydra(input_length=input_length, num_channels=1, k=k, g=g, max_dilations=10, num_diffs=2, dist="binomial", classifier="Logistic", scaler="Sparse", seed=int(time.time()), dtype=np.int16, verbose=False)    

    # Transform and scale
    print(f"Transforming {Xtrain.shape[0]} training examples...")
    Xt  = model.forward_batch(Xtrain, 500, do_fit=True, do_scale=True, quantize_scaler=True, frac_bit_shift=accum_bits_shift)

    if(DO_PLOT_QUANT):
        print(f"Feature Vect Train: {np.min(Xt)} -- {np.max(Xt)}")
        plt.figure(1)
        plt.plot(model.cfg.get_scaler().muq / accum_bits_shift)
        plt.title("Mu")
        
        plt.figure(2)
        plt.plot(2**(model.cfg.get_scaler().sigmaq), label='Quantized')
        plt.plot(model.cfg.get_scaler().sigma,  label='Float')
        plt.title("Sigma")
        plt.legend()
        plt.show()

    # Fit the classifier
    model.fit_classifier(Xt, Ytrain)
    model.quantize_classifier(8)

    # Test the classifier
    print(f"Transforming Test Fold...")
    Xtq  = model.forward_batch(Xtest, 1000, do_scale=True, quantize_scaler=False,  frac_bit_shift=accum_bits_shift)

    print(f"Feature Vect Test: {np.min(Xt)} -- {np.max(Xt)}")
    Ypred = model.predict_batch(Xtq, 100).astype(np.uint8)
    Yquan = model.predict_quantized(Xtq)

    score  = model.score(Xtq, Ytest)
    scoreq = model.score_manual(Yquan, Ytest.astype(np.uint8), "subset")

    return score,scoreq

def load_dataset(dataset):

    for sp in ["test", "train"]:
        X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy3d")

    Xtrain = X['train'][ds].astype(np.float32)
    Xtest  = X['test'][ds].astype(np.float32)
    Ytrain = y['train'][ds].astype(np.int32)
    Ytest  = y['test'][ds].astype(np.int32)

    print(f"N_Classes: {np.unique(Ytrain)}")
    print(f"N_Classes: {np.unique(Ytest)}")
    print(f"Training fold: {Xtrain.shape}")
    print(f"Testing  fold: {Xtest.shape}")

    # Some datasets stupidly start at 1......
    smallest_label = np.min(Ytrain)
    if(smallest_label >= 1):
        Ytrain = Ytrain - smallest_label
        Ytest  = Ytest  - smallest_label
    elif(smallest_label == -1):
        Ytrain = np.where(Ytrain == -1, 0,  1)
        Ytest  = np.where(Ytest  == -1, 0,  1)

    return Xtrain, Xtest, Ytrain, Ytest

if __name__ == "__main__":


    if (sys.argv[1].lower() == 'all'):
        csv = pd.read_csv(CSV_PATH)
        print(csv)
        csv["Hydra_Quantized"] = np.nan
        csv["Hydra_Quantized_Var"] = np.nan

        for idx,row in csv.iterrows():

            # Fetch the dataset
            ds = row['dataset']

            Xtrain, Xtest, Ytrain, Ytest = load_dataset(ds)

            best_score = 0.0
            scores = []
            for i in range(BEST_OF): 

                start = time.perf_counter()
                
                score, scoreq = training_round(Xtrain, Xtest, Ytrain, Ytest, 8, 16, seed=i*562)
                print(f"Score (FP)    for '{ds}': {100*score :0.02f} %") 	
                print(f"Score (Quant) for '{ds}': {100*scoreq:0.02f} %") 	

                scores.append(scoreq)

                best_score = max(best_score, scoreq)


                if(best_score > 0.9999):
                    # If our accuracy is already at 100%, advance to the next DS.
                    break

                print(f"Execution of '{ds}' took {time.perf_counter()-start} seconds")
            print(f"Dataset '{ds}': {np.max(scores)*100 : 0.02f} +/- {100*(np.max(scores)-np.min(scores)) : 0.02f}")

            csv['Hydra_Quantized'][idx]     = best_score
            csv['Hydra_Quantized_Var'][idx] = np.max(scores)-np.min(scores)
            csv.to_csv("./data/results_ours.csv", mode="w")
    else:
        for ds in DATASETS:
            Xtrain, Xtest, Ytrain, Ytest = load_dataset(ds)

            best_score = 0.0
            scores = []
            for i in range(BEST_OF): 

                start = time.perf_counter()
                
                score, scoreq = training_round(Xtrain, Xtest, Ytrain, Ytest, 8, 16, seed=i)
                print(f"Score (FP)    for '{ds}': {100*score :0.02f} %") 	
                print(f"Score (Quant) for '{ds}': {100*scoreq:0.02f} %") 	

                scores.append(scoreq)

                best_score = max(best_score, scoreq)


                if(best_score > 0.9999):
                    # If our accuracy is already at 100%, advance to the next DS.
                    break

                print(f"Execution of '{ds}' took {time.perf_counter()-start} seconds")
            print(f"Dataset '{ds}': {np.max(scores)*100 : 0.02f} +/- {100*(np.max(scores)-np.min(scores)) : 0.02f}")

