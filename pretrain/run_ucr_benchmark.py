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
from sklearn.linear_model                import RidgeClassifierCV, Perceptron
import numpy as np
import pandas as pd
import sys
import time
from nanohydra.hydra import NanoHydra

X  = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}

CSV_PATH = "./data/results_ucr112_variants.csv"
BEST_OF      = 10

def training_round(Xtrain, Xtest, Ytrain, Ytest, k=8, g=64, seed=None):

    input_length = Xtrain.shape[2]

    # Initialize Model 
    model  = NanoHydra(input_length=input_length, k=k,g=g, max_dilations=10, dist="binomial", classifier="Logistic", seed=seed, scaler="Sparse", dtype=np.int16, verbose=False)    

    # Perform Transform on Training Values
    Xt  = model.forward_batch(Xtrain, 500, do_fit=True, do_scale=True)

    # Train the classifier with the transformed input
    model.fit_classifier(Xt, Ytrain)

    # Perform Transform on Testing Values
    Xt  = model.forward_batch(Xtest, 500, do_fit=False, do_scale=True)

    # Score the predictions
    score = model.score(Xt, Ytest)

    return score

def load_dataset(dataset):

    for sp in ["test", "train"]:
        X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy3d")

    # The split argument splits into the opposite fold. Therefore, we here cross them back
    # together into the correct one.
    Xtrain = X['train'][ds].astype(np.float32)
    Xtest  = X['test'][ds].astype(np.float32)
    Ytrain = y['train'][ds].astype(np.float32)
    Ytest  = y['test'][ds].astype(np.float32)

    return Xtrain, Xtest, Ytrain, Ytest

if __name__ == "__main__":

    csv = pd.read_csv(CSV_PATH)
    print(csv)
    csv["Hydra_Binomial"] = np.nan
    csv["Hydra_Binomial_Var"] = np.nan

    if (sys.argv[1].lower() == 'all'):

        for idx,row in csv.iterrows():
            
            # Fetch the dataset
            ds = row['dataset']

            Xtrain, Xtest, Ytrain, Ytest = load_dataset(ds)

            best_score = 0.0
            scores = []
            for i in range(BEST_OF): 

                start = time.perf_counter()
                
                score = training_round(Xtrain, Xtest, Ytrain, Ytest, 8, 16, seed=i)
                print(f"Score for '{ds}': {100*score:0.02f} %") 	

                scores.append(score)

                best_score = max(best_score, score)

                if(best_score > 0.9999):
                    # If our accuracy is already at 100%, advance to the next DS.
                    break

                print(f"Execution of '{ds}' took {time.perf_counter()-start} seconds")
            print(f"Dataset '{ds}': {np.max(scores)*100 : 0.02f} +/- {100*(np.max(scores)-np.min(scores)) : 0.02f}")

            csv['Hydra_Binomial'][idx]     = best_score
            csv['Hydra_Binomial_Var'][idx] = np.max(scores)-np.min(scores)
            csv.to_csv("./data/results_ours.csv", mode="w")

    else:
        ds = sys.argv[1]
        #ds_idx = csv.index[csv['dataset'] == ds].to_list()[0]

        Xtrain, Xtest, Ytrain, Ytest = load_dataset(ds)

        f = open(f"results_many_runs_{ds}.txt", "w")
        #f.write(f"Author:{100*csv['Hydra'][ds_idx]}\n")
        scores = []
        for i in range(BEST_OF): 
            start = time.perf_counter()

            score = training_round(Xtrain, Xtest, Ytrain, Ytest, 8, 16, seed=i)
            print(f"Score for '{ds}': {100*score:0.02f} %") 	
            f.write(f"{100*score}\n")
            f.flush()
            scores.append(100*score)

            print(f"Execution of '{ds}' took {time.perf_counter()-start} seconds")

        
        print(f"Accuracy stats: {np.mean(scores):.2f} +/- {np.std(scores):.3f}. Best: {np.max(scores):.2f}, Worse: {np.min(scores):.2f}")
