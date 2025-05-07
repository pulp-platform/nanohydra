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
from mlutils.quantizer import LayerQuantizer

DATASETS        = ["ECG5000"]
SHOW_HISTOGRAMS = False
SHOW_CONFMATRIX = False
SHOW_EXAMPLES   = False
SHOW_TRANSFORM  = False
USE_CACHED      = False
SHOW_ALPHAS_RR  = False
DO_QUANTIZE     = True
DO_SHOW_MU_SIGMA_VALS = False

X  = {'test': {}, 'train': {}}
y  = {'test': {}, 'train': {}}


if __name__ == "__main__":

    num_ex = int(sys.argv[1])

    start = time.perf_counter()

    for ds in DATASETS:
        
        # Fetch the dataset
        for sp in ["test", "train"]:
            X[sp][ds],y[sp][ds]  = load_ucr_ds(ds, split=sp, return_type="numpy3d")

        Ns = min(X['test'][ds].shape[0], num_ex)

        # The split argument splits into the opposite fold. Therefore, we here cross them back
        # together into the correct one.
        Xtrain = X['train'][ds][:Ns,:].astype(np.float32)
        Xtest  = X['test'][ds][:Ns,:].astype(np.float32)
        Ytrain = y['train'][ds][:Ns]
        Ytest  = y['test'][ds][:Ns]
        print(np.unique(Ytrain))
        print(f"Training fold: {Xtrain.shape}")
        print(f"Testing  fold: {Xtest.shape}")

        input_length = Xtrain.shape[0]

        # Display Histograms
        if(SHOW_HISTOGRAMS):
            plt.figure(1)
            plt.hist(Ytrain)
            plt.title(f"Dataset '{ds}' Training Classes Histogram.")
            plt.figure(2)
            plt.hist(Ytest)
            plt.title(f"Dataset '{ds}' Testing Classes Histogram.")

            print(f"Sample   Size: {len(Xtrain[0,:])}")
            print(f"Training Size: {Ytrain.shape[0]}")
            print(f"Testing  Size: {Ytest.shape[0]}")
            print(f"Num   Classes: {len(np.unique(Ytest))}")

        if(DO_QUANTIZE):                
            lq_input = LayerQuantizer(Xtrain, 16)
            Xtrain = lq_input.quantize(Xtrain)
            Xtest  = lq_input.quantize(Xtest)
            print(f"Input Vector Quant.: {lq_input}")


        if(SHOW_EXAMPLES):
            plt.figure(3)
            plt.plot(Xtrain[498,0,:])
            #for c in np.unique(Ytest):
            #    idx = 0
            #    while(Ytrain[idx] != c):
            #        idx += 1
            #    plt.plot(Xtrain[idx], label=f"Class {c}")
            plt.legend()
            plt.show()

        # Initialize the kernel transformer, scaler and classifier
        model  = NanoHydra(input_length=input_length, num_channels=1, k=8, g=16, max_dilations=8, dist="binomial", classifier="Logistic", scaler="Sparse", seed=int(time.time()), dtype=np.int16, verbose=False)    

        # Transform and scale
        print(f"Transforming {Xtrain.shape[0]} training examples...")
        Xt = model.load_transform(ds, "./work", "train") 
        if(Xt is None or not USE_CACHED):
            Xt  = model.forward_batch(Xtrain, 100, do_fit=True, do_scale=True)

            print(f"Feature Vect Train: {np.min(Xt)} -- {np.max(Xt)}")

            # Quantize the feature vector
            if(DO_QUANTIZE):
                lq_featvec = LayerQuantizer(Xt, 16)
                print(f"Feature Vector Quant.: {lq_featvec}")
                Xt = lq_featvec.quantize(Xt)

            model.save_transform(Xt, ds, "./work", "train")
        else:
            print("Using cached transform...")


        if(SHOW_TRANSFORM):
            plt.figure(5)
            plt.imshow(Xt, vmin=np.min(Xt), vmax=np.max(Xt))
            plt.title(f"Transformed Training Set (Full, Not Shuffled)")

            # Display Training 10 Examples per class
            idxs = Ytrain.astype(np.float32).argsort()
            Xt_sorted = Xt[idxs]

            change_idxs,  = np.nonzero(np.diff(sorted(Ytrain.astype(np.float32))))
            print(change_idxs)

            plt.figure(9)
            ax = plt.subplot()
            ax.imshow(Xt_sorted)
            for i,y in enumerate(change_idxs):
                ax.text(Xt.shape[1], y, f"Class {i+1}")
                ax.axhline(y, color='r', linestyle='-')
            plt.title(f"Transformed Training Set (Ordered by classes)")

        # Fit the classifier
        model.fit_classifier(Xt, Ytrain)

        # Test the classifier
        print(f"Transforming Test Fold...")
        Xt = model.load_transform(ds, "./work", "test") 
        if(Xt is None) or not USE_CACHED:
            Xt  = model.forward_batch(Xtest, 100, do_scale=True)

            print(f"Feature Vect Test: {np.min(Xt)} -- {np.max(Xt)}")
            # Quantize the feature vector
            if(DO_QUANTIZE):
                lq_featvec = LayerQuantizer(Xt, 16)
                print(f"Feature Vector Quant.: {lq_featvec}")
                Xt = lq_featvec.quantize(Xt)
            model.save_transform(Xt, ds, "./work", "test")
        else:
            print("Using cached transform...")

        Ypred = model.predict_batch(Xt, 100)
        print(f"Ypred shape: {Ypred.shape}")
        model.quantize_classifier(16)
        score_man = model.score_manual(Ypred, Ytest, "subset")
        score = model.score(Xt, Ytest)
        print(f"Score (Aut) for '{ds}': {100*score:0.02f} %") 	
        print(f"Score (Man) for '{ds}': {100*score:0.02f} %") 	

        # Display Confusion Matrix
        if(SHOW_CONFMATRIX):
            cm = confusion_matrix(Ytest, Ypred, labels=model.cfg.get_classf().classes_)

            # Show accuracy instead of abs count of samples
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.cfg.get_classf().classes_)
            cmd.plot()

        # Display Alphas
        if(SHOW_ALPHAS_RR):
            #plt.figure(4)
            #plt.plot(np.logspace(-6,4,20), model.cfg.get_classf().attrs.cv_values_)
            print(f"Best alpha: {model.cfg.get_classf().alpha_}")
            
        if(SHOW_CONFMATRIX or SHOW_HISTOGRAMS or SHOW_EXAMPLES):
            plt.show()

    print(f"Execution of {Ns} examples took {time.perf_counter()-start} seconds")
