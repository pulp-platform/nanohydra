# NanoHydra: Energy-Efficient Time-Series Classification at the Edge

The current repository contains the implementation of NanoHydra, enabling server-side and on-device inference for time series classification. The method is tailored for TinyML devices and relies on lightweight binary random convolutional kernels to extract meaningful features from data streams. The features are further processed through different counting strategies (i.e., soft counting accumulates convolutional activations, whereas hard counting accumulates occurrences), followed by applying a fully connected layer to produce the classification result. The system was demonstrated on the ultra-low-power GAP9 microcontroller.

## Requirements

To install the packages required to pretrain NanoHydra and the associated libraries, run
```
pip install -r requirements.txt
python ./nanohydra/setup.py build_ext --build-lib=./nanohydra/optimized_fns
```


## Example

### Pretraining

All pretraining (i.e., server-side) files are located in `pretrain/`. All datasets evaluated in the current work are wrapped by the `sktime` package (i.e., a unified framework for multiple time series machine learning tasks). Change directory to run the following examples.

To train and evaluate NanoHydra on all UCR datasets, run
```
python run_ucr_benchmark.py
```
or the quantized version:
```
python run_ucr_benchmark_quant.py
```
whereas to focus on the ECG5000 dataset, you can train (i.e., `python run_ecg5000.py`) NanoHydra accordingly.

### Deployment

To deploy NanoHydra on GAP9 MCU, change directory to `deployment/`. We assume that GAP9 SDK was installed correctly, the installation instructions can be found [here](https://github.com/GreenWaves-Technologies/gap_sdk). The implementation of NanoHydra is available in `src/` and `include/`. `application/` countains the utilities required to build the GAP9 application. To build and run the application, run

```
./deploy.sh
```

## Contributors

* Cristian Cioflan, ETH Zurich [cioflanc@iis.ee.ethz.ch](cioflanc@iis.ee.ethz.ch)
* Jose Fonseca, ETH Zurich
* Xiaying Wang, ETH Zurich


## License

Unless explicitly stated otherwise, the code is released under Apache 2.0, see the LICENSE file in the root of this repository for details.

As an exception, the preprocessed samples and the pretrained model available in `./deployment/application/dist` are released under Creative Commons Attribution-NoDerivatives 4.0 International. Please see the LICENSE file in their respective directory.
