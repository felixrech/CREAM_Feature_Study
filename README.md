# Code of a feature study on the CREAM dataset

This repo contains a package of the features in a feature study for the CREAM dataset implemented in Python using numpy, pandas, SciPy and PyWavelets. Additionally, notebooks with the code used for the data analysis, plots and feature evaluation in the paper are available.

This work was done as part of a Bachelor's thesis in Computer Science at the Technical University of Munich.

## Installation

```bash
git clone https://github.com/FelixRech/CREAM_Feature_Study

# Install/update the necessary dependencies for using the features
pip3 install -U numpy pandas scipy PyWavelets

# For notebooks only (and the feature_boxplot function)
pip3 install -U jupyter matplotlib pyplot-themes
```

## Usage

Usage is dependent on the use case:

### Utilizing the features

If you are only interested in the features themself, utilize the features (sub-) module:

```python3
from features import vi, spectral, wavelet

vi.vi_trajectory(my_voltage_measurements, my_current_measurements)
```

Note that you will have to adjust the `POWER_FREQUENCY`, `SAMPLING_RATE`, and `PERIOD_LENGTH` parameters to match your dataset. This can be done by using the respective arguments available in each feature's function or by editing the constants in the top of each submodule. The default values for the CREAM dataset are a power frequency of 50Hz, a sampling rate of 6.4kHz and a resulting period length of 128.

A hosted version of the documentation is available [here](https://felixrech.github.io/CREAM_Feature_Study/CREAM_Feature_Study/features/).

### Interacting with the notebooks created for the thesis

Navigate to location where you cloned the feature study and start up a Jupyter session:

```bash
cd CREAM_Feature_Study/

jupyter notebook ./
```

A browser window should pop up and you can navigate to the notebooks folder to open the notebook you are interested in.
