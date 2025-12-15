# TimeSeriesSRC

A comprehensive Python package for time series analysis and prediction modeling, providing a complete suite of tools for analyzing, modeling, and forecasting time series data.

## Overview

TimeSeriesSRC is a professional-grade time series analysis toolbox translated from MATLAB to Python, offering extensive functionality for both univariate and multivariate time series analysis. The package implements classical time series models including ARX, ARMAX, ARMA, and Box-Jenkins Transfer Function (BJTF) models with advanced estimation algorithms.

## Key Features

### Model Types
- **ARX** (AutoRegressive with eXogenous inputs): Linear models with external inputs
- **ARMAX** (ARX with Moving Average): ARX models enhanced with moving average components
- **ARMA** (AutoRegressive Moving Average): Univariate time series models
- **BJTF** (Box-Jenkins Transfer Function): Most general form supporting multiple inputs and seasonal components
- **Regression**: Linear regression models with time delays

### Analysis Capabilities
- **GPAC Analysis** (Generalized Partial Autocorrelation): Model order selection and identification
- **Autocorrelation & Partial Autocorrelation**: Complete correlation analysis tools
- **Impulse Response Estimation**: System identification and characterization
- **Differencing Operations**: Seasonal and non-seasonal differencing
- **Chi-Square Testing**: Statistical validation of model residuals

### Estimation Methods
- **Levenberg-Marquardt Algorithm**: Non-linear optimization for parameter estimation
- **Multiple Performance Metrics**: AIC, BIC, MSE for model selection
- **Model Simulation**: One-step-ahead predictions and multi-step forecasts

## Installation

```bash
pip install TimeSeriesSRC
```

### Requirements
- Python >= 3.7
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- SciPy >= 1.5.0

## Quick Start

### Example 1: ARX Model

```python
import numpy as np
from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.estimate import estimate

# Create ARX model with 2 autoregressive terms, 1 input with 3 terms, delay of 1
model = pmodel('arx', na=2, nb=[3], delay=[1])

# Set training data
y = np.array([...])  # Output data
u = np.array([[...]])  # Input data (1 x n_samples)
model.set_data(y, u)

# Estimate parameters
model = estimate(model, y, u)

# Make predictions
y_pred = model.predict(y, u)
```

### Example 2: GPAC Analysis for Model Order Selection

```python
from TimeSeriesSRC.TimeSeries.TSAnalysis import TimesSeries
import numpy as np

# Create TimeSeries object
ts = TimesSeries()

# Perform univariate analysis
y = np.array([...])  # Your time series data
yacf, ypacf, ygpac = ts.uniAnal(y, na=20, nump=10, nrg=5, ncg=5)

# Plot GPAC table
ts.plotgpac(ygpac, 'GPAC Analysis')
```

### Example 3: ARMA Model

```python
from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.estimate import estimate

# Create ARMA model
model = pmodel('arma', nc=[3], nd=[2], diff=[0])

# Estimate and predict
model.set_data(y)
model = estimate(model, y)
y_pred = model.predict(y)
```

## Package Structure

```
TimeSeriesSRC/
├── basefunctions/          # Core analysis functions
│   ├── xcorr.py           # Cross-correlation and autocorrelation
│   ├── parcor.py          # Partial autocorrelation function
│   ├── gpac.py            # Generalized partial autocorrelation
│   ├── plotgpac.py        # GPAC visualization
│   ├── impest.py          # Impulse response estimation
│   ├── sdiff.py           # Differencing operations
│   ├── chisqrdf.py        # Chi-square cumulative distribution
│   ├── uniAnal.py         # Univariate analysis
│   ├── multiAnal.py       # Multivariate analysis
│   └── ...                # Additional utility functions
│
├── Model/                  # Prediction models
│   ├── model.py           # Core model class (pmodel)
│   ├── estimate.py        # Model parameter estimation
│   ├── estimlm.py         # Levenberg-Marquardt estimation
│   ├── jacobian.py        # Jacobian computation
│   ├── pmodaic.py         # Akaike Information Criterion
│   ├── pmodbic.py         # Bayesian Information Criterion
│   ├── pmodmse.py         # Mean Squared Error
│   ├── pmodsim.py         # Model simulation
│   └── selpmod.py         # Model selection utilities
│
├── TimeSeries/             # High-level interface
│   ├── TSAnalysis.py      # TimesSeries class
│   └── QAscript.py        # Quality assurance utilities
│
├── Examples/               # Usage examples
│   ├── NoteBooks/         # Jupyter notebook tutorials
│   │   ├── 00_All_Models_Furnace_Data.ipynb
│   │   ├── 01_ARX_Model_Complete.ipynb
│   │   ├── 02_ARMAX_Model_Complete.ipynb
│   │   ├── 03_ARMA_Model_Complete.ipynb
│   │   └── 04_BJTF_Model_Complete.ipynb
│   │
│   └── PyFiles/           # Python script examples
│       └── ...
│
└── test_Scripts/          # Unit tests
    └── ...
```

## Documentation

### Base Functions

#### Analysis Functions
- `gpac(acf, nrows, ncols)`: Compute GPAC table from autocorrelation
- `impest(u, y, k)`: Estimate impulse response between input and output
- `parcor(acf, nump)`: Compute partial autocorrelation function
- `partoacf(phi, theta, lagmax, var_a)`: Convert parameters to ACF
- `uniAnal(y, ...)`: Comprehensive univariate time series analysis
- `multiAnal(u, y, ...)`: Multivariate input-output analysis

#### Pre/Post Processing
- `sdiff(y, d, p)`: Apply differencing (seasonal and non-seasonal)

#### Utility Functions
- `xcorr(a, b, maxlags, flag)`: Cross-correlation and autocorrelation
- `chisqrdf(q, n)`: Chi-square cumulative density function
- `makerow(y)`: Matrix transformation utility

### Model Class (pmodel)

#### Initialization
```python
model = pmodel(
    xtype,              # Model type: 'arx', 'armax', 'arma', 'bjtf', 'regr'
    na=[0,1],           # Autoregressive order
    nb=[],              # Input orders (list)
    nc=[],              # Moving average orders
    nd=[],              # Denominator orders
    nf=[],              # Transfer function denominators
    delay=[],           # Input delays
    diff=[0],           # Differencing operations
    per=[],             # Seasonal periods
    eFcn='estimlm',     # Estimation function
    indexFcn='pmodmse', # Performance index
    initFcn='initrand'  # Parameter initialization
)
```

#### Methods
- `set_data(y, u)`: Set training data
- `predict(y, u)`: One-step-ahead prediction
- `getmX()`: Get model parameters as vector
- `setmX(X)`: Set model parameters from vector

### Estimation Functions

- `estimate(model, y, u=None)`: Estimate model parameters using configured algorithm
- `func_estimlm(...)`: Levenberg-Marquardt parameter estimation

### Model Selection

- `pmodaic(model, y, u)`: Compute Akaike Information Criterion
- `pmodbic(model, y, u)`: Compute Bayesian Information Criterion
- `pmodmse(model, y, u)`: Compute Mean Squared Error

## Examples and Tutorials

The package includes comprehensive examples demonstrating all functionality:

1. **Complete Model Comparison**: Compare all model types on real furnace data
2. **ARX Modeling**: Step-by-step ARX model development
3. **ARMAX Modeling**: ARMAX model with moving average components
4. **ARMA Modeling**: Univariate time series modeling
5. **BJTF Modeling**: Advanced Box-Jenkins transfer function models

Access examples in the `TimeSeriesSRC/Examples/` directory:
- Jupyter notebooks: `Examples/NoteBooks/`
- Python scripts: `Examples/PyFiles/`

## Testing

The package includes comprehensive unit tests for all core functions:

```bash
# Run all tests
python -m pytest TimeSeriesSRC/test_Scripts/

# Run specific test
python -m pytest TimeSeriesSRC/test_Scripts/test_gpac.py
```

## Use Cases

- **System Identification**: Identify linear dynamic systems from input-output data
- **Forecasting**: Generate time series forecasts with confidence intervals
- **Control System Design**: Design and validate control strategies
- **Signal Processing**: Filter and analyze time series signals
- **Economic Modeling**: Model and forecast economic time series
- **Process Monitoring**: Monitor and predict industrial process variables

## Performance

All core functions are optimized using NumPy vectorization for efficient computation on large datasets. The Levenberg-Marquardt estimation algorithm provides robust convergence for non-linear parameter optimization.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source and available under the MIT License.

## Citation

If you use this package in your research, please cite:

```
TimeSeriesSRC: A Comprehensive Python Package for Time Series Analysis
Version 0.1.0 (2022)
Translation from MATLAB to Python
```

## Authors

Martin Hagan

Amir Jafari

Lilian S. De Rivera 

## Version History

- **0.1.0** (2025): Initial release
  - Complete MATLAB to Python translation
  - Implementation of ARX, ARMAX, ARMA, BJTF models
  - Levenberg-Marquardt estimation
  - Comprehensive analysis tools (GPAC, ACF, PACF)
  - Example notebooks and scripts

## Support

For questions, issues, or feature requests, please open an issue on the project repository.

## Acknowledgments

This package is a Python translation of classical time series analysis tools originally implemented in MATLAB, providing the Python community with professional-grade time series modeling capabilities.
