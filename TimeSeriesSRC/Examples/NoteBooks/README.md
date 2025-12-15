# Time Series Models - Comprehensive Notebooks

This directory contains complete end-to-end notebooks for all time series model types.

## Available Notebooks

### 1. **01_ARX_Model_Complete.ipynb**
**AutoRegressive with eXogenous Inputs**

- Complete workflow: GPAC → create → simulate → estimate → validate → predict
- Model structure: `y(t) = -a*y(t-1) + b*u(t-d) + e(t)`
- Multiple inputs demonstration
- GPAC visualization for model order selection
- Prediction and validation on test data
- Replicates Matlab test: `testarx.m`

**When to use:** System with measurable external inputs affecting output

---

### 2. **02_ARMAX_Model_Complete.ipynb**
**AutoRegressive Moving Average with eXogenous Inputs**

- Complete workflow: GPAC → create → simulate → estimate → validate → predict
- Extends ARX with Moving Average noise component
- Model structure: `y(t) = ARX + c*e(t-1)`
- Handles autocorrelated noise
- GPAC visualization for model order selection
- Prediction and validation on test data
- Replicates Matlab test: `testarmax.m`

**When to use:** ARX model where noise has autocorrelation

---

### 3. **03_ARMA_Model_Complete.ipynb**
**AutoRegressive Moving Average**

- Complete workflow: GPAC → create → simulate → estimate → validate → predict
- No exogenous inputs - models the series itself
- Model structure: `y(t) = -d*y(t-1) + c*e(t-1) + e(t)`
- Pure time series forecasting
- GPAC visualization for model order selection
- Prediction and validation on test data
- Replicates Matlab test: `testarma.m`

**When to use:** No external inputs, modeling/forecasting the series itself

---

### 4. **04_BJTF_Model_Complete.ipynb**
**Box-Jenkins Transfer Function**

- Complete workflow: GPAC → create → simulate → estimate → validate → predict
- Most general form
- Model structure: `y(t) = [B/F]*u(t) + [C/D]*e(t)`
- Separate dynamics for input and noise
- GPAC visualization for model order selection
- Prediction and validation on test data
- Replicates Matlab test: `testbjtf.m`

**When to use:** Complex systems requiring separate input and noise dynamics

---

### 5. **RUN_ALL_TESTS.ipynb**
**Comprehensive Test Suite**

- Tests all 4 model types (ARX, ARMAX, ARMA, BJTF)
- Validates parameter estimation
- Pass/Fail status for each model
- Summary report with SSE and timing

**Use this to:** Verify installation and validate all fixes are working

---

## How to Use

### Quick Start

```bash
# Navigate to notebooks directory
cd "D:\Deep Learning Book\All_repo_source_file\Packages\TimeSeries\TimeSeriesSRC\Examples\NoteBooks"

# Start Jupyter
jupyter notebook
```

### Recommended Order

1. **RUN_ALL_TESTS.ipynb** - Verify everything works (5 min)
2. **01_ARX_Model_Complete.ipynb** - Learn complete workflow with GPAC and prediction (20 min)
3. **02_ARMAX_Model_Complete.ipynb** - Explore ARMAX with GPAC and prediction (15 min)
4. **03_ARMA_Model_Complete.ipynb** - Explore ARMA with GPAC and prediction (15 min)
5. **04_BJTF_Model_Complete.ipynb** - Explore BJTF with GPAC and prediction (15 min)

---

## Each Notebook Includes

✅ **GPAC Analysis**: Model order selection using Generalized Partial Autocorrelation
✅ **Model Creation**: Set up with known parameters
✅ **Data Generation**: Simulate synthetic data
✅ **Visualization**: Plot inputs, outputs, and results
✅ **Parameter Estimation**: Levenberg-Marquardt algorithm
✅ **Validation**: Compare estimated vs true parameters
✅ **Prediction**: Generate predictions on test data
✅ **Analysis**: SSE, relative error, standard errors, prediction MSE

---

## Model Comparison

| Model | Inputs | AR | MA | Complexity | Use Case |
|-------|--------|----|----|------------|----------|
| **ARX** | ✓ | ✓ | ✗ | Low | System ID with inputs |
| **ARMAX** | ✓ | ✓ | ✓ | Medium | ARX + noise correlation |
| **ARMA** | ✗ | ✓ | ✓ | Low | Pure forecasting |
| **BJTF** | ✓ | ✓ | ✓ | High | Complex processes |

---

## Requirements

```bash
pip install numpy scipy matplotlib jupyter
```

---

## Validation Status

All notebooks have been:
- ✅ Tested and validated
- ✅ Compared with Matlab implementations
- ✅ Verified to produce correct results
- ✅ Fixed for all import and runtime errors

---

## Troubleshooting

### Issue: Import errors
**Solution**: Make sure you run Cell 1 (imports) first in each notebook

### Issue: Estimation not converging
**Solution**: Increase epochs or adjust learning rate:
```python
pmodb.estimParams.epochs = 200
pmodb.estimParams.mu = 0.01
```

### Issue: Results don't match
**Solution**: Check random seed is set: `np.random.seed(42)`

---

## Common Workflow Pattern

All notebooks follow the same pattern:

1. **Import libraries** and set up paths
2. **Define parameters** (na, nb, nc, etc.)
3. **Create true model** with known parameters
4. **Generate data** using simulation
5. **Visualize** inputs and outputs
6. **GPAC Analysis** for model order selection (NEW!)
7. **Create estimation model** with random init
8. **Estimate parameters** using Levenberg-Marquardt
9. **Compare** estimated vs true parameters
10. **Visualize** results and training history
11. **Predict and validate** on test data (NEW!)

---

## Next Steps

After completing these notebooks:

1. **Apply to your data**: Load your own CSV/MAT files
2. **Model selection**: Use GPAC for order selection
3. **Cross-validation**: Test on held-out data
4. **Production**: Integrate into your workflow

---

## Support

For issues:
1. Check `TROUBLESHOOTING.md` in the parent directory
2. Run `RUN_ALL_TESTS.ipynb` to verify installation
3. Review notebook cell outputs for errors

---

**Last Updated**: December 2024
**Status**: All notebooks tested and ready to use
**Python Version**: 3.8+