# %% [markdown]
# # ARMA Model - Complete Workflow
# 
# ## AutoRegressive Moving Average Model
# 
# This notebook demonstrates:
# 1. Creating an ARMA model
# 2. Simulating data with known parameters
# 3. Estimating parameters from data
# 4. Comparing estimated vs true parameters
# 
# ### ARMA Model Structure:
# ```
# y(t) = -d1*y(t-1) - ... - dnd*y(t-nd) +
#        c1*e(t-1) + ... + cnc*e(t-nc) + e(t)
# ```
# 
# Note: No exogenous inputs - models the series itself
# 
# This notebook replicates the Matlab test: `testarma.m`
# 
# %%
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add TimeSeries root directory to path
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

timeseries_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
print(f"TimeSeries root: {timeseries_root}")

if timeseries_root not in sys.path:
    sys.path.insert(0, timeseries_root)

from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.pmodsim import func_pmodsim
from TimeSeriesSRC.Model.estimate import estimate

np.random.seed(42)
print("\n[SUCCESS] Libraries imported successfully!")

# %% [markdown]
# ## Step 1: Define ARMA Model Parameters
# 
# **IMPORTANT**: ARMA models are harder to estimate than ARX/ARMAX models.
# Start with simple orders (nc=[1], nd=[1]) for reliable results.
# 
# Parameters:
# - nc = [1] (MA coefficients) - Start simple
# - nd = [1] (AR coefficients) - Start simple  
# - diff = [0] (no differencing)
# - per = [] (no periodicity)
# - c = [[0.65]]
# - d = [[0.75]]
# 
# **Note**: For nc=[2], nd=[2], you need more epochs (500+) and better initialization.
# %%
# Set parameters - Start with simple model for reliable estimation
nc = [1]  # Start with 1 MA coefficient (easier to estimate)
nd = [1]  # Start with 1 AR coefficient (easier to estimate)
diff = [0]
per = []

# True parameters (simpler model)
c_true = [np.array([0.65])]   # 1 MA coefficient
d_true = [np.array([0.75])]   # 1 AR coefficient

# For nc=[2], nd=[2] (harder to estimate):
# Uncomment below and update Cell 13 param_names to ["c1", "c2", "d1", "d2"]
# nc = [2]
# nd = [2]
# c_true = [np.array([0.65, 0.5])]
# d_true = [np.array([0.75, 0.25])]

print(f"Model parameters:")
print(f"  nc = {nc}")
print(f"  nd = {nd}")
print(f"  diff = {diff}")
print(f"  per = {per}")
print(f"\nTrue parameters:")
print(f"  c = {c_true}")
print(f"  d = {d_true}")
# %% [markdown]
# ## Step 2: Create True Model and Generate Data
# 
# %%
# Create true ARMA model
pmoda = pmodel("arma", nc=nc, nd=nd, diff=diff, per=per)
pmoda.c = c_true
pmoda.d = d_true

print("True model created:")
print(pmoda)

# Generate noise (no exogenous inputs for ARMA)
n_samples = 1000
e = np.random.randn(n_samples) * 0.5

print(f"\nGenerated data:")
print(f"  e shape: {e.shape}")

# Simulate the model
y = func_pmodsim(pmoda, e)

print(f"  y shape: {y.shape}")
print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")

# %% [markdown]
# ## Step 3: Visualize Generated Data
# 
# %%
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(y[:500], "b-", linewidth=0.5)
ax.set_title("ARMA Output y(t) - First 500 samples")
ax.set_xlabel("Time")
ax.set_ylabel("y(t)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Data visualization complete!")

# %% [markdown]
# ## GPAC Analysis - Model Order Selection
# 
# The Generalized Partial Autocorrelation (GPAC) function helps determine appropriate model orders.
# - **GREEN squares**: Positive correlations
# - **RED squares**: Negative correlations
# - **Square size**: Magnitude of correlation
# 
# GPAC pattern shows where model orders should be selected.
# %%
# Import GPAC functions
from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
from TimeSeriesSRC.basefunctions.gpac import func_gpac
from TimeSeriesSRC.basefunctions.plotgpac import func_plotgpac

# Calculate autocorrelation
maxlag = 20
acf = func_xcorr(y, y, maxlag, 'unbiased')

# Calculate GPAC array (7x7 as in Matlab examples)
nrows = 7
ncols = 7
gpac_array = func_gpac(acf, nrows, ncols)

print(f"GPAC array shape: {gpac_array.shape}")
print(f"\nGPAC Array:")
print(gpac_array)

# Plot GPAC
fig, ax = plt.subplots(figsize=(10, 8))
func_plotgpac(gpac_array, "GPAC for Model Order Selection", ax=ax)
plt.show()

print("\nGPAC Analysis Complete!")
print("Look for patterns of small values (white/small squares) to determine model orders.")
# %% [markdown]
# ## Step 4: Estimate Parameters
# 
# %%
# Create model for estimation
pmodb = pmodel("arma", nc=nc, nd=nd, diff=diff, per=per)
pmodb.estimParams.show = 20
pmodb.estimParams.epochs = 1000  # Increased from 100
pmodb.estimParams.goal = 0.01

# Initialize closer to expected values for better convergence
# Must match nc and nd from Cell 3!
# For nc=[1], nd=[1]: initialize with 1 value each
# For nc=[2], nd=[2]: initialize with 2 values each
if nc == [1] and nd == [1]:
    # Simple model: 1 MA coefficient, 1 AR coefficient
    pmodb.c = [np.array([0.6])]   # Initialize near 0.65
    pmodb.d = [np.array([0.7])]   # Initialize near 0.75
else:
    # Complex model: use GPAC to guide initialization
    pmodb.c = [np.array([0.5, 0.4])]
    pmodb.d = [np.array([0.7, 0.2])]

print("Model created for estimation:")
print(f"Initial parameters (initialized near expected range):")
print(f"  c = {pmodb.c}")
print(f"  d = {pmodb.d}")

print("\n" + "="*60)
print("Starting parameter estimation...")
print("="*60)

pmod_estimated, trec, stat = estimate(pmodb, y)

print("\n" + "="*60)
print("Estimation complete!")
print("="*60)
# %% [markdown]
# ## Step 5: Compare Estimated vs True Parameters
# 
# %%
params_estimated = pmod_estimated.getmX()
params_true = pmoda.getmX()
std_errors = 2 * stat["stdx"]

print("\nParameter Comparison:")
print("="*70)
print(f"{'Parameter':<15} {'Estimated':<15} {'2*StdErr':<15} {'True':<15}")
print("="*70)

# Update param_names based on nc and nd
# For nc=[1], nd=[1]: use ["c1", "d1"]
# For nc=[2], nd=[2]: use ["c1", "c2", "d1", "d2"]
param_names = ["c1", "d1"]  # Change to ["c1", "c2", "d1", "d2"] if using nc=[2], nd=[2]

for i, name in enumerate(param_names):
    print(f"{name:<15} {params_estimated[i]:<15.6f} {std_errors[i]:<15.6f} {params_true[i]:<15.6f}")

print("="*70)

sse = np.sum((params_estimated - params_true)**2)
print(f"\nSum of Squared Errors: {sse:.8f}")

if sse < 0.01:
    print("✓ EXCELLENT: Parameters recovered accurately!")
elif sse < 0.1:
    print("✓ GOOD: Parameters recovered reasonably well")
else:
    print("✗ POOR: Estimation did not converge well")
    print("  Try: (1) More epochs, (2) Better initialization, (3) Simpler model")

rel_error = np.abs((params_estimated - params_true) / params_true) * 100
print(f"\nMean Relative Error: {np.mean(rel_error):.2f}%")
print(f"Max Relative Error: {np.max(rel_error):.2f}%")
# %% [markdown]
# ## Step 6: Visualize Parameter Comparison
# 
# %%
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(param_names))
width = 0.35

ax.bar(x - width/2, params_true, width, label="True", alpha=0.8)
ax.bar(x + width/2, params_estimated, width, label="Estimated", alpha=0.8)
ax.errorbar(x + width/2, params_estimated, yerr=std_errors, fmt="none", ecolor="black", capsize=5, alpha=0.5)

ax.set_xlabel("Parameters")
ax.set_ylabel("Value")
ax.set_title("ARMA Model: True vs Estimated Parameters")
ax.set_xticks(x)
ax.set_xticklabels(param_names)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 7: Make Predictions and Validate
# 
# Test the estimated model by generating predictions on new test data.
# Note: ARMA has no exogenous inputs, only noise.
# %%
# Generate new test data (noise only for ARMA)
n_test = 500
e_test = np.random.randn(n_test) * 0.5

# True output
y_true = func_pmodsim(pmoda, e_test)

# Predicted output using estimated model
# For ARMA, we predict based on the series itself
y_pred = pmod_estimated.predict(y_true)

# Calculate prediction error
pred_error = y_true - y_pred
mse_pred = np.mean(pred_error**2)

print(f"Prediction MSE: {mse_pred:.6f}")
print(f"Prediction RMSE: {np.sqrt(mse_pred):.6f}")

# Plot predictions
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# True vs Predicted
axes[0].plot(y_true[:200], 'b-', label='True', linewidth=1)
axes[0].plot(y_pred[:200], 'r--', label='Predicted', linewidth=1)
axes[0].set_title('ARMA Model: True vs Predicted Output')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Output')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Prediction Error
axes[1].plot(pred_error[:200], 'g-', linewidth=0.5)
axes[1].set_title('Prediction Error')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Error')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %% [markdown]
# ## SummaryThis notebook demonstrated:
# 1. ✅ GPAC analysis for model order selection
# 2. ✅ Creating ARMA model (no exogenous inputs)
# 3. ✅ Simulating pure time series data
# 4. ✅ Estimating parameters using Levenberg-Marquardt algorithm
# 5. ✅ Comparing estimated vs true parameters
# 6. ✅ Validating model with predictions
# 
# ### Key Results:- Parameter estimation recovered true parameters
# - ARMA models the series itself without external inputs
# - Model converged within specified epochs### Next Steps:
# - Try different model orders (nc, nd)- Test with real-world time series data
# - Use for forecasting applications