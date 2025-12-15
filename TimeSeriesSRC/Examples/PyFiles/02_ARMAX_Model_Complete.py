# %% [markdown]
# # ARMAX Model - Complete Workflow
# 
# ## AutoRegressive Moving Average with eXogenous Inputs
# 
# This notebook demonstrates:
# 1. Creating an ARMAX model
# 2. Simulating data with known parameters
# 3. Estimating parameters from data
# 4. Comparing estimated vs true parameters
# 5. Making predictions
# 
# ### ARMAX Model Structure:
# ```
# y(t) = -a1*y(t-1) - ... - ana*y(t-na) +
#        b0*u(t-d) + ... + bnb*u(t-d-nb) +
#        c1*e(t-1) + ... + cnc*e(t-nc) + e(t)
# ```
# 
# This notebook replicates the Matlab test: `testarmax.m`
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
# ## Step 1: Define ARMAX Model Parameters
# 
# Parameters:
# - na = 1 (AR coefficient)
# - nb = [1] (1 input)
# - nc = 1 (MA coefficient)
# - delay = [0]
# - a = [0.7]
# - b = [[1.5, 0.8]]
# - c = [0.5]
# 
# %%
# Set parameters
na = 1
nb = [1]
nc = 1
delay = [0]

# True parameters
a_true = np.array([0.7])
b_true = [np.array([1.5, 0.8])]
c_true = np.array([0.5])

print(f"Model parameters:")
print(f"  na = {na}")
print(f"  nb = {nb}")
print(f"  nc = {nc}")
print(f"  delay = {delay}")
print(f"\nTrue parameters:")
print(f"  a = {a_true}")
print(f"  b[0] = {b_true[0]}")
print(f"  c = {c_true}")

# %% [markdown]
# ## Step 2: Create True Model and Generate Data
# 
# %%
# Create true ARMAX model
pmoda = pmodel("armax", na=na, nb=nb, nc=nc, delay=delay)
pmoda.a[0] = a_true
pmoda.b = b_true
pmoda.c[0] = c_true

print("True model created:")
print(pmoda)

# Generate random inputs and noise
n_samples = 1000
u = np.random.randn(1, n_samples)
e = np.random.randn(n_samples) * 0.3

print(f"\nGenerated data:")
print(f"  u shape: {u.shape}")
print(f"  e shape: {e.shape}")

# Simulate the model
y = func_pmodsim(pmoda, e, u)

print(f"  y shape: {y.shape}")
print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")

# %% [markdown]
# ## Step 3: Visualize Generated Data
# 
# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Output
axes[0].plot(y[:500], "b-", linewidth=0.5)
axes[0].set_title("Output y(t) - First 500 samples")
axes[0].set_ylabel("y(t)")
axes[0].grid(True, alpha=0.3)

# Input
axes[1].plot(u[0, :500], linewidth=0.5)
axes[1].set_title("Input u(t) - First 500 samples")
axes[1].set_ylabel("u(t)")
axes[1].set_xlabel("Time")
axes[1].grid(True, alpha=0.3)

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
pmodb = pmodel("armax", na=na, nb=nb, nc=nc, delay=delay)
pmodb.estimParams.show = 20
pmodb.estimParams.epochs = 100
pmodb.estimParams.goal = 0.01

print("Model created for estimation:")
print(f"Initial parameters (random):")
print(f"  a[0] = {pmodb.a[0]}")
print(f"  b[0] = {pmodb.b[0]}")
print(f"  c[0] = {pmodb.c[0]}")

print("\n" + "="*60)
print("Starting parameter estimation...")
print("="*60)

pmod_estimated, trec, stat = estimate(pmodb, y, u)

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

param_names = ["a1", "b1[0]", "b1[1]", "c1"]
for i, name in enumerate(param_names):
    print(f"{name:<15} {params_estimated[i]:<15.6f} {std_errors[i]:<15.6f} {params_true[i]:<15.6f}")

print("="*70)

sse = np.sum((params_estimated - params_true)**2)
print(f"\nSum of Squared Errors: {sse:.8f}")

rel_error = np.abs((params_estimated - params_true) / params_true) * 100
print(f"Mean Relative Error: {np.mean(rel_error):.2f}%")
print(f"Max Relative Error: {np.max(rel_error):.2f}%")

# %% [markdown]
# ## Step 6: Visualize Parameter Comparison
# 
# %%
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(param_names))
width = 0.35

ax.bar(x - width/2, params_true, width, label="True", alpha=0.8)
ax.bar(x + width/2, params_estimated, width, label="Estimated", alpha=0.8)
ax.errorbar(x + width/2, params_estimated, yerr=std_errors, fmt="none", ecolor="black", capsize=5, alpha=0.5)

ax.set_xlabel("Parameters")
ax.set_ylabel("Value")
ax.set_title("ARMAX Model: True vs Estimated Parameters")
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
# %%
# Generate new test data
n_test = 500
u_test = np.random.randn(1, n_test)
e_test = np.random.randn(n_test) * 0.3

# True output
y_true = func_pmodsim(pmoda, e_test, u_test)

# Predicted output using estimated model
y_pred = pmod_estimated.predict(y_true, u_test)

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
axes[0].set_title('ARMAX Model: True vs Predicted Output')
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
# 2. ✅ Creating ARMAX model with MA noise component
# 3. ✅ Simulating data with autocorrelated noise
# 4. ✅ Estimating parameters using Levenberg-Marquardt algorithm
# 5. ✅ Comparing estimated vs true parameters
# 6. ✅ Validating model with predictions
# 
# ### Key Results:
# - Parameter estimation successfully recovered true parameters
# - MA component correctly captures noise autocorrelation
# - Model converged within specified epochs
# 
# ### Next Steps:
# - Try different model orders (na, nb, nc)
# - Test with real-world data with correlated noise- Compare with ARX and BJTF models