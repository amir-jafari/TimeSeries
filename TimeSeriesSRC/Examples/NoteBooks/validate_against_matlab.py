"""
Validation Script: Python vs Matlab Implementation
This script replicates the exact Matlab test files and validates Python implementation
"""

import numpy as np
import sys
import os
from time import time

# Add TimeSeries root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
timeseries_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if timeseries_root not in sys.path:
    sys.path.insert(0, timeseries_root)

from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.pmodsim import func_pmodsim
from TimeSeriesSRC.Model.estimate import estimate

print("="*80)
print("PYTHON vs MATLAB VALIDATION TEST")
print("="*80)
print("\nThis script replicates exact Matlab test parameters and validates results")
print()

# Set random seed for reproducibility
np.random.seed(42)

results = {}

#==============================================================================
# TEST 1: ARX Model (replicates testarx.m)
#==============================================================================
print("\n" + "="*80)
print("TEST 1: ARX Model (testarx.m)")
print("="*80)

print("\nMatlab Parameters:")
print("  na = 2")
print("  nb = [1, 1, 2]")
print("  delay = [1, 2, 3]")
print("  a = [-1, 0.25]")
print("  b = {[1, 1], [2, 3], [1, -1, 0.25]}")
print("  samples = 2000")

# Set parameters exactly as in Matlab testarx.m
na = 2
nb = [1, 1, 2]
delay = [1, 2, 3]
a_true = np.array([-1.0, 0.25])
b_true = [
    np.array([1.0, 1.0]),
    np.array([2.0, 3.0]),
    np.array([1.0, -1.0, 0.25])
]

# Generate data
n_samples = 2000
u = np.random.randn(3, n_samples)
e = np.random.randn(n_samples) * 0.5

# Create true model
pmoda_arx = pmodel('arx', na=na, nb=nb, delay=delay)
pmoda_arx.a[0] = a_true
pmoda_arx.b = b_true

# Simulate
y_arx = func_pmodsim(pmoda_arx, e, u)

# Estimate
pmodb_arx = pmodel('arx', na=na, nb=nb, delay=delay)
pmodb_arx.estimParams.show = 0  # Silent mode
pmodb_arx.estimParams.epochs = 100

start = time()
pmod_est_arx, trec_arx, stat_arx = estimate(pmodb_arx, y_arx, u)
elapsed = time() - start

# Compare
params_est_arx = pmod_est_arx.getmX()
params_true_arx = pmoda_arx.getmX()
sse_arx = np.sum((params_est_arx - params_true_arx)**2)

print("\nResults:")
print(f"  Estimation time: {elapsed:.3f} seconds")
print(f"  Final MSE: {trec_arx['perf'][-1]:.6f}")
print(f"  SSE (parameters): {sse_arx:.8f}")
print(f"  Mean parameter error: {np.mean(np.abs(params_est_arx - params_true_arx)):.6f}")
print(f"  Max parameter error: {np.max(np.abs(params_est_arx - params_true_arx)):.6f}")

# Detailed comparison
print("\n  Parameter Comparison:")
print(f"  {'Param':<10} {'True':<12} {'Estimated':<12} {'Error':<12}")
print("  " + "-"*46)
param_names = ['a1', 'a2', 'b1[0]', 'b1[1]', 'b2[0]', 'b2[1]', 'b3[0]', 'b3[1]', 'b3[2]']
for i, name in enumerate(param_names):
    err = params_est_arx[i] - params_true_arx[i]
    print(f"  {name:<10} {params_true_arx[i]:<12.6f} {params_est_arx[i]:<12.6f} {err:<12.6f}")

results['ARX'] = {
    'sse': sse_arx,
    'mse': trec_arx['perf'][-1],
    'time': elapsed,
    'pass': sse_arx < 0.01
}

#==============================================================================
# TEST 2: ARMAX Model (replicates testarmax.m)
#==============================================================================
print("\n" + "="*80)
print("TEST 2: ARMAX Model (testarmax.m)")
print("="*80)

print("\nMatlab Parameters:")
print("  na = 2")
print("  nb = [1, 1, 2]")
print("  nc = 1")
print("  delay = [1, 2, 3]")
print("  a = [-1, 0.25]")
print("  b = {[1, 1], [2, 3], [1, -1, 0.25]}")
print("  c = [0.65]")
print("  samples = 2000")

# Set parameters exactly as in Matlab testarmax.m
na = 2
nb = [1, 1, 2]
nc = 1
delay = [1, 2, 3]
a_true = np.array([-1.0, 0.25])
b_true = [
    np.array([1.0, 1.0]),
    np.array([2.0, 3.0]),
    np.array([1.0, -1.0, 0.25])
]
c_true = np.array([0.65])

# Generate data
u = np.random.randn(3, 2000)
e = np.random.randn(2000) * 0.5

# Create true model
pmoda_armax = pmodel('armax', na=na, nb=nb, nc=nc, delay=delay)
pmoda_armax.a[0] = a_true
pmoda_armax.b = b_true
pmoda_armax.c[0] = c_true

# Simulate
y_armax = func_pmodsim(pmoda_armax, e, u)

# Estimate
pmodb_armax = pmodel('armax', na=na, nb=nb, nc=nc, delay=delay)
pmodb_armax.estimParams.show = 0
pmodb_armax.estimParams.epochs = 100

start = time()
pmod_est_armax, trec_armax, stat_armax = estimate(pmodb_armax, y_armax, u)
elapsed = time() - start

# Compare
params_est_armax = pmod_est_armax.getmX()
params_true_armax = pmoda_armax.getmX()
sse_armax = np.sum((params_est_armax - params_true_armax)**2)

print("\nResults:")
print(f"  Estimation time: {elapsed:.3f} seconds")
print(f"  Final MSE: {trec_armax['perf'][-1]:.6f}")
print(f"  SSE (parameters): {sse_armax:.8f}")
print(f"  Mean parameter error: {np.mean(np.abs(params_est_armax - params_true_armax)):.6f}")
print(f"  Max parameter error: {np.max(np.abs(params_est_armax - params_true_armax)):.6f}")

# Detailed comparison
print("\n  Parameter Comparison:")
print(f"  {'Param':<10} {'True':<12} {'Estimated':<12} {'Error':<12}")
print("  " + "-"*46)
param_names = ['a1', 'a2', 'b1[0]', 'b1[1]', 'b2[0]', 'b2[1]', 'b3[0]', 'b3[1]', 'b3[2]', 'c1']
for i, name in enumerate(param_names):
    err = params_est_armax[i] - params_true_armax[i]
    print(f"  {name:<10} {params_true_armax[i]:<12.6f} {params_est_armax[i]:<12.6f} {err:<12.6f}")

results['ARMAX'] = {
    'sse': sse_armax,
    'mse': trec_armax['perf'][-1],
    'time': elapsed,
    'pass': sse_armax < 0.01
}

#==============================================================================
# TEST 3: ARMA Model (replicates testarma.m)
#==============================================================================
print("\n" + "="*80)
print("TEST 3: ARMA Model (testarma.m)")
print("="*80)

print("\nMatlab Parameters:")
print("  nc = [1, 2, 1]")
print("  nd = [2, 1, 1]")
print("  diff = [0, 0, 0]")
print("  per = [3, 12]")
print("  c = {[0.65], [0.5, -0.25], [-0.6]}")
print("  d = {[-1, 0.25], [0.7], [0.3]}")
print("  samples = 2000")

# Set parameters exactly as in Matlab testarma.m
nc = [1, 2, 1]
nd = [2, 1, 1]
diff = [0, 0, 0]
per = [3, 12]
c_true = [np.array([0.65]), np.array([0.5, -0.25]), np.array([-0.6])]
d_true = [np.array([-1.0, 0.25]), np.array([0.7]), np.array([0.3])]

# Generate data
e = np.random.randn(2000) * 0.5

# Create true model
pmoda_arma = pmodel('arma', nc=nc, nd=nd, diff=diff, per=per)
pmoda_arma.c = c_true
pmoda_arma.d = d_true

# Simulate
y_arma = func_pmodsim(pmoda_arma, e)

# Estimate
pmodb_arma = pmodel('arma', nc=nc, nd=nd, diff=diff, per=per)
pmodb_arma.estimParams.show = 0
pmodb_arma.estimParams.epochs = 100

start = time()
pmod_est_arma, trec_arma, stat_arma = estimate(pmodb_arma, y_arma)
elapsed = time() - start

# Compare
params_est_arma = pmod_est_arma.getmX()
params_true_arma = pmoda_arma.getmX()
sse_arma = np.sum((params_est_arma - params_true_arma)**2)

print("\nResults:")
print(f"  Estimation time: {elapsed:.3f} seconds")
print(f"  Final MSE: {trec_arma['perf'][-1]:.6f}")
print(f"  SSE (parameters): {sse_arma:.8f}")
print(f"  Mean parameter error: {np.mean(np.abs(params_est_arma - params_true_arma)):.6f}")
print(f"  Max parameter error: {np.max(np.abs(params_est_arma - params_true_arma)):.6f}")

# Detailed comparison
print("\n  Parameter Comparison:")
print(f"  {'Param':<10} {'True':<12} {'Estimated':<12} {'Error':<12}")
print("  " + "-"*46)
param_names = ['c1[0]', 'c2[0]', 'c2[1]', 'c3[0]', 'd1[0]', 'd1[1]', 'd2[0]', 'd3[0]']
for i, name in enumerate(param_names):
    err = params_est_arma[i] - params_true_arma[i]
    print(f"  {name:<10} {params_true_arma[i]:<12.6f} {params_est_arma[i]:<12.6f} {err:<12.6f}")

results['ARMA'] = {
    'sse': sse_arma,
    'mse': trec_arma['perf'][-1],
    'time': elapsed,
    'pass': sse_arma < 0.1  # More lenient for ARMA
}

#==============================================================================
# TEST 4: BJTF Model (replicates testbjtf.m)
#==============================================================================
print("\n" + "="*80)
print("TEST 4: BJTF Model (testbjtf.m)")
print("="*80)

print("\nMatlab Parameters:")
print("  nb = [1, 1, 2]")
print("  nc = [1, 2, 1]")
print("  nd = [2, 1, 1]")
print("  nf = [1, 2, 1]")
print("  delay = [1, 2, 3]")
print("  diff = [0, 0, 0]")
print("  per = [3, 12]")
print("  samples = 200")  # Note: Matlab uses 200 for BJTF

# Set parameters exactly as in Matlab testbjtf.m
nb = [1, 1, 2]
nc = [1, 2, 1]
nd = [2, 1, 1]
nf = [1, 2, 1]
delay = [1, 2, 3]
diff = [0, 0, 0]
per = [3, 12]

b_true = [np.array([1.0, 1.0]), np.array([2.0, 3.0]), np.array([1.0, -1.0, 0.25])]
c_true = [np.array([0.65]), np.array([0.5, -0.25]), np.array([-0.6])]
d_true = [np.array([-1.0, 0.25]), np.array([0.7]), np.array([0.3])]
f_true = [np.array([0.35]), np.array([-1.0, 0.3]), np.array([0.5])]

# Generate data
u = np.random.randn(3, 200)
e = np.random.randn(200) * 0.5

# Create true model
pmoda_bjtf = pmodel('bjtf', nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per=per)
pmoda_bjtf.b = b_true
pmoda_bjtf.c = c_true
pmoda_bjtf.d = d_true
pmoda_bjtf.f = f_true

# Simulate
y_bjtf = func_pmodsim(pmoda_bjtf, e, u)

# Estimate
pmodb_bjtf = pmodel('bjtf', nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per=per)
pmodb_bjtf.estimParams.show = 0
pmodb_bjtf.estimParams.epochs = 150

start = time()
pmod_est_bjtf, trec_bjtf, stat_bjtf = estimate(pmodb_bjtf, y_bjtf, u)
elapsed = time() - start

# Compare
params_est_bjtf = pmod_est_bjtf.getmX()
params_true_bjtf = pmoda_bjtf.getmX()
sse_bjtf = np.sum((params_est_bjtf - params_true_bjtf)**2)

print("\nResults:")
print(f"  Estimation time: {elapsed:.3f} seconds")
print(f"  Final MSE: {trec_bjtf['perf'][-1]:.6f}")
print(f"  SSE (parameters): {sse_bjtf:.8f}")
print(f"  Mean parameter error: {np.mean(np.abs(params_est_bjtf - params_true_bjtf)):.6f}")
print(f"  Max parameter error: {np.max(np.abs(params_est_bjtf - params_true_bjtf)):.6f}")

print("\n  First 10 Parameter Comparison:")
print(f"  {'Index':<10} {'True':<12} {'Estimated':<12} {'Error':<12}")
print("  " + "-"*46)
for i in range(min(10, len(params_true_bjtf))):
    err = params_est_bjtf[i] - params_true_bjtf[i]
    print(f"  {i:<10} {params_true_bjtf[i]:<12.6f} {params_est_bjtf[i]:<12.6f} {err:<12.6f}")

results['BJTF'] = {
    'sse': sse_bjtf,
    'mse': trec_bjtf['perf'][-1],
    'time': elapsed,
    'pass': sse_bjtf < 0.5  # More lenient for BJTF (complex model, less data)
}

#==============================================================================
# SUMMARY
#==============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

print(f"\n{'Model':<15} {'SSE':<15} {'Final MSE':<15} {'Time (s)':<12} {'Status':<10}")
print("-"*67)

for model, res in results.items():
    status = "PASS" if res['pass'] else "FAIL"
    status_symbol = "✓" if res['pass'] else "✗"
    print(f"{model:<15} {res['sse']:<15.8f} {res['mse']:<15.6f} {res['time']:<12.3f} {status_symbol} {status}")

print("\n" + "="*80)

# Overall pass/fail
all_pass = all(res['pass'] for res in results.values())
if all_pass:
    print("OVERALL RESULT: ALL TESTS PASSED ✓")
    print("\nPython implementation successfully replicates Matlab functionality!")
    print("Parameters are recovered with high accuracy.")
else:
    print("OVERALL RESULT: SOME TESTS FAILED ✗")
    failed_models = [model for model, res in results.items() if not res['pass']]
    print(f"\nFailed models: {', '.join(failed_models)}")
    print("Consider increasing epochs or adjusting convergence criteria.")

print("="*80)

# Save results to file
import json
with open('validation_results.json', 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    json_results = {}
    for model, res in results.items():
        json_results[model] = {
            'sse': float(res['sse']),
            'mse': float(res['mse']),
            'time': float(res['time']),
            'pass': bool(res['pass'])
        }
    json.dump(json_results, f, indent=2)

print("\nResults saved to: validation_results.json")