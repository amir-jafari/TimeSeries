# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from time import time

# Add TimeSeries root directory to path
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

timeseries_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
print(f"TimeSeries root: {timeseries_root}")

if timeseries_root not in sys.path:
    sys.path.insert(0, timeseries_root)

from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.pmodsim import func_pmodsim
from TimeSeriesSRC.Model.estimate import estimate  # Correct function name

# Set random seed
np.random.seed(42)

print("\n[OK] Libraries imported successfully")

# Test results storage
test_results = {}
# %% [markdown]
# # Time Series Models - Comprehensive Test Suite
# 
# This notebook tests all model types:
# - **ARX**: AutoRegressive with eXogenous inputs
# - **ARMAX**: ARX with Moving Average
# - **ARMA**: AutoRegressive Moving Average
# - **BJTF**: Box-Jenkins Transfer Function
# 
# Each test:
# 1. Creates a model with known parameters
# 2. Simulates data
# 3. Estimates parameters from the data
# 4. Compares estimated vs true parameters
# 5. Reports PASS/FAIL status
# %%
print("="*70)
print("TEST 1: ARX MODEL")
print("="*70)

start_time = time()

try:
    # Parameters (from testarx.m)
    na = 2
    nb = [1, 1, 2]
    delay = [1, 2, 3]
    
    # True parameters
    a_true = np.array([-1.0, 0.25])
    b_true = [
        np.array([1.0, 1.0]),
        np.array([2.0, 3.0]),
        np.array([1.0, -1.0, 0.25])
    ]
    
    # Create true model
    pmoda = pmodel('arx', na=na, nb=nb, delay=delay)
    pmoda.a[0] = a_true
    pmoda.b = b_true
    
    # Generate data
    n = 2000
    u = np.random.randn(3, n)
    e = np.random.randn(n) * 0.5
    y = func_pmodsim(pmoda, e, u)
    
    print(f"[OK] Generated {n} samples")
    
    # Estimate
    pmodb = pmodel('arx', na=na, nb=nb, delay=delay)
    pmodb.estimParams.show = 50
    pmodb.estimParams.epochs = 100
    
    pmod_est, trec, stat = estimate(pmodb, y, u)
    
    # Compare
    params_est = pmod_est.getmX()
    params_true = pmoda.getmX()
    sse = np.sum((params_est - params_true)**2)
    
    print(f"\n[RESULT] Sum of Squared Errors: {sse:.8f}")
    
    # Success criterion
    success = sse < 0.01
    
    test_results['ARX'] = {
        'status': 'PASS' if success else 'FAIL',
        'sse': sse,
        'time': time() - start_time
    }
    
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n{status} ARX Model Test")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    test_results['ARX'] = {'status': 'ERROR', 'error': str(e)}

print(f"Time elapsed: {time() - start_time:.2f}s")
print("="*70)
# %%
print("\n" + "="*70)
print("TEST 2: ARMAX MODEL")
print("="*70)

start_time = time()

try:
    # Parameters
    na = 1
    nb = [1]
    nc = 1
    delay = [0]
    
    # True parameters
    a_true = np.array([0.7])
    b_true = [np.array([1.5, 0.8])]
    c_true = np.array([0.5])
    
    # Create and simulate
    pmoda = pmodel('armax', na=na, nb=nb, nc=nc, delay=delay)
    pmoda.a[0] = a_true
    pmoda.b = b_true
    pmoda.c[0] = c_true
    
    n = 1000
    u = np.random.randn(1, n)
    e = np.random.randn(n) * 0.3
    y = func_pmodsim(pmoda, e, u)
    
    print(f"[OK] Generated {n} samples")
    
    # Estimate
    pmodb = pmodel('armax', na=na, nb=nb, nc=nc, delay=delay)
    pmodb.estimParams.show = 50
    pmodb.estimParams.epochs = 100
    
    pmod_est, trec, stat = estimate(pmodb, y, u)
    
    # Compare
    params_est = pmod_est.getmX()
    params_true = pmoda.getmX()
    sse = np.sum((params_est - params_true)**2)
    
    print(f"\n[RESULT] Sum of Squared Errors: {sse:.8f}")
    
    success = sse < 0.05
    
    test_results['ARMAX'] = {
        'status': 'PASS' if success else 'FAIL',
        'sse': sse,
        'time': time() - start_time
    }
    
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n{status} ARMAX Model Test")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    test_results['ARMAX'] = {'status': 'ERROR', 'error': str(e)}

print(f"Time elapsed: {time() - start_time:.2f}s")
print("="*70)
# %%
print("\n" + "="*70)
print("TEST 3: ARMA MODEL")
print("="*70)

start_time = time()

try:
    # Parameters (simplified from testarma.m)
    nc = [1]
    nd = [1]
    diff = [0]
    per = []
    
    # True parameters
    c_true = [np.array([0.65])]
    d_true = [np.array([0.75])]
    
    # Create and simulate
    pmoda = pmodel('arma', nc=nc, nd=nd, diff=diff, per=per)
    pmoda.c = c_true
    pmoda.d = d_true
    
    n = 1000
    e = np.random.randn(n) * 0.5
    y = func_pmodsim(pmoda, e)
    
    print(f"[OK] Generated {n} samples")
    
    # Estimate
    pmodb = pmodel('arma', nc=nc, nd=nd, diff=diff, per=per)
    pmodb.estimParams.show = 50
    pmodb.estimParams.epochs = 100
    
    pmod_est, trec, stat = estimate(pmodb, y)  # Using 'estimate' function
    
    # Compare
    params_est = pmod_est.getmX()
    params_true = pmoda.getmX()
    sse = np.sum((params_est - params_true)**2)
    
    print(f"\n[RESULT] Sum of Squared Errors: {sse:.8f}")
    
    success = sse < 0.05
    
    test_results['ARMA'] = {
        'status': 'PASS' if success else 'FAIL',
        'sse': sse,
        'time': time() - start_time
    }
    
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n{status} ARMA Model Test")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    test_results['ARMA'] = {'status': 'ERROR', 'error': str(e)}

print(f"Time elapsed: {time() - start_time:.2f}s")
print("="*70)
# %%
print("\n" + "="*70)
print("TEST 4: BJTF MODEL")
print("="*70)

start_time = time()

try:
    # Parameters (simplified from testbjtf.m)
    nb = [1]
    nc = [1]
    nd = [1]
    nf = [1]
    delay = [1]
    diff = [0]
    per = []
    
    # True parameters
    b_true = [np.array([1.0, 0.8])]
    c_true = [np.array([0.5])]
    d_true = [np.array([0.7])]
    f_true = [np.array([0.3])]
    
    # Create and simulate
    pmoda = pmodel('bjtf', nb=nb, nc=nc, nd=nd, nf=nf, 
                   delay=delay, diff=diff, per=per)
    pmoda.b = b_true
    pmoda.c = c_true
    pmoda.d = d_true
    pmoda.f = f_true
    
    n = 500
    u = np.random.randn(1, n)
    e = np.random.randn(n) * 0.3
    y = func_pmodsim(pmoda, e, u)
    
    print(f"[OK] Generated {n} samples")
    
    # Estimate
    pmodb = pmodel('bjtf', nb=nb, nc=nc, nd=nd, nf=nf,
                   delay=delay, diff=diff, per=per)
    pmodb.estimParams.show = 50
    pmodb.estimParams.epochs = 100
    
    pmod_est, trec, stat = estimate(pmodb, y, u)
    
    # Compare
    params_est = pmod_est.getmX()
    params_true = pmoda.getmX()
    sse = np.sum((params_est - params_true)**2)
    
    print(f"\n[RESULT] Sum of Squared Errors: {sse:.8f}")
    
    success = sse < 0.1
    
    test_results['BJTF'] = {
        'status': 'PASS' if success else 'FAIL',
        'sse': sse,
        'time': time() - start_time
    }
    
    status = "[PASS]" if success else "[FAIL]"
    print(f"\n{status} BJTF Model Test")
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    test_results['BJTF'] = {'status': 'ERROR', 'error': str(e)}

print(f"Time elapsed: {time() - start_time:.2f}s")
print("="*70)
# %% [markdown]
# ## Visualization of Results
# %% [markdown]
# ## Test Summary
# %%
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

total_tests = len(test_results)
passed = sum(1 for r in test_results.values() if r['status'] == 'PASS')
failed = sum(1 for r in test_results.values() if r['status'] == 'FAIL')
errors = sum(1 for r in test_results.values() if r['status'] == 'ERROR')

print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Errors: {errors}")

print("\nDetailed Results:")
print("-" * 70)
print(f"{'Model':<15} {'Status':<10} {'SSE':<15} {'Time (s)':<10}")
print("-" * 70)

for model, result in test_results.items():
    status = result['status']
    sse = f"{result.get('sse', 0):.6f}" if 'sse' in result else 'N/A'
    elapsed = f"{result.get('time', 0):.2f}" if 'time' in result else 'N/A'
    print(f"{model:<15} {status:<10} {sse:<15} {elapsed:<10}")

print("="*70)

# Overall result
if errors > 0:
    print("\n[ERROR] Some tests encountered errors!")
elif failed > 0:
    print("\n[FAIL] Some tests failed!")
else:
    print("\n[SUCCESS] All tests passed!")

print("\nNote: These tests validate that:")
print("  1. Models can be created successfully")
print("  2. Data simulation works correctly")
print("  3. Parameter estimation converges")
print("  4. Estimated parameters are close to true values")