# %% [markdown]
# # ARIMA Model — System Identification Walkthrough
#
# This script demonstrates the four-step system identification process for a
# **non-stationary** univariate time series using the **ARIMA** model class.
#
# | Step | Task                     | Tool                                            |
# |------|--------------------------|-------------------------------------------------|
# | 1    | Choose a model class     | Raw ACF; differencing to achieve stationarity   |
# | 2    | Select model order       | `uniAnal` on nabla y — ACF, PACF, GPAC          |
# | 3    | Estimate parameters      | `estimate`, `pmoddisp`                          |
# | 4    | Validate the model       | `partoacf_pmod`, `uniAnal` on residuals, `uniChi` |
#
# **Dataset:** Box-Jenkins Series C — Chemical Temperature (226 observations)

# %% [markdown]
# ## ARIMA Model Structure
#
# The ARIMA(nd, d, nc) model is:
#
#   D(q) * nabla^d * y(t) = C(q) * e(t)
#
# where nabla = 1 - q^-1 is the backward difference operator.  Differencing d
# times converts a non-stationary (integrated) series into a stationary one,
# after which a standard ARMA(nd, nc) model is fitted to nabla^d y(t).
# The noise transfer function is H(q) = C(q)/D(q).

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
timeseries_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
if timeseries_root not in sys.path:
    sys.path.insert(0, timeseries_root)

from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.estimate import estimate
from TimeSeriesSRC.Model.selpmod import func_selpmod as selpmod
from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal as uniAnal
from TimeSeriesSRC.basefunctions.uniChi import func_uniChi as uniChi
from TimeSeriesSRC.basefunctions.partoacf import func_partoacf_pmod as partoacf_pmod
from TimeSeriesSRC.Model.pmodmse import func_pmodmse as pmodmse
from TimeSeriesSRC.Model.pmoddisp import func_pmoddisp as pmoddisp
from TimeSeriesSRC.Model.pmoddisp import func_pmodpzplot as pmodpzplot

np.random.seed(42)
print('Setup complete.')

# %% [markdown]
# ## Series C — Chemical Temperature
#
# Box-Jenkins Series C is a univariate time series of 226 temperature
# measurements from a chemical process.  There is no observed external input.
# The raw series exhibits a gradual drift, suggesting non-stationarity that
# will require first differencing (d=1) before an ARMA model can be fitted.
# The series mean is removed for numerical centering.

# %%
data_path = os.path.join(timeseries_root, 'TimeSeriesSRC', 'TestData',
                         'Series_C_Chemical_Temperature.csv')
df = pd.read_csv(data_path)
y = np.array(df['Temperature'])
y = y - np.mean(y)
N = y.size
print(f'Loaded chemical temperature data: N={N} samples')
print(f'y: mean={y.mean():.2e}, std={y.std():.3f}')

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(y)
ax.set_title('Process sequence  y(t) — Temperature (mean removed)')
ax.set_xlabel('Sample')
ax.set_ylabel('Temperature')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1 — Choose a Model Class
#
# With no external input, the general prediction model reduces to a pure noise
# model.  Before choosing ARMA or ARIMA, we must determine whether the series
# is **stationary**.
#
# A stationary series has a constant mean and variance; its ACF decays quickly
# to zero.  A non-stationary (integrated) series has a slowly decaying or
# non-decaying ACF.  We apply `uniAnal` to the raw (mean-removed) series to
# check.

# %%
acf_raw, pacf_raw, gpac_raw = uniAnal(y, na=20, nump=10, nrg=6, ncg=6)

# %% [markdown]
# ### Stationarity Check
#
# A slowly decaying ACF — where values remain large across many lags — is the
# hallmark of a non-stationary series.  First differencing (d=1) transforms
# y(t) to nabla y(t) = y(t) - y(t-1), which removes a stochastic linear trend
# (random-walk behaviour) and should yield a stationary series whose ACF decays
# quickly.  We compute and plot nabla y to verify.

# %%
dy = np.diff(y)   # nabla y(t) = y(t) - y(t-1)
Nd = dy.size
print(f'Differenced series: N={Nd} samples')
print(f'∇y: mean={dy.mean():.2e}, std={dy.std():.3f}')

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(y)
axes[0].set_title('Original series  y(t) — Temperature (mean removed)')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Temperature')

axes[1].plot(dy)
axes[1].set_title('First-differenced series  ∇y(t)')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('∇y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 — Select Model Order
#
# We apply `uniAnal` to the differenced series nabla y(t) to identify the ARMA
# orders nd and nc.  Together with d=1, these specify the ARIMA(nd, 1, nc)
# model.
#
# - **ACF** of nabla y — should now decay quickly, confirming stationarity.
# - **PACF** — cuts off sharply after lag nd for a pure AR component.
# - **GPAC** — a column of approximately constant (near-zero) entries
#   identifies nd; the row where the constant pattern begins identifies nc.

# %%
acf_dy, pacf_dy, gpac_dy = uniAnal(dy, na=20, nump=10, nrg=6, ncg=6)

# %% [markdown]
# ### GPAC Interpretation
#
# In the GPAC table for nabla y:
#
# - The column index of the first approximately constant column is the
#   candidate AR order nd=1.
# - The row where the constant pattern begins is the candidate MA order nc=0.
#   Also, after the first column, the remainder of row 0 is close to zero.
#
# Use these to form the ARIMA(1, 1, 0) model estimated in Step 3.

# %% [markdown]
# ## Step 3 — Estimate Parameters
#
# The ARIMA model is built with `pmodel` using `diff=[1]`.  The `estimate`
# function receives the **original** (undifferenced) series y — it
# pre-differences internally before running the Levenberg-Marquardt optimizer,
# so the parameters are estimated by minimising the mean-squared prediction
# error on nabla y.
#
# For **validation** (residuals, ACF comparison, chi-square test), `predict`
# and `uniChi` must receive the **differenced** series nabla y (dy), because
# the fitted ARMA model operates on nabla y.
#
# `pmoddisp` prints the estimated parameters with +-2sigma confidence intervals.
# `pmodpzplot` shows the pole-zero map of the noise transfer function
# H(q) = C(q)/D(q), confirming that the ARMA part is stable and invertible.

# %%
pmod = pmodel('arma', nc=[0], nd=[1], diff=[1], per=[])
pmod, trec, stat = estimate(pmod, y)

# %%
pmoddisp(pmod, stat)
pmodpzplot(pmod)
plt.show()

# %% [markdown]
# ## Step 4 — Validate the Model
#
# A well-fitted ARIMA model should leave white residuals.  We check this three
# ways:
#
# 1. **Theoretical ACF** — `partoacf_pmod` computes the exact autocovariance
#    function of the fitted ARMA part using the Yule-Walker method.  We compare
#    it against the experimental ACF of nabla y from Step 2.
# 2. **Residual ACF** — plot the ACF, PACF, and GPAC of the residuals with
#    `uniAnal`.  All values should fall within the 95% confidence bounds.
# 3. **Statistical test** — `uniChi` performs the portmanteau chi-square test
#    on the residuals.  A p-value > 0.05 indicates an adequate model.

# %% [markdown]
# ### Check 1 — Theoretical vs Experimental ACF
#
# `partoacf_pmod` uses the fitted ARMA polynomials D(q) and C(q) together with
# the noise variance to compute the theoretical autocovariance function of the
# stationary part of the model.  Close agreement with the experimental ACF of
# nabla y confirms that the model has captured the correlation structure of the
# differenced series.

# %%
# Noise variance estimate from one-step prediction errors on the differenced series
var_e, _ = pmodmse(pmod, dy)

# Theoretical autocovariance function from the fitted ARMA part
lagmax = 21
acf_theory, _, _ = partoacf_pmod(pmod, var_e, lagmax)

# Experimental ACF of nabla y from Step 2 (na=20 -> center index is 20)
acf_exp = acf_dy.squeeze()[20:20 + lagmax]

# Normalize both to lag-0 = 1 for a shape comparison
acf_theory_norm = acf_theory / acf_theory[0]
acf_exp_norm    = acf_exp    / acf_exp[0]

lags = np.arange(lagmax)
fig, ax = plt.subplots(figsize=(10, 4))
ax.stem(lags, acf_exp_norm, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental (∇y)')
ax.plot(lags, acf_theory_norm, 'r-o', markersize=5, linewidth=1.5,
        label='Theoretical (ARIMA fit)')
ax.set_title('ACF of ∇y: Experimental vs Theoretical')
ax.set_xlabel('Lag')
ax.set_ylabel('Normalized autocovariance')
ax.legend()
plt.tight_layout()
plt.show()

# %%
e = dy - pmod.predict(dy)
print(f'Residual std: {e.std():.4f}')
acf_e, pacf_e, gpac_e = uniAnal(e, na=20, nump=10, nrg=6, ncg=6)

# %%
passed, q_arima, n_arima, pval = uniChi(pmod, dy)

print(f'\nChi-square test on ARIMA model:  Q={q_arima:.2f},  df={n_arima},  '
      f'pass={bool(passed)},  pval={pval:.3f}')
print('(pass=True means residuals are consistent with white noise at 95% confidence)')

# %% [markdown]
# ## Conclusion
#
# The chi-square test passes (p-value > 0.05), the residual ACF and PACF fall
# within the 95% confidence bounds, and the theoretical and experimental ACFs
# of nabla y are broadly consistent — confirming that the **ARIMA(1, 1, 0)**
# model is validated for Box-Jenkins Series C.
#
# The fitted pole-zero map from `pmodpzplot` confirms that the AR polynomial
# is stable (pole inside the unit circle).
#
# If the residuals show remaining structure, return to Step 2 and adjust nd or
# nc, or use `selpmod` in the next section for an automated grid search.  The
# BIC-optimal model identified by `selpmod` is typically ARIMA(1, 1, 1), which
# achieves an even lower Q statistic.

# %% [markdown]
# ## Automated Model Selection with `selpmod`
#
# `selpmod` searches a grid of ARMA orders applied to the differenced series
# (fixed d=1) and selects the best ARIMA structure by AIC and BIC.  We search
# over nd in {1, 2, 3} and nc in {0, 1, 2, 3} — 12 combinations — to verify
# the manually chosen structure.

# %%
arima_spec = {
    'models': [{
        'type': 'arma',
        'nc': [0, 1, 2, 3],
        'nd': [1, 2, 3],
        'diff': [1]
    }]
}
result  = selpmod(arima_spec, y)
aicmod  = result['arma']['aicmod']
bicmod  = result['arma']['bicmod']
aicstat = result['arma']['aicstat']
bicstat = result['arma']['bicstat']

# %%
print(f'Best AIC model: ARIMA(nd={int(aicmod.nd[0])}, d=1, nc={int(aicmod.nc[0])})')
print(f'Best BIC model: ARIMA(nd={int(bicmod.nd[0])}, d=1, nc={int(bicmod.nc[0])})')

print('\n=== Best AIC model ===')
pmoddisp(aicmod, aicstat)
pmodpzplot(aicmod)
plt.show()

print('\n=== Best BIC model ===')
pmoddisp(bicmod, bicstat)
pmodpzplot(bicmod)
plt.show()

# %% [markdown]
# Note that the optimal AIC model added additional c and d parameters.  The
# chi-square test was passed, but the additional parameters produced approximate
# pole/zero cancellations.  The optimal BIC model is the same one that was
# developed from the initial analysis of the GPAC.
