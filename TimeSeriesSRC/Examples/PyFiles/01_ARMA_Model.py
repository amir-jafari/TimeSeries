# %% [markdown]
# # ARMA Model — System Identification Walkthrough
#
# This script demonstrates the four-step system identification process for a
# univariate time series using the **ARMA** model class.
#
# | Step | Task                | Tool                                 |
# |------|---------------------|--------------------------------------|
# | 1    | Choose a model class | Domain knowledge                    |
# | 2    | Select model order  | `uniAnal` — ACF, PACF, GPAC         |
# | 3    | Estimate parameters | `estimate`, `pmoddisp`               |
# | 4    | Validate the model  | `uniAnal` on residuals, `uniChi`     |
#
# **Dataset:** Box-Jenkins Series A — Chemical Concentration (197 observations)

# %% [markdown]
# The ARMA(nd, nc) model is:
#
#   D(q) y(t) = C(q) e(t)
#
# where
#   D(q) = 1 + d1*q^-1 + ... + dnd*q^-nd
#   C(q) = 1 + c1*q^-1 + ... + cnc*q^-nc
#
# The noise is modelled by the transfer function H(q) = C(q)/D(q).

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# Locate the TimeSeries repo root relative to this script's location.
# Both PyFiles/ and NoteBooks/ are one level below Examples/, which is one
# level below TimeSeriesSRC/, which is one level below the repo root.
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
# ## Series A — Chemical Concentration
#
# Box-Jenkins Series A is a univariate time series of 197 concentration
# measurements from a chemical process.  There is no observed external input,
# making the ARMA class the natural starting point.  The series mean is removed
# before modelling.

# %%
data_path = os.path.join(timeseries_root, 'TimeSeriesSRC', 'TestData',
                          'Series_A_Chemical_Concentration.csv')
df = pd.read_csv(data_path)
y = np.array(df['Concentration'])

# %% [markdown]
# ## Step 1 — Choose a Model Class
#
# With no external input, the general prediction model reduces to a pure noise
# model:
#
#   y(t) = H(q) e(t) = C(q)/D(q) e(t)
#
# This is the ARMA(nd, nc) model.  The orders are determined in Step 2.

# %%
y = y - np.mean(y)
N = y.size
print(f'Loaded chemical process data: N={N} samples')
print(f'y: mean={y.mean():.2e}, std={y.std():.3f}')

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(y)
ax.set_title('Process sequence  y(t) — Concentration')
ax.set_xlabel('Sample')
ax.set_ylabel('Concentration')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 — Select Model Order
#
# `uniAnal` computes the ACF, PACF, and Generalized Partial Autocorrelation
# (GPAC) of the series.
#
# - **ACF** — exponential or oscillatory decay suggests AR behaviour; a sharp
#   cutoff at lag nc suggests a pure MA model.
# - **PACF** — cuts off sharply after lag nd for a pure AR model; decays slowly
#   for a pure MA model.
# - **GPAC** — the primary tool for ARMA order identification.  A column of
#   approximately constant (near-zero) entries identifies the AR order nd; the
#   row at which the constant pattern begins identifies the MA order nc.

# %%
acf, pacf, gpac = uniAnal(y, na=20, nump=10, nrg=6, ncg=6)

# %% [markdown]
# ### GPAC Interpretation
#
# In the GPAC table, look for a column that becomes approximately constant
# (values near zero from some row onward):
#
# - The column index is the candidate AR order nd.
# - The row where the constant pattern begins is the candidate MA order nc.
#
# For Series A, column 1 of the GPAC becomes approximately constant starting
# at row 1, indicating an ARMA(1, 1) model (nd=1, nc=1).

# %% [markdown]
# ## Step 3 — Estimate Parameters
#
# We fit an ARMA(1, 1) model using `estimate`, which minimises the sum of
# squared one-step prediction errors via the Levenberg-Marquardt algorithm.
#
# `pmoddisp` prints the estimated parameters with ±2σ confidence intervals and
# a confidence-interval plot.  `pmodpzplot` shows the pole-zero map for the
# noise transfer function H(q) = C(q)/D(q), confirming stability and
# invertibility.

# %%
pmod = pmodel('arma', nc=[1], nd=[1], diff=[0], per=[])
pmod, trec, stat = estimate(pmod, y)

# %%
pmoddisp(pmod, stat)
pmodpzplot(pmod)
plt.show()

# %% [markdown]
# ## Step 4 — Validate the Model
#
# A well-fitted ARMA model should leave white residuals.  We check this three
# ways:
#
# 1. **Theoretical ACF** — `partoacf_pmod` computes the theoretical
#    autocovariance function of the fitted model using the Yule-Walker method.
#    Overlaying it on the experimental ACF from Step 2 confirms that the model
#    captures the correlation structure of the data.
# 2. **Residual ACF** — plot the ACF, PACF, and GPAC of the residuals with
#    `uniAnal`.  All values should fall within the 95% confidence bounds.
# 3. **Statistical test** — `uniChi` performs the portmanteau chi-square test
#    on the residuals.  A p-value > 0.05 indicates an adequate model.

# %% [markdown]
# ### Check 1 — Theoretical vs Experimental ACF
#
# `partoacf_pmod` uses the Yule-Walker equations to compute the exact
# autocovariance function implied by the fitted ARMA polynomials and the noise
# variance.  Close agreement with the experimental ACF from Step 2 confirms
# that the model has captured the dominant correlation structure.

# %%
# Noise variance estimate from one-step prediction errors
var_e, _ = pmodmse(pmod, y)

# Theoretical autocovariance function from the fitted ARMA(1,1) model
lagmax = 21
acf_theory, _, _ = partoacf_pmod(pmod, var_e, lagmax)

# Experimental ACF from Step 2: uniAnal returned acf of shape (1, 2*na+1)
# with na=20, so index 20 is lag 0; take the positive-lag half
acf_exp = acf.squeeze()[20:20 + lagmax]

# Normalize both to lag-0 = 1 for a shape comparison
acf_theory_norm = acf_theory / acf_theory[0]
acf_exp_norm    = acf_exp    / acf_exp[0]

lags = np.arange(lagmax)

fig, ax = plt.subplots(figsize=(10, 4))
ax.stem(lags, acf_exp_norm, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental')
ax.plot(lags, acf_theory_norm, 'r-o', markersize=5, linewidth=1.5,
        label='Theoretical (ARMA fit)')
ax.set_title('Autocorrelation Function: Experimental vs Theoretical')
ax.set_xlabel('Lag')
ax.set_ylabel('Normalized autocovariance')
ax.legend()
plt.tight_layout()
plt.show()

# %%
e = y - pmod.predict(y)
print(f'Residual std: {e.std():.4f}')
acf_e, pacf_e, gpac_e = uniAnal(e, na=20, nump=10, nrg=6, ncg=6)

# %%
passed, q_arma, n_arma, pval = uniChi(pmod, y)

print(f'\nChi-square test on residuals:  Q={q_arma:.2f},  df={n_arma},  '
      f'pass={bool(passed)},  pval={pval:.3f}')
print('(pass=True means residuals are consistent with white noise at 95% confidence)')

# %% [markdown]
# ## Conclusion
#
# The residual ACF is within the 95% confidence bounds (approximating an
# impulse function), the first row of the GPAC is approximately zero
# (indicating white noise), and the chi-square test is passed (p-value > 0.05),
# therefore the ARMA(1, 1) model is validated.
#
# The fitted parameter table from `pmoddisp` shows the estimated values with
# ±2σ confidence intervals, and none of the intervals includes zero.  The
# pole-zero map from `pmodpzplot` confirms stability and invertibility — all
# poles and zeros should lie inside the unit circle (for stability/invertibility
# in the z^{-1} convention used here).  There is no approximate pole/zero
# cancellation, which would indicate that the model orders could be reduced.
#
# If the residuals show remaining structure, return to Step 2 and explore
# higher-order structures, or use `selpmod` for an automated grid search over
# ARMA orders.

# %% [markdown]
# ## Automated Model Selection with `selpmod`
#
# `selpmod` performs a grid search over a set of candidate ARMA structures,
# estimating each one and ranking by AIC and BIC.  We search over
# nd in {1, 2, 3} and nc in {0, 1, 2, 3} — 12 combinations — to verify that
# the ARMA(1, 1) structure identified in Step 2 is indeed optimal.

# %%
arma_spec = {
    'models': [{
        'type': 'arma',
        'nc': [0, 1, 2, 3],
        'nd': [1, 2, 3],
        'diff': [0]
    }]
}
result  = selpmod(arma_spec, y)
aicmod  = result['arma']['aicmod']
bicmod  = result['arma']['bicmod']
aicstat = result['arma']['aicstat']
bicstat = result['arma']['bicstat']

# %%
print(f'Best AIC model: ARMA(nd={int(aicmod.nd[0])}, nc={int(aicmod.nc[0])})')
print(f'Best BIC model: ARMA(nd={int(bicmod.nd[0])}, nc={int(bicmod.nc[0])})\n')

print('=== Best AIC model ===')
pmoddisp(aicmod, aicstat)
pmodpzplot(aicmod)
plt.show()

print('=== Best BIC model ===')
pmoddisp(bicmod, bicstat)
pmodpzplot(bicmod)
plt.show()

# %% [markdown]
# Notice that the best AIC model may increase nc and nd, but pole/zero
# cancellations in the pole-zero plot suggest the effective order reverts to
# the original model.  The best BIC model is the same as the ARMA(1, 1)
# selected from the preliminary analysis.
