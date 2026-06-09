# %% [markdown]
# # BJTF Model — System Identification Walkthrough
#
# This script demonstrates the four-step system identification process for a
# **two-variable, input-output** system using the **Box-Jenkins Transfer
# Function (BJTF)** model class.
#
# | Step | Task                     | Tool                                                       |
# |------|--------------------------|------------------------------------------------------------|
# | 1    | Choose a model class     | Identify input-output structure; zero-mean both signals    |
# | 2    | Select model order       | `uniAnal` on u -> ARMA model; `multiAnal` -> impulse resp  |
# | 3    | Estimate parameters      | `estimate`, `pmoddisp`, `pmodpzplot`                       |
# | 4    | Validate the model       | `multiChi` — whiteness test (Q) and cross-correlation (S)  |
#
# **Dataset:** Box-Jenkins Series J — Gas Furnace (296 observations)
# Input u: gas flow rate (ft3/min, zero-meaned)
# Output y: percent CO2 in exit gas (zero-meaned)

# %% [markdown]
# ## BJTF Model Structure
#
# The BJTF model separates the input-output dynamics (G) from the noise (H):
#
#   y(t) = G(q^-1) u(t-k) + H(q^-1) e(t)
#
# where
#   G(q^-1) = B(q^-1) / F(q^-1)   (b0 + b1*q^-1 + ... ) / (1 + f1*q^-1 + ...)
#   H(q^-1) = C(q^-1) / D(q^-1)   (1 + c1*q^-1 + ...) / (1 + d1*q^-1 + ...)
#
# k is a pure time delay and e(t) is white noise with variance sigma^2.
#
# BJTF is an output-error model: G and H are estimated independently, giving
# the most flexible representation when the disturbance model differs from the
# input dynamics.  Compare to ARMAX (equation-error), where G and H share the
# same denominator polynomial A.
#
# In PredictMod notation:
#   pmodel('bjtf', nb=[nb], nf=[nf], nc=[nc], nd=[nd], delay=[k], diff=[0], per=[])

# %%
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
timeseries_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
if timeseries_root not in sys.path:
    sys.path.insert(0, timeseries_root)

from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.estimate import estimate
from TimeSeriesSRC.Model.selpmod import func_selpmod as selpmod
from TimeSeriesSRC.Model.pmodaic import func_pmodaic as pmodaic
from TimeSeriesSRC.Model.pmodbic import func_pmodbic as pmodbic
from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal as uniAnal
from TimeSeriesSRC.basefunctions.multiAnal import func_multiAnal as multiAnal
from TimeSeriesSRC.basefunctions.uniChi import func_uniChi as uniChi
from TimeSeriesSRC.basefunctions.multiChi import func_multiChi as multiChi
from TimeSeriesSRC.basefunctions.partoacf import func_partoacf_pmod as partoacf_pmod
from TimeSeriesSRC.Model.pmodmse import func_pmodmse as pmodmse
from TimeSeriesSRC.Model.pmoddisp import func_pmoddisp as pmoddisp
from TimeSeriesSRC.Model.pmoddisp import func_pmodpzplot as pmodpzplot

np.random.seed(42)
print('Setup complete.')

# %%
data_path = os.path.join(timeseries_root, 'TimeSeriesSRC', 'TestData',
                         'Series_J_Gas_Furnace.csv')
df    = pd.read_csv(data_path)
u_raw = np.array(df['InputGasRate'], dtype=float)
y_raw = np.array(df['CO2'],          dtype=float)

u = u_raw - u_raw.mean()
y = y_raw - y_raw.mean()
N = y.size

print(f'Loaded gas furnace data: N={N} samples')
print(f'u: mean={u.mean():.2e}, std={u.std():.3f}')
print(f'y: mean={y.mean():.2e}, std={y.std():.3f}')

fig, axes = plt.subplots(2, 1, figsize=(12, 5))
axes[0].plot(u_raw)
axes[0].set_title('Input  u(t) — Gas Flow Rate')
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Gas rate (ft³/min)')
axes[1].plot(y_raw)
axes[1].set_title('Output  y(t) — Percent CO₂ in Exit Gas')
axes[1].set_xlabel('Sample')
axes[1].set_ylabel('% CO₂')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1 — Choose a Model Class
#
# The gas furnace experiment measures a cause-and-effect relationship: the gas
# flow rate u(t) is manipulated and the CO2 concentration y(t) responds after
# a delay.  This input-output structure points to a model with both a transfer
# function G from u to y and a separate noise model H.
#
# Three candidate model classes are compared in this notebook:
#
# | Model | Structure              | Key property                                     |
# |-------|------------------------|--------------------------------------------------|
# | BJTF  | y = B/F*u + C/D*e      | Output-error: G and H estimated independently    |
# | ARMAX | A*y = B*u + C*e        | Equation-error: G and H share the same poles (A) |
# | ARX   | A*y = B*u + e          | Simplest equation-error: H = 1/A, no MA term     |
#
# We work through the full four-step process for BJTF and use `selpmod` to
# compare ARMAX and ARX at the end.

# %% [markdown]
# ## Step 2 — Select Model Order
#
# ### Step 2a — Univariate Analysis of the Input u
#
# Before fitting the BJTF model, we model u(t) as an ARMA process.  This
# serves two purposes:
# 1. **Prewhitening** — `multiAnal` uses the ARMA model to prewhiten u before
#    computing the impulse response and G/H GPAC arrays.
# 2. **Preliminary structure check** — the ACF, PACF, and GPAC of u reveal its
#    autocorrelation structure independently of y.

# %%
uacf, upacf, ugpac = uniAnal(u, 20, 10)
print('GPAC for u:')
print(np.round(ugpac, 3))

# %% [markdown]
# ### ACF/GPAC Interpretation for u
#
# Key features in the correlation structure of u:
#
# - The PACF cuts off sharply after lag 3, with negligible values beyond.
# - The GPAC shows an approximately constant value in column 3 starting from
#   row 1, with near-zero entries in columns 1 and 2.
#
# This is the signature of a pure AR(3) process — ARMA with nc=0 and nd=3.

# %% [markdown]
# ### Step 2b — Fit and Validate the ARMA(0, 3) Model for u
#
# We fit the AR(3) model and inspect parameter confidence intervals, the
# residual ACF, and the chi-square statistic.

# %%
pmod_u = pmodel('arma', nc=[0], nd=[3], diff=[0], per=[])
pmod_u, trec_u, stat_u = estimate(pmod_u, u)
pmoddisp(pmod_u, stat_u)
pmodpzplot(pmod_u)
plt.show()

# %% [markdown]
# All three AR parameters should have narrow confidence intervals that exclude
# zero — no order reduction indicated.  There are no zeros (nc=0), so the
# pole-zero plot shows only poles.
#
# Validation checks:
# 1. Residual ACF/PACF/GPAC — residuals should resemble white noise.
# 2. Portmanteau test — chi-square test on the residual ACF.
# 3. Theoretical vs experimental ACF — the fitted model should reproduce the
#    autocorrelation structure of u.

# %%
e_u = u - pmod_u.predict(u)
uracf, urpacf, urgpac = uniAnal(e_u, 20, 10)

passed_u, q_u, n_u, pval_u = uniChi(pmod_u, u)
print(f'Portmanteau test:  Q = {q_u:.2f},  df = {n_u},  p-value = {pval_u:.3f}')
print(f'Result: {"PASS" if passed_u else "FAIL"}')

var_e_u, _ = pmodmse(pmod_u, u)
lagmax = 21
acf_theory_u, _, _ = partoacf_pmod(pmod_u, var_e_u, lagmax)
uacf_s    = uacf.squeeze()
mid       = len(uacf_s) // 2
acf_exp_u = uacf_s[mid:mid + lagmax]

lags = np.arange(lagmax)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(lags, acf_exp_u    / acf_exp_u[0],    'C0o-', label='Experimental (u)')
ax.plot(lags, acf_theory_u / acf_theory_u[0], 'r-o',  markersize=4,
        label='Theoretical (ARMA fit)')
ax.set_title('ACF of u: Experimental vs Theoretical')
ax.set_xlabel('Lag')
ax.set_ylabel('Normalised autocovariance')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Step 2c — Multivariate Analysis (u -> y)
#
# `multiAnal` prewhitens u using the fitted ARMA model, then computes:
#
# - The estimated impulse response g_hat(tau) from u to y.
# - The residual autocorrelation rv(tau) of the disturbance v = y - g_hat * u.
# - The G GPAC — identifies orders nb (B numerator) and nf (F denominator).
# - The H GPAC — identifies orders nc (C numerator) and nd (D denominator).
#
# The lag at which g_hat(tau) first becomes significantly non-zero gives the
# pure delay k.

# %%
g_ir, rv, g_gpac, h_gpac = multiAnal(u, y, 8, 6, 8, 6)
print('Impulse response (first 10 lags):')
print(np.round(g_ir[:10], 4))
print('\nG GPAC:')
print(np.round(g_gpac, 3))
print('\nH GPAC:')
print(np.round(h_gpac, 3))

# %% [markdown]
# ### Interpretation and Model Order Selection
#
# **Impulse response** g_hat(tau):
# - Near zero at tau=0,1,2; becomes non-zero at tau=3 -> delay k=3.
#
# **G GPAC** (transfer function from u to y):
# - Large values in the first three rows reflect the pure delay.
# - After accounting for the delay, an approximately constant value appears in
#   column 2 — indicating nb=2 zeros and nf=2 poles in G.
#
# **H GPAC** (noise transfer function):
# - An approximately constant value in column 2 starting from row 1 ->
#   nc=0, nd=2.
#
# **Selected model:**
#   pmodel('bjtf', nb=[2], nf=[2], nc=[0], nd=[2], delay=[3])

# %% [markdown]
# ## Step 3 — Estimate Parameters
#
# We estimate the BJTF model using the Levenberg-Marquardt algorithm.  The
# output y must be the first argument to `estimate` and the input u the second.
#
# The selected model has nb+nf+nc+nd = 2+2+0+2 = 6 free parameters.

# %%
pmod_bjtf = pmodel('bjtf', nb=[2], nf=[2], nc=[0], nd=[2],
                   delay=[3], diff=[0], per=[])
pmod_bjtf.estimParams.epochs = 50
pmod_bjtf.estimParams.goal   = 0.01
pmod_bjtf, trec_bjtf, stat_bjtf = estimate(pmod_bjtf, y, u)

# %%
pmoddisp(pmod_bjtf, stat_bjtf)
pmodpzplot(pmod_bjtf)
plt.show()

# %% [markdown]
# The parameter table shows all estimates with 95% confidence intervals.  Note
# that f2 may have a wide interval that includes zero — suggesting nf could be
# reduced from 2 to 1.  The `selpmod` search in Step 5 will check this.
#
# The pole-zero map shows G (B zeros, F poles) and H (C zeros, D poles)
# separately.  No pole-zero cancellations are visible.

# %% [markdown]
# ## Step 4 — Validate the Model
#
# For BJTF models we apply three complementary checks:
#
# 1. **Whiteness and cross-correlation tests** — `multiChi` applies two
#    chi-square tests on the one-step prediction errors e(t) = y(t)-y_hat(t|t-1):
#    - Q statistic: residuals are white noise (ACF test).  Pass: p_Q > 0.05.
#    - S statistic: residuals are uncorrelated with the prewhitened input.
#                   Pass: p_S > 0.05.
# 2. **Theoretical vs experimental impulse response** — `partoacf_pmod`
#    computes the theoretical impulse response of G, which should match the
#    experimental impulse response g_hat(tau) from `multiAnal` in Step 2.
# 3. **Theoretical vs experimental H autocorrelation** — `partoacf_pmod` also
#    computes the theoretical ACF of v(t) = H(q^-1)e(t), which should match
#    the residual autocorrelation rv(tau) from `multiAnal`.

# %%
pass_arr, q_val, pvalq, s_val, pvals, nq, ns = multiChi(pmod_bjtf, y, u)
print(f'Q: Q = {q_val:.2f},  df = {nq},  p-value = {pvalq:.3f},  pass = {bool(pass_arr[0])}')
print(f'S: S = {s_val:.2f},  df = {ns},  p-value = {pvals:.3f},  pass = {bool(pass_arr[1])}')

y_pred_bjtf = pmod_bjtf.predict(y, u)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y,           'C0o', ms=3, label='y(t) measured')
ax.plot(y_pred_bjtf, 'rx',  ms=3, label='ŷ(t|t−1) predicted')
ax.set_title('BJTF One-Step-Ahead Prediction')
ax.set_xlabel('Sample')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Both tests should pass (p-values > 0.05): residuals consistent with white
# noise and uncorrelated with the prewhitened input.  The prediction plot
# shows the model tracking the measured output closely.

# %% [markdown]
# ### Check 2 — Theoretical vs Experimental Impulse Response
#
# The theoretical impulse response of G(q^-1) = B(q^-1)/F(q^-1)*q^-k is
# computed from the estimated coefficients using `partoacf_pmod` and compared
# with the experimental impulse response g_hat(tau) returned by `multiAnal`
# in Step 2.  The pure delay k=3 should appear as three leading zeros in both.

# %%
# Noise variance from one-step prediction errors
e_bjtf = y - pmod_bjtf.predict(y, u)
var_e_bjtf = np.var(e_bjtf)
print(f'Noise variance estimate: {var_e_bjtf:.6f}')

# Theoretical impulse response (G) and H ACF from the fitted model.
lagmax_g = len(g_ir)                        # match experimental g_ir length
lagmax_h = rv.squeeze().shape[0] // 2 + 1  # positive lags of rv
lagmax_c = max(lagmax_g, lagmax_h)
acf_theory_H, _, g_ir_theory = partoacf_pmod(pmod_bjtf, var_e_bjtf, lagmax_c)

# Plot 1: G impulse response
lags_g = np.arange(lagmax_g)
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags_g, g_ir, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental g_hat(tau) (multiAnal)')
ax.plot(lags_g, g_ir_theory[:lagmax_g], 'r-o', markersize=5, linewidth=1.5,
        label='Theoretical (BJTF fit)')
ax.set_title('G Impulse Response: Experimental vs Theoretical')
ax.set_xlabel('Lag tau')
ax.set_ylabel('g(tau)')
ax.legend()
plt.tight_layout()
plt.show()

# Plot 2: H autocorrelation — rv vs theoretical
rv_s   = rv.squeeze()
mid_rv = len(rv_s) // 2
rv_pos = rv_s[mid_rv:mid_rv + lagmax_h]

acf_H_norm = acf_theory_H[:lagmax_h] / acf_theory_H[0]
rv_norm    = rv_pos / rv_pos[0]

lags_h = np.arange(lagmax_h)
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags_h, rv_norm, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental rv (multiAnal)')
ax.plot(lags_h, acf_H_norm, 'r-o', markersize=4, linewidth=1.5,
        label='Theoretical (H model)')
ax.set_title('H Autocorrelation: Experimental rv vs Theoretical')
ax.set_xlabel('Lag')
ax.set_ylabel('Normalised autocovariance')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5 — Automated Model Selection with `selpmod`
#
# `selpmod` searches a grid of BJTF structures and selects the best by AIC/BIC.
#
# | Parameter          | Values  |
# |--------------------|---------|
# | nb (B numerator)   | 1, 2    |
# | nc (C numerator)   | 0, 1, 2 |
# | nd (D denominator) | 0, 1, 2 |
# | nf (F denominator) | 0, 1, 2 |
# | Delay k            | 3 fixed |
#
# 2 x 3^3 = 54 total combinations.  The BIC-optimal model is expected to
# reduce nf from 2 to 1.

# %%
bjtf_spec = {
    'models': [{
        'type':  'bjtf',
        'nb':    [1, 2],
        'nc':    [0, 1, 2],
        'nd':    [0, 1, 2],
        'nf':    [0, 1, 2],
        'delay': [3],
        'diff':  [0]
    }]
}
result_bjtf = selpmod(bjtf_spec, y, u)
print('\nModel selection complete.')

# %%
pmod_bic = result_bjtf['bjtf']['bicmod']
stat_bic = result_bjtf['bjtf']['bicstat']

print('Best BJTF model by BIC:')
print(f'  b = {pmod_bic.b}')
print(f'  f = {pmod_bic.f}')
print(f'  c = {pmod_bic.c}')
print(f'  d = {pmod_bic.d}')
print(f'  delay = {pmod_bic.delay}')

pmoddisp(pmod_bic, stat_bic)
pmodpzplot(pmod_bic)
plt.show()

y_pred_bic = pmod_bic.predict(y, u)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y,          'C0o', ms=3, label='y(t) measured')
ax.plot(y_pred_bic, 'rx',  ms=3, label='best BJTF (BIC)')
ax.set_title('Best BJTF Model (BIC) — One-Step-Ahead Prediction')
ax.set_xlabel('Sample')
ax.legend()
plt.tight_layout()
plt.show()

pass2, q2, pvalq2, s2, pvals2, nq2, ns2 = multiChi(pmod_bic, y, u)
print(f'Q: Q={q2:.2f},  df={nq2},  p={pvalq2:.3f},  pass={bool(pass2[0])}')
print(f'S: S={s2:.2f},  df={ns2},  p={pvals2:.3f},  pass={bool(pass2[1])}')

aic_bjtf = pmodaic(pmod_bic, y, u)
bic_bjtf = pmodbic(pmod_bic, y, u)
print(f'\nBest BJTF (BIC)  —  AIC = {aic_bjtf:.4f},  BIC = {bic_bjtf:.4f}')

# %% [markdown]
# ## Comparison: ARMAX Model
#
# The ARMAX model is an equation-error model:
#
#   A(q^-1) y(t) = B(q^-1) u(t-k) + C(q^-1) e(t)
#
# Both transfer functions share the same denominator A:
#   G = B/A,  H = C/A
#
# ARMAX cannot independently model G and H dynamics.  The pole-zero plot often
# shows approximate cancellations in one transfer function — a known limitation
# of equation-error models (Ljung, 1999).

# %%
armax_spec = {
    'models': [{
        'type':  'armax',
        'na':    [1, 2, 3],
        'nb':    [1, 2, 3, 4],
        'nc':    [0, 1, 2],
        'delay': [3],
        'diff':  [0]
    }]
}
result_armax = selpmod(armax_spec, y, u)
print('\nModel selection complete.')

# %%
pmod_armax = result_armax['armax']['aicmod']
stat_armax = result_armax['armax']['aicstat']

print('Best ARMAX model by AIC:')
print(f'  a = {pmod_armax.a}')
print(f'  b = {pmod_armax.b}')
print(f'  c = {pmod_armax.c}')
print(f'  delay = {pmod_armax.delay}')

pmoddisp(pmod_armax, stat_armax)
pmodpzplot(pmod_armax)
plt.show()

y_pred_armax = pmod_armax.predict(y, u)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y,            'C0o', ms=3, label='y(t) measured')
ax.plot(y_pred_armax, 'rx',  ms=3, label='best ARMAX (AIC)')
ax.set_title('Best ARMAX Model (AIC) — One-Step-Ahead Prediction')
ax.set_xlabel('Sample')
ax.legend()
plt.tight_layout()
plt.show()

aic_armax = pmodaic(pmod_armax, y, u)
bic_armax = pmodbic(pmod_armax, y, u)
print(f'\nBest ARMAX (AIC)  —  AIC = {aic_armax:.4f},  BIC = {bic_armax:.4f}')

# %% [markdown]
# There are some parameters with error bars that include 0, but they don't
# correspond to the final parameter, so the model orders cannot be reduced from
# that alone.  Notice that there is an approximate pole/zero cancellation in
# the G transfer function.  Since both G and H have the same denominator in
# the ARMAX model, we don't have the flexibility to reduce the orders of B and A
# because the root of A is not cancelled in the H transfer function.  The
# approximate cancellation is a diagnostic sign that the shared-pole constraint
# is forcing the model to absorb noise-model dynamics into the input transfer
# function (Ljung, 1999, Ch. 4).

# %%
# Noise variance from one-step prediction errors
e_armax = y - pmod_armax.predict(y, u)
var_e_armax = np.var(e_armax)
print(f'Noise variance estimate: {var_e_armax:.6f}')

lagmax_g = len(g_ir)
lagmax_h = rv.squeeze().shape[0] // 2 + 1
lagmax_c = max(lagmax_g, lagmax_h)
acf_theory_H, _, g_ir_theory = partoacf_pmod(pmod_armax, var_e_armax, lagmax_c)

lags_g = np.arange(lagmax_g)
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags_g, g_ir, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental g_hat(tau) (multiAnal)')
ax.plot(lags_g, g_ir_theory[:lagmax_g], 'r-o', markersize=5, linewidth=1.5,
        label='Theoretical (ARMAX fit)')
ax.set_title('G Impulse Response: Experimental vs Theoretical (ARMAX)')
ax.set_xlabel('Lag tau')
ax.set_ylabel('g(tau)')
ax.legend()
plt.tight_layout()
plt.show()

rv_s   = rv.squeeze()
mid_rv = len(rv_s) // 2
rv_pos = rv_s[mid_rv:mid_rv + lagmax_h]
acf_H_norm = acf_theory_H[:lagmax_h] / acf_theory_H[0]
rv_norm    = rv_pos / rv_pos[0]

lags_h = np.arange(lagmax_h)
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags_h, rv_norm, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental rv (multiAnal)')
ax.plot(lags_h, acf_H_norm, 'r-o', markersize=4, linewidth=1.5,
        label='Theoretical (H model)')
ax.set_title('H Autocorrelation: Experimental rv vs Theoretical (ARMAX)')
ax.set_xlabel('Lag')
ax.set_ylabel('Normalised autocovariance')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The impulse response of G is accurate, but the ACF of the disturbance is much
# less accurate than with the BJTF model.  The ARMAX model has limited
# flexibility in representing H since it can only adjust the numerator.

# %% [markdown]
# ## Comparison: ARX Model
#
# The ARX model is the simplest equation-error model — no MA term:
#
#   A(q^-1) y(t) = B(q^-1) u(t-k) + e(t)
#
# The noise transfer function is H = 1/A, so the noise is forced to be a pure
# AR process.  ARX is the cheapest to estimate (linear least squares) but will
# generally have higher AIC/BIC than BJTF or ARMAX unless the disturbance is
# genuinely AR.
#
# In PredictMod notation: pmodel('arx', na=[na], nb=[nb], delay=[k])

# %%
arx_spec = {
    'models': [{
        'type':  'arx',
        'na':    [1, 2, 3],
        'nb':    [1, 2, 3],
        'delay': [3]
    }]
}
result_arx = selpmod(arx_spec, y, u)
print('\nModel selection complete.')

# %%
pmod_arx = result_arx['arx']['bicmod']
stat_arx = result_arx['arx']['bicstat']

print('Best ARX model by BIC:')
print(f'  a = {pmod_arx.a}')
print(f'  b = {pmod_arx.b}')
print(f'  delay = {pmod_arx.delay}')

pmoddisp(pmod_arx, stat_arx)
pmodpzplot(pmod_arx)
plt.show()

y_pred_arx = pmod_arx.predict(y, u)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y,           'C0o', ms=3, label='y(t) measured')
ax.plot(y_pred_arx,  'rx',  ms=3, label='best ARX (BIC)')
ax.set_title('Best ARX Model (BIC) — One-Step-Ahead Prediction')
ax.set_xlabel('Sample')
ax.legend()
plt.tight_layout()
plt.show()

aic_arx = pmodaic(pmod_arx, y, u)
bic_arx = pmodbic(pmod_arx, y, u)
print(f'\nBest ARX (BIC)  —  AIC = {aic_arx:.4f},  BIC = {bic_arx:.4f}')

# %%
# Noise variance from one-step prediction errors
e_arx = y - pmod_arx.predict(y, u)
var_e_arx = np.var(e_arx)
print(f'Noise variance estimate: {var_e_arx:.6f}')

lagmax_g = len(g_ir)
lagmax_h = rv.squeeze().shape[0] // 2 + 1
lagmax_c = max(lagmax_g, lagmax_h)
acf_theory_H, _, g_ir_theory = partoacf_pmod(pmod_arx, var_e_arx, lagmax_c)

lags_g = np.arange(lagmax_g)
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags_g, g_ir, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental g_hat(tau) (multiAnal)')
ax.plot(lags_g, g_ir_theory[:lagmax_g], 'r-o', markersize=5, linewidth=1.5,
        label='Theoretical (ARX fit)')
ax.set_title('G Impulse Response: Experimental vs Theoretical (ARX)')
ax.set_xlabel('Lag tau')
ax.set_ylabel('g(tau)')
ax.legend()
plt.tight_layout()
plt.show()

rv_s   = rv.squeeze()
mid_rv = len(rv_s) // 2
rv_pos = rv_s[mid_rv:mid_rv + lagmax_h]
acf_H_norm = acf_theory_H[:lagmax_h] / acf_theory_H[0]
rv_norm    = rv_pos / rv_pos[0]

lags_h = np.arange(lagmax_h)
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags_h, rv_norm, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental rv (multiAnal)')
ax.plot(lags_h, acf_H_norm, 'r-o', markersize=4, linewidth=1.5,
        label='Theoretical (H model)')
ax.set_title('H Autocorrelation: Experimental rv vs Theoretical (ARX)')
ax.set_xlabel('Lag')
ax.set_ylabel('Normalised autocovariance')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The impulse response for G is reasonably accurate, but the ACF of the
# disturbance is not accurate.  This is expected, since the ARX model has no
# flexibility in setting the H transfer function.

# %% [markdown]
# ## Conclusion
#
# The system identification process for Box-Jenkins Series J (gas furnace)
# confirmed the **BJTF model** as the most flexible and interpretable choice.
#
# **Summary of the four-step process:**
# 1. Model class: BJTF — separate G (input TF) and H (noise TF) polynomials.
# 2. Order selection: uniAnal on u -> AR(3) for the input; multiAnal ->
#    impulse response identified delay k=3; G GPAC identified nb=2, nf=2;
#    H GPAC identified nc=0, nd=2.
# 3. Estimation: Levenberg-Marquardt converged cleanly; f2 confidence interval
#    included zero.
# 4. Validation: Both multiChi tests (Q and S) passed.
# 5. Automated selection: selpmod with BIC confirmed the model and reduced nf
#    to 1.
#
# **Model class comparison:**
#
# | Model | G and H poles shared? | Flexibility | Typical AIC/BIC rank |
# |-------|-----------------------|-------------|----------------------|
# | BJTF  | No — independently    | Highest     | Best                 |
# | ARMAX | Yes, via A polynomial | Medium      | Comparable           |
# | ARX   | Yes, no MA term       | Lowest      | Highest              |
#
# BJTF provides the clearest physical picture: G captures how changes in gas
# flow rate propagate to CO2 output after a 3-sample delay, while H
# independently captures the stochastic disturbance dynamics.
#
# Reference: Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis:
# Forecasting and Control (Rev. ed.), Chapter 11.
