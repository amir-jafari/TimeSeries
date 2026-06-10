# %% [markdown]
# # BJTF Model — System Identification Walkthrough
#
# This script uses the **pip-installed** `timeseries-toolbox` package.
# Install it with:  pip install timeseries-toolbox
#
# **Dataset:** Box-Jenkins Series J — Gas Furnace (296 observations)
# Input u: gas flow rate.  Output y: percent CO2 in exit gas.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from importlib.resources import files

from TimeSeriesSRC.Model.model     import pmodel
from TimeSeriesSRC.Model.estimate  import estimate
from TimeSeriesSRC.Model.selpmod   import func_selpmod  as selpmod
from TimeSeriesSRC.Model.pmodaic   import func_pmodaic  as pmodaic
from TimeSeriesSRC.Model.pmodbic   import func_pmodbic  as pmodbic
from TimeSeriesSRC.Model.pmodmse   import func_pmodmse  as pmodmse
from TimeSeriesSRC.Model.pmoddisp  import func_pmoddisp   as pmoddisp
from TimeSeriesSRC.Model.pmoddisp  import func_pmodpzplot as pmodpzplot
from TimeSeriesSRC.basefunctions.uniAnal  import func_uniAnal       as uniAnal
from TimeSeriesSRC.basefunctions.multiAnal import func_multiAnal    as multiAnal
from TimeSeriesSRC.basefunctions.uniChi   import func_uniChi        as uniChi
from TimeSeriesSRC.basefunctions.multiChi import func_multiChi      as multiChi
from TimeSeriesSRC.basefunctions.partoacf import func_partoacf_pmod as partoacf_pmod

np.random.seed(42)
print('Setup complete.')

# %%
data_path = str(files('TimeSeriesSRC').joinpath('TestData/Series_J_Gas_Furnace.csv'))
df    = pd.read_csv(data_path)
u_raw = np.array(df['InputGasRate'], dtype=float)
y_raw = np.array(df['CO2'],          dtype=float)

u = u_raw - u_raw.mean()
y = y_raw - y_raw.mean()
N = y.size
print(f'Loaded gas furnace data: N={N} samples')

fig, axes = plt.subplots(2, 1, figsize=(12, 5))
axes[0].plot(u_raw); axes[0].set_title('Input  u(t) — Gas Flow Rate')
axes[1].plot(y_raw); axes[1].set_title('Output  y(t) — Percent CO₂')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 — Select Model Order
# ### Step 2a — Univariate analysis of input u

# %%
uacf, upacf, ugpac = uniAnal(u, 20, 10)
print('GPAC for u:')
print(np.round(ugpac, 3))

# %%
pmod_u = pmodel('arma', nc=[0], nd=[3], diff=[0], per=[])
pmod_u, trec_u, stat_u = estimate(pmod_u, u)
pmoddisp(pmod_u, stat_u)
pmodpzplot(pmod_u)
plt.show()

passed_u, q_u, n_u, pval_u = uniChi(pmod_u, u)
print(f'Portmanteau test:  Q={q_u:.2f},  df={n_u},  p-value={pval_u:.3f},  {"PASS" if passed_u else "FAIL"}')

# %% [markdown]
# ### Step 2b — Multivariate analysis (u -> y)

# %%
g_ir, rv, g_gpac, h_gpac = multiAnal(u, y, 8, 6, 8, 6)
print('Impulse response (first 10 lags):')
print(np.round(g_ir[:10], 4))

# %% [markdown]
# ## Step 3 — Estimate Parameters
# Selected model: BJTF with nb=2, nf=2, nc=0, nd=2, delay=3

# %%
pmod_bjtf = pmodel('bjtf', nb=[2], nf=[2], nc=[0], nd=[2], delay=[3], diff=[0], per=[])
pmod_bjtf.estimParams.epochs = 50
pmod_bjtf.estimParams.goal   = 0.01
pmod_bjtf, trec_bjtf, stat_bjtf = estimate(pmod_bjtf, y, u)

pmoddisp(pmod_bjtf, stat_bjtf)
pmodpzplot(pmod_bjtf)
plt.show()

# %% [markdown]
# ## Step 4 — Validate the Model

# %%
pass_arr, q_val, pvalq, s_val, pvals, nq, ns = multiChi(pmod_bjtf, y, u)
print(f'Q: Q={q_val:.2f},  df={nq},  p={pvalq:.3f},  pass={bool(pass_arr[0])}')
print(f'S: S={s_val:.2f},  df={ns},  p={pvals:.3f},  pass={bool(pass_arr[1])}')

y_pred = pmod_bjtf.predict(y, u)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y,      'C0o', ms=3, label='y(t) measured')
ax.plot(y_pred, 'rx',  ms=3, label='ŷ(t|t−1) predicted')
ax.set_title('BJTF One-Step-Ahead Prediction')
ax.set_xlabel('Sample')
ax.legend()
plt.tight_layout()
plt.show()

# %%
e_bjtf     = y - pmod_bjtf.predict(y, u)
var_e_bjtf = np.var(e_bjtf)
lagmax_g   = len(g_ir)
lagmax_h   = rv.squeeze().shape[0] // 2 + 1
lagmax_c   = max(lagmax_g, lagmax_h)
acf_theory_H, _, g_ir_theory = partoacf_pmod(pmod_bjtf, var_e_bjtf, lagmax_c)

lags_g = np.arange(lagmax_g)
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags_g, g_ir, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental g_hat(tau)')
ax.plot(lags_g, g_ir_theory[:lagmax_g], 'r-o', markersize=5, linewidth=1.5,
        label='Theoretical (BJTF fit)')
ax.set_title('G Impulse Response: Experimental vs Theoretical')
ax.set_xlabel('Lag tau')
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Automated Model Selection with `selpmod`

# %%
bjtf_spec = {
    'models': [{'type': 'bjtf', 'nb': [1, 2], 'nc': [0, 1, 2],
                'nd': [0, 1, 2], 'nf': [0, 1, 2], 'delay': [3], 'diff': [0]}]
}
result_bjtf = selpmod(bjtf_spec, y, u)
pmod_bic    = result_bjtf['bjtf']['bicmod']
stat_bic    = result_bjtf['bjtf']['bicstat']

print(f'Best BJTF (BIC):  b={pmod_bic.b}  f={pmod_bic.f}  c={pmod_bic.c}  d={pmod_bic.d}')
pmoddisp(pmod_bic, stat_bic)
pmodpzplot(pmod_bic)
plt.show()

aic_bjtf = pmodaic(pmod_bic, y, u)
bic_bjtf = pmodbic(pmod_bic, y, u)
print(f'Best BJTF (BIC)  —  AIC={aic_bjtf:.4f},  BIC={bic_bjtf:.4f}')