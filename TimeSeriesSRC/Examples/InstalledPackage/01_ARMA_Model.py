# %% [markdown]
# # ARMA Model — System Identification Walkthrough
#
# This script uses the **pip-installed** `timeseries-toolbox` package.
# Install it with:  pip install timeseries-toolbox
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

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from importlib.resources import files

from TimeSeriesSRC.Model.model     import pmodel
from TimeSeriesSRC.Model.estimate  import estimate
from TimeSeriesSRC.Model.selpmod   import func_selpmod  as selpmod
from TimeSeriesSRC.Model.pmodmse   import func_pmodmse  as pmodmse
from TimeSeriesSRC.Model.pmoddisp  import func_pmoddisp as pmoddisp
from TimeSeriesSRC.Model.pmoddisp  import func_pmodpzplot as pmodpzplot
from TimeSeriesSRC.basefunctions.uniAnal  import func_uniAnal       as uniAnal
from TimeSeriesSRC.basefunctions.uniChi   import func_uniChi        as uniChi
from TimeSeriesSRC.basefunctions.partoacf import func_partoacf_pmod as partoacf_pmod

np.random.seed(42)
print('Setup complete.')

# %% [markdown]
# ## Series A — Chemical Concentration

# %%
data_path = str(files('TimeSeriesSRC').joinpath('TestData/Series_A_Chemical_Concentration.csv'))
df = pd.read_csv(data_path)
y  = np.array(df['Concentration'])

# %% [markdown]
# ## Step 1 — Choose a Model Class
# No external input -> ARMA(nd, nc) model.

# %%
y = y - np.mean(y)
N = y.size
print(f'Loaded chemical process data: N={N} samples')

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(y)
ax.set_title('Process sequence  y(t) — Concentration')
ax.set_xlabel('Sample')
ax.set_ylabel('Concentration')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 — Select Model Order
# `uniAnal` computes ACF, PACF, and GPAC.

# %%
acf, pacf, gpac = uniAnal(y, na=20, nump=10, nrg=6, ncg=6)

# %% [markdown]
# ## Step 3 — Estimate Parameters

# %%
pmod = pmodel('arma', nc=[1], nd=[1], diff=[0], per=[])
pmod, trec, stat = estimate(pmod, y)

pmoddisp(pmod, stat)
pmodpzplot(pmod)
plt.show()

# %% [markdown]
# ## Step 4 — Validate the Model

# %%
var_e, _ = pmodmse(pmod, y)
lagmax   = 21
acf_theory, _, _ = partoacf_pmod(pmod, var_e, lagmax)

acf_exp        = acf.squeeze()[20:20 + lagmax]
acf_theory_norm = acf_theory / acf_theory[0]
acf_exp_norm    = acf_exp    / acf_exp[0]

lags = np.arange(lagmax)
fig, ax = plt.subplots(figsize=(10, 4))
ax.stem(lags, acf_exp_norm, linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental')
ax.plot(lags, acf_theory_norm, 'r-o', markersize=5, linewidth=1.5,
        label='Theoretical (ARMA fit)')
ax.set_title('ACF: Experimental vs Theoretical')
ax.set_xlabel('Lag')
ax.legend()
plt.tight_layout()
plt.show()

# %%
e = y - pmod.predict(y)
print(f'Residual std: {e.std():.4f}')
acf_e, pacf_e, gpac_e = uniAnal(e, na=20, nump=10, nrg=6, ncg=6)

# %%
passed, q_arma, n_arma, pval = uniChi(pmod, y)
print(f'Chi-square test:  Q={q_arma:.2f},  df={n_arma},  pass={bool(passed)},  pval={pval:.3f}')

# %% [markdown]
# ## Automated Model Selection with `selpmod`

# %%
arma_spec = {
    'models': [{'type': 'arma', 'nc': [0, 1, 2, 3], 'nd': [1, 2, 3], 'diff': [0]}]
}
result  = selpmod(arma_spec, y)
aicmod  = result['arma']['aicmod']
bicmod  = result['arma']['bicmod']
aicstat = result['arma']['aicstat']
bicstat = result['arma']['bicstat']

print(f'Best AIC model: ARMA(nd={int(aicmod.nd[0])}, nc={int(aicmod.nc[0])})')
print(f'Best BIC model: ARMA(nd={int(bicmod.nd[0])}, nc={int(bicmod.nc[0])})')

pmoddisp(bicmod, bicstat)
pmodpzplot(bicmod)
plt.show()