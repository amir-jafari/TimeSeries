# %% [markdown]
# # ARIMA Model — System Identification Walkthrough
#
# This script uses the **pip-installed** `timeseries-toolbox` package.
# Install it with:  pip install timeseries-toolbox
#
# **Dataset:** Box-Jenkins Series C — Chemical Temperature (226 observations)

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

# %%
data_path = str(files('TimeSeriesSRC').joinpath('TestData/Series_C_Chemical_Temperature.csv'))
df = pd.read_csv(data_path)
y  = np.array(df['Temperature'])
y  = y - np.mean(y)
N  = y.size
print(f'Loaded chemical temperature data: N={N} samples')

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(y)
ax.set_title('Process sequence  y(t) — Temperature (mean removed)')
ax.set_xlabel('Sample')
ax.set_ylabel('Temperature')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1 — Choose a Model Class
# Check for stationarity using `uniAnal` on the raw series.

# %%
acf_raw, pacf_raw, gpac_raw = uniAnal(y, na=20, nump=10, nrg=6, ncg=6)

# %%
dy = np.diff(y)
print(f'Differenced series: N={dy.size} samples')

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(y);  axes[0].set_title('Original  y(t)');    axes[0].set_xlabel('Sample')
axes[1].plot(dy); axes[1].set_title('Differenced  ∇y(t)'); axes[1].set_xlabel('Sample')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 — Select Model Order

# %%
acf_dy, pacf_dy, gpac_dy = uniAnal(dy, na=20, nump=10, nrg=6, ncg=6)

# %% [markdown]
# ## Step 3 — Estimate Parameters

# %%
pmod = pmodel('arma', nc=[0], nd=[1], diff=[1], per=[])
pmod, trec, stat = estimate(pmod, y)

pmoddisp(pmod, stat)
pmodpzplot(pmod)
plt.show()

# %% [markdown]
# ## Step 4 — Validate the Model

# %%
var_e, _ = pmodmse(pmod, dy)
lagmax   = 21
acf_theory, _, _ = partoacf_pmod(pmod, var_e, lagmax)

acf_exp         = acf_dy.squeeze()[20:20 + lagmax]
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
ax.legend()
plt.tight_layout()
plt.show()

# %%
e = dy - pmod.predict(dy)
print(f'Residual std: {e.std():.4f}')
acf_e, pacf_e, gpac_e = uniAnal(e, na=20, nump=10, nrg=6, ncg=6)

# %%
passed, q_arima, n_arima, pval = uniChi(pmod, dy)
print(f'Chi-square test:  Q={q_arima:.2f},  df={n_arima},  pass={bool(passed)},  pval={pval:.3f}')

# %% [markdown]
# ## Automated Model Selection with `selpmod`

# %%
arima_spec = {
    'models': [{'type': 'arma', 'nc': [0, 1, 2, 3], 'nd': [1, 2, 3], 'diff': [1]}]
}
result = selpmod(arima_spec, y)
aicmod = result['arma']['aicmod']
bicmod = result['arma']['bicmod']

print(f'Best AIC model: ARIMA(nd={int(aicmod.nd[0])}, d=1, nc={int(aicmod.nc[0])})')
print(f'Best BIC model: ARIMA(nd={int(bicmod.nd[0])}, d=1, nc={int(bicmod.nc[0])})')