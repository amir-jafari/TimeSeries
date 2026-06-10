# %% [markdown]
# # Seasonal ARIMA Model — System Identification Walkthrough
#
# This script uses the **pip-installed** `timeseries-toolbox` package.
# Install it with:  pip install timeseries-toolbox
#
# **Dataset:** Box-Jenkins Series G — Monthly Airline Passengers (144 observations)

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
from TimeSeriesSRC.basefunctions.sdiff    import func_sdiff         as sdiff

np.random.seed(42)
print('Setup complete.')

# %%
data_path = str(files('TimeSeriesSRC').joinpath('TestData/Series_G_Airline_Passengers.csv'))
df = pd.read_csv(data_path)
y  = np.array(df['Passengers'], dtype=float)
N  = y.size
print(f'Loaded airline passenger data: N={N} monthly observations')

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y)
ax.set_title('Box-Jenkins Series G — Monthly Airline Passengers')
ax.set_xlabel('Month (Jan 1949 = 0)')
ax.set_ylabel('Passengers (thousands)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1 — Choose a Model Class
# Log transform + double differencing (d=1, d_s=1 at s=12).

# %%
ly = np.log(y)

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(y);  axes[0].set_title('Raw  y(t)');       axes[0].set_xlabel('Month')
axes[1].plot(ly); axes[1].set_title('Log  log y(t)');   axes[1].set_xlabel('Month')
plt.tight_layout()
plt.show()

acf_ly, pacf_ly, gpac_ly = uniAnal(ly, na=36, nump=10, nrg=6, ncg=6)

# %%
dly = sdiff(ly,  1,  1).flatten()
w   = sdiff(dly, 1, 12).flatten()
print(f'∇log y: N={len(dly)},  w=∇₁₂∇log y: N={len(w)}')

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].plot(dly); axes[0].set_title('∇log y(t)');           axes[0].set_xlabel('Month')
axes[1].plot(w);   axes[1].set_title('w = ∇₁₂∇log y(t)');   axes[1].set_xlabel('Month')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 — Select Model Order

# %%
acf_w, pacf_w, gpac_w = uniAnal(w, na=36, nump=10, nrg=6, ncg=6)

# %% [markdown]
# ## Step 3 — Estimate Parameters
# Classical airline model: ARIMA(0,1,1)×(0,1,1)₁₂

# %%
pmod = pmodel('arma', nc=[1, 1], nd=[0, 0], diff=[1, 1], per=[12])
pmod.estimParams.epochs = 100
pmod.estimParams.goal   = 1e-4
pmod, trec, stat = estimate(pmod, ly)

pmoddisp(pmod, stat)
pmodpzplot(pmod)
plt.show()

c1  = pmod.c[0][0]
cs1 = pmod.c[1][0]
print(f'Regular MA:  c1  = {c1:.4f}  (reference ~= -0.40)')
print(f'Seasonal MA: cs1 = {cs1:.4f}  (reference ~= -0.61)')

# %% [markdown]
# ## Step 4 — Validate the Model

# %%
var_e, _ = pmodmse(pmod, w)
lagmax   = 37
acf_theory, _, _ = partoacf_pmod(pmod, var_e, lagmax)

acf_exp         = acf_w.squeeze()[36:36 + lagmax]
acf_theory_norm = acf_theory / acf_theory[0]
acf_exp_norm    = acf_exp    / acf_exp[0]

lags = np.arange(lagmax)
fig, ax = plt.subplots(figsize=(12, 4))
ax.stem(lags, acf_exp_norm,    linefmt='C0-', markerfmt='C0o', basefmt='k-',
        label='Experimental (w)')
ax.plot(lags, acf_theory_norm, 'r-o', markersize=5, linewidth=1.5,
        label='Theoretical (SARIMA fit)')
ax.set_title('ACF of w = ∇₁₂∇log y: Experimental vs Theoretical')
ax.set_xlabel('Lag')
ax.legend()
plt.tight_layout()
plt.show()

# %%
e = w - pmod.predict(w)
print(f'Residual std: {e.std():.4f}')
acf_e, pacf_e, gpac_e = uniAnal(e, na=36, nump=10, nrg=6, ncg=6)

passed, q_val, n_val, pval = uniChi(pmod, w)
print(f'Portmanteau test: Q={q_val:.2f}, df={n_val}, p-value={pval:.3f}, {"PASS" if passed else "FAIL"}')

# %% [markdown]
# ## Automated Model Selection with `selpmod`

# %%
sarima_spec = {
    'models': [{'type': 'arma', 'nc': [0, 1, 2], 'nd': [0, 1, 2], 'diff': [1], 'per': [12]}]
}
result = selpmod(sarima_spec, ly)
aicmod = result['arma']['aicmod']
bicmod = result['arma']['bicmod']

def fmt_sarima(mod):
    nc   = int(mod.nc[0]) if len(mod.nc) > 0 else 0
    nc_s = int(mod.nc[1]) if len(mod.nc) > 1 else 0
    nd   = int(mod.nd[0]) if len(mod.nd) > 0 else 0
    nd_s = int(mod.nd[1]) if len(mod.nd) > 1 else 0
    return f'ARIMA({nd},1,{nc})×({nd_s},1,{nc_s})₁₂'

print(f'Best AIC: {fmt_sarima(aicmod)}')
print(f'Best BIC: {fmt_sarima(bicmod)}')