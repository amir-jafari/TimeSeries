# TimeSeriesSRC

A Python toolbox for time series analysis and prediction modeling, based on the classical framework of **Box and Jenkins** (*Time Series Analysis: Forecasting and Control*) and **Ljung** (*System Identification: Theory for the User*).

---

## System Identification Process

Building a prediction model is an iterative four-step process.

<p align="center">
  <img src="docs/SysId_Process.svg" alt="System Identification Process" width="220">
</p>

**1. Choose a model class.**  Select the family of models appropriate for your data and application — for example, ARMA for a univariate series with no external input, ARX or ARMAX when an input is available but the model structure should be simple, or BJTF when the input dynamics and noise model need to be identified independently.  The choice is guided by physical knowledge of the system and by preliminary data analysis.

**2. Select model order.**  Determine the polynomial orders ($n_a$, $n_b$, $n_c$, $n_d$, $n_f$) and the input delay $k$.  The toolbox provides `uniAnal` and `multiAnal` to compute the ACF, PACF, GPAC, and impulse response — the standard tools for reading off candidate orders from data.  `selpmod` automates this step by fitting a grid of structures and selecting the best by AIC or BIC.

**3. Estimate parameters.**  Fit the chosen model structure to data using `estimate`, which minimises the sum of squared one-step prediction errors via the Levenberg–Marquardt algorithm and returns the estimated parameter values along with their standard deviations.

**4. Validate the model.**  Check whether the fitted model is adequate.  `uniChi` and `multiChi` perform portmanteau chi-square tests on the residuals and on the cross-correlation between residuals and inputs.  `pmoddisp` shows parameter confidence intervals.  `pmodpzplot` plots the pole-zero map.  If the model fails validation, return to step 2 and adjust the order — or return to step 1 and try a different model class.

---

## The Prediction Model Framework

All models in this toolbox share a common structure: a linear filter driven by a white noise input $e(t)$ and, optionally, an observed external input $u(t)$.  Every model can be written

$$y(t) = G(q)u(t) + H(q)e(t)$$

where $G(q)$ is the transfer function from input to output and $H(q)$ is the noise model.  The models differ in how $G$ and $H$ are parameterized.

### The Backward Shift Operator

All models are expressed using the **backward shift operator** $q^{-1}$, defined by

$$q^{-k}y(t) = y(t-k)$$

A polynomial in $q^{-1}$ of order $n_p$ is written

$$P(q) = 1 + p_1 q^{-1} + p_2 q^{-2} + \cdots + p_{n_p} q^{-n_p}$$

so that $P(q)y(t) = y(t) + p_1 y(t-1) + \cdots + p_n y(t-n_p)$.

Throughout this document $e(t)$ denotes a **white noise** sequence with zero mean and variance $\sigma^2$.

---

### Autoregressive (AR) Model

The simplest model expresses the current value of $y(t)$ as a weighted sum of its own past values plus white noise:

$$D(q)y(t) = e(t)$$

where the **autoregressive polynomial** is

$$D(q) = 1 + d_1 q^{-1} + \cdots + d_{n_d} q^{-n_d}$$

This is an AR($n_d$) model.  The model is stationary when all roots of $D(z) = 0$ lie **outside** the unit circle.  In the toolbox, a pure AR model is a special case of ARMA with $C(q) = 1$:

```python
pmodel('arma', nc=[0], nd=[nd], diff=[0], per=[])
```

---

### Moving Average (MA) Model

A moving average model expresses $y(t)$ as a weighted sum of current and past noise values:

$$y(t) = C(q)e(t)$$

where the **moving average polynomial** is

$$C(q) = 1 + c_1 q^{-1} + \cdots + c_{n_c} q^{-n_c}$$

This is an MA($n_c$) model.  The model is invertible when all roots of $C(z) = 0$ lie **outside** the unit circle.  A pure MA model sets $D(q) = 1$:

```python
pmodel('arma', nc=[nc], nd=[0], diff=[0], per=[])
```

---

### Autoregressive Moving Average (ARMA) Model

Combining both components gives the ARMA($n_d$, $n_c$) model:

$$D(q)y(t) = C(q)e(t)$$

The noise is modeled by the transfer function $H(q) = C(q)/D(q)$.  An ARMA model is typically more parsimonious than a pure AR or MA model of equivalent fit quality: a low-order ARMA can often replace a high-order AR.

```python
pmodel('arma', nc=[nc], nd=[nd], diff=[0], per=[])
```

---

### ARIMA Model

Many real-world series are **non-stationary** — their mean or variance drifts over time.  The ARIMA($n_d$, $d$, $n_c$) model handles this by differencing the series $d$ times before fitting an ARMA model.  The difference operator is

$$\nabla = 1 - q^{-1}, \qquad \nabla^d y(t) = (1 - q^{-1})^dy(t)$$

so that $\nabla y(t) = y(t) - y(t-1)$ and $\nabla^2 y(t) = y(t) - 2y(t-1) + y(t-2)$.  The model equation is

$$D(q)\nabla^d y(t) = C(q)e(t)$$

```python
pmodel('arma', nc=[nc], nd=[nd], diff=[d], per=[])
```

---

### Seasonal ARIMA Model

Seasonal data — such as hourly data with a daily cycle or monthly data with a yearly cycle — requires both a regular and a **seasonal** ARMA component.  Let $s$ denote the period (e.g., $s = 24$ for hourly data with a daily pattern).  The seasonal difference operator is $\nabla_s = 1 - q^{-s}$.

Seasonal AR and MA polynomials involve lags that are multiples of $s$, with orders $n_{d,s}$ and $n_{c,s}$ respectively:

$$D_s(q^{-s}) = 1 + d_{s,1}q^{-s} + \cdots + d_{s,n_{d,s}}q^{-n_{d,s} s}$$

$$C_s(q^{-s}) = 1 + c_{s,1}q^{-s} + \cdots + c_{s,n_{c,s}}q^{-n_{c,s} s}$$

The seasonal ARIMA model with non-seasonal differencing order $d$ and seasonal differencing order $d_s$ is

$$D(q)D_s(q^{-s})\nabla^d\nabla_s^{d_s}y(t) = C(q)C_s(q^{-s})e(t)$$

```python
pmodel('arma', nc=[nc, nc_s], nd=[nd, nd_s], diff=[d, d_s], per=[s])
```

---

### ARX Model — Equation Error Form

When an observed external input $u(t)$ is available, the simplest extension adds an input term and an autoregressive filter:

$$A(q)y(t) = B(q)q^{-k}u(t) + e(t)$$

where $k$ is the pure input delay and

$$A(q) = 1 + a_1 q^{-1} + \cdots + a_{n_a} q^{-n_a}$$

$$B(q) = b_0 + b_1 q^{-1} + \cdots + b_{n_b} q^{-n_b}$$

The transfer functions are

$$G(q) = \frac{B(q)}{A(q)}q^{-k}, \qquad H(q) = \frac{1}{A(q)}$$

Note that the **same polynomial** $A(q)$ governs both the input dynamics and the noise model.  Because the noise $e(t)$ appears as an additive "equation error," the parameters can be estimated by linear least squares — a significant computational advantage.  The trade-off is that the noise poles are constrained to equal the input poles.

```python
pmodel('arx', na=na, nb=[nb], delay=[k])
```

---

### ARMAX Model — Equation Error Form

Adding a moving average term to the noise model relaxes the noise-pole constraint:

$$A(q)y(t) = B(q)q^{-k}u(t) + C(q)e(t)$$

The transfer functions are

$$G(q) = \frac{B(q)}{A(q)}q^{-k}, \qquad H(q) = \frac{C(q)}{A(q)}$$

The polynomial $A(q)$ still appears in both $G$ and $H$, so the noise poles remain tied to the input poles.  The extra $C(q)$ numerator provides more flexibility in shaping the noise spectrum without adding new poles.  Parameter estimation requires non-linear optimization; the toolbox uses the Levenberg–Marquardt algorithm.

```python
pmodel('armax', na=na, nb=[nb], nc=nc, delay=[k])
```

---

### Box-Jenkins Transfer Function (BJTF) Model — Output Error Form

The most general model in the toolbox gives the input dynamics and the noise model **completely independent** parameterizations:

$$y(t) = \frac{B(q)}{F(q)}q^{-k}u(t) + \frac{C(q)}{D(q)}e(t)$$

where

$$F(q) = 1 + f_1 q^{-1} + \cdots + f_{n_f} q^{-n_f}$$

The four polynomials $B$, $F$, $C$, $D$ can be chosen independently.  This separation means the noise model can be identified without contamination from the input dynamics — the defining property of the "output error" form.

The noise transfer function $H(q) = C(q)/D(q)$ reduces to $1/D(q)$ when $C(q) = 1$, giving a special case with a purely autoregressive noise model.  Setting $F(q) = A(q)$ and $D(q) = A(q)$ recovers ARMAX; additionally setting $C(q) = 1$ recovers ARX.

```python
pmodel('bjtf', nb=[nb], nc=[nc], nd=[nd], nf=[nf], delay=[k])
```

---

### One-Step-Ahead Predictor

All models share a common predictor structure.  Given the noise transfer function $H(q) = C(q)/D(q)$, the optimal one-step-ahead predictor is

$$\hat{y}(t \mid t-1) = \left[1 - H^{-1}(q)\right]y(t) + H^{-1}(q)G(q)u(t)$$

For the BJTF model this expands to

$$\hat{y}(t) = \frac{C(q) - D(q)}{C(q)}y(t) + \frac{D(q)B(q)q^{-k}}{C(q)F(q)}u(t)$$

The toolbox minimizes the sum of squared one-step prediction errors $\sum_t [y(t) - \hat{y}(t)]^2$ using the **Levenberg–Marquardt** algorithm.

---

### Model Summary

| Model | AR poly. | Input $B$ | MA poly. $C$ | Input den. $F$ | Notes |
|-------|----------|-----------|--------------|----------------|-------|
| AR    | $D$      | —         | —            | —              | $C=1$ |
| MA    | —        | —         | $C$          | —              | $D=1$ |
| ARMA  | $D$      | —         | $C$          | —              | noise only |
| ARIMA | $D$      | —         | $C$          | —              | + differencing $\nabla^d$ |
| Seasonal ARIMA | $D,D_s$ | — | $C,C_s$ | —         | + seasonal $\nabla_s^{d_s}$ |
| ARX   | $A$      | $B$       | —            | $A$ (shared)   | equation error |
| ARMAX | $A$      | $B$       | $C$          | $A$ (shared)   | equation error |
| BJTF  | $D$      | $B$       | $C$          | $F$            | output error |

---

## Toolbox

📖 **[User Guide](https://amir-jafari.github.io/TimeSeries/UserGuide.html)** — complete function reference with calling formats and argument descriptions.

### Installation

```bash
pip install timeseries-toolbox
```

**Requirements:** Python ≥ 3.8 · NumPy ≥ 1.19 · SciPy ≥ 1.5 · Matplotlib ≥ 3.3

---

### Quick Start

```python
import numpy as np
from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.estimate import estimate
from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal as uniAnal
from TimeSeriesSRC.basefunctions.multiAnal import func_multiAnal as multiAnal
from TimeSeriesSRC.Model.selpmod import func_selpmod as selpmod
from TimeSeriesSRC.Model.pmoddisp import func_pmoddisp as pmoddisp
```

**Univariate analysis** — ACF, PACF, GPAC
```python
acf, pacf, gpac = uniAnal(y, na=20, nump=10)
```

**Fit and estimate an ARMA model**
```python
pmod = pmodel('arma', nc=[3], nd=[2], diff=[0], per=[])
pmod, trec, stat = estimate(pmod, y)
yhat = pmod.predict(y)
pmoddisp(pmod, stat)          # parameter table + confidence intervals
```

**Fit a BJTF model**
```python
pmod = pmodel('bjtf', nb=[2], nc=[0], nd=[2], nf=[2], delay=[3], diff=[0], per=[])
pmod, trec, stat = estimate(pmod, y, u)
yhat = pmod.predict(y, u)
```

**Automatic model selection by AIC / BIC**
```python
spec = {
    'models': [{'type': 'arma', 'nc': [0, 1, 2], 'nd': [1, 2, 3], 'diff': [0]}]
}
result = selpmod(spec, y)
best = result['arma']['bicmod']
```

---

### Package Structure

```
TimeSeriesSRC/
├── basefunctions/     # uniAnal, multiAnal, gpac, xcorr, partoacf, uniChi, multiChi, …
├── Model/             # pmodel, estimate, selpmod, pmodaic, pmodbic, pmoddisp, …
├── Examples/
│   ├── NoteBooks/     # Jupyter notebook walkthroughs
│   └── PyFiles/       # Python script versions
└── TestData/          # Gas furnace and other benchmark datasets
```

---

### Example Notebooks

End-to-end walkthroughs of the four-step system identification process, each on a different Box-Jenkins benchmark dataset.

| Notebook | Model class | Dataset |
|----------|-------------|---------|
| [01 — ARMA Model](TimeSeriesSRC/Examples/NoteBooks/01_ARMA_Model.ipynb) | ARMA | Series A — Chemical Concentration (197 obs.) |
| [02 — ARIMA Model](TimeSeriesSRC/Examples/NoteBooks/02_ARIMA_Model.ipynb) | ARIMA | Series C — Chemical Temperature (226 obs.) |
| [03 — Seasonal ARIMA Model](TimeSeriesSRC/Examples/NoteBooks/03_Seasonal_ARIMA_Model.ipynb) | Seasonal ARIMA | Series G — Airline Passengers (144 obs.) |
| [04 — BJTF / ARMAX / ARX Model](TimeSeriesSRC/Examples/NoteBooks/04_BJTF_Model.ipynb) | BJTF, ARMAX, ARX | Series J — Gas Furnace (296 obs.) |

Python script equivalents are in [TimeSeriesSRC/Examples/PyFiles/](TimeSeriesSRC/Examples/PyFiles/).

---

### Key Functions

#### Analysis

| Function | Description |
|----------|-------------|
| `uniAnal(y, na, nump)` | ACF, PACF and GPAC for a single series |
| `multiAnal(u, y, ...)` | Impulse response, residual ACF and GPAC for $u \to y$ |
| `uniChi(pmod, y)` | Chi-square test on model residuals |
| `multiChi(pmod, y, u)` | Chi-square test for transfer function residuals |
| `partoacf_pmod(pmod, var_e, lagmax)` | Theoretical ACF from a fitted model |

#### Model Selection

| Function | Description |
|----------|-------------|
| `selpmod(spec, y, u)` | Grid search over model structures; returns best AIC and BIC models |
| `pmodaic(pmod, y, u)` | Akaike Information Criterion |
| `pmodbic(pmod, y, u)` | Bayesian Information Criterion |
| `pmodmse(pmod, y, u)` | Mean squared prediction error |

#### Display and Diagnostics

| Function | Description |
|----------|-------------|
| `pmoddisp(pmod, stat)` | Parameter table with ±2σ confidence intervals and error-bar plot |
| `pmodpzplot(pmod)` | Pole-zero map for the $G$ and $H$ transfer functions |

---

## References

- G. E. P. Box and G. M. Jenkins, *Time Series Analysis: Forecasting and Control*, Holden-Day, 1970.
- L. Ljung, *System Identification: Theory for the User*, Prentice Hall, 1987.

---

## Authors

Martin Hagan · Amir Jafari · Lilian S. De Rivera

## License

MIT
