# TimeSeriesSRC — User Guide

This guide describes every public function in the toolbox: its purpose, calling format, arguments, and return values.  Functions are grouped by task.

---

## Contents

1. [Model Building](#1-model-building)
   - [pmodel](#pmodel)
   - [estimate](#estimate)
   - [selpmod](#selpmod)
   - [pmodel.predict](#pmodelpredict)
   - [pmodsim](#pmodsim)
2. [Model Assessment](#2-model-assessment)
   - [pmodmse](#pmodmse)
   - [pmodaic](#pmodaic)
   - [pmodbic](#pmodbic)
   - [pmoddisp](#pmoddisp)
   - [pmodpzplot](#pmodpzplot)
3. [Time Series Analysis](#3-time-series-analysis)
   - [uniAnal](#unianal)
   - [multiAnal](#multianal)
4. [Diagnostic Tests](#4-diagnostic-tests)
   - [uniChi](#unichi)
   - [multiChi](#multichi)
5. [Theoretical Tools](#5-theoretical-tools)
   - [partoacf](#partoacf)
   - [partoacf\_pmod](#partoacf_pmod)
6. [Lower-Level Utilities](#6-lower-level-utilities)
   - [xcorr](#xcorr)
   - [gpac](#gpac)
   - [parcor](#parcor)
   - [impest](#impest)
   - [sdiff](#sdiff)
   - [chisqrdf](#chisqrdf)
   - [plotacf](#plotacf)

---

## 1. Model Building


### `pmodel`

**Module:** `TimeSeriesSRC.Model.model`

```python
from TimeSeriesSRC.Model.model import pmodel
```

**Purpose.**  Creates a prediction model object of the specified type.  The object stores the polynomial orders, delay, differencing, and seasonal period, initialises the parameter arrays, and provides `predict`, `getmX`, and `setmX` methods.

**Syntax**

```python
pmod = pmodel(xtype, na=0, nb=[], nc=[], nd=[], nf=[],
              delay=[], diff=[0], per=[])
```

| Argument | Type | Description |
|----------|------|-------------|
| `xtype`  | str  | Model type: `'arma'`, `'arx'`, `'armax'`, `'bjtf'`, `'regr'` |
| `na`     | int  | Order of the $A(q)$ polynomial (ARX, ARMAX) |
| `nb`     | list[int] | Orders of the $B(q)$ polynomials; one entry per input |
| `nc`     | int or list[int] | Order(s) of the $C(q)$ polynomial(s) |
| `nd`     | list[int] | Order(s) of the $D(q)$ polynomial(s) |
| `nf`     | list[int] | Order(s) of the $F(q)$ polynomial(s) (BJTF only) |
| `delay`  | list[int] | Pure input delay $k$ for each input channel |
| `diff`   | list[int] | Differencing orders; `diff[0]` is the non-seasonal order, `diff[1]` the seasonal order at period `per[0]`, etc. |
| `per`    | list[int] | Seasonal periods; one entry fewer than `nc` / `nd` for seasonal ARMA |

**Polynomial storage conventions**

| Polynomial | Attribute | Storage |
|------------|-----------|---------|
| $A(q) = 1 + a_1 q^{-1} + \cdots$ | `pmod.a[0]` | `[a1, a2, …, ana]` (leading 1 implicit) |
| $B(q) = b_0 + b_1 q^{-1} + \cdots$ | `pmod.b[i]` | `[b0, b1, …, bnb]` (b0 stored explicitly) |
| $C(q) = 1 + c_1 q^{-1} + \cdots$ | `pmod.c[i]` | `[c1, c2, …, cnc]` (leading 1 implicit) |
| $D(q) = 1 + d_1 q^{-1} + \cdots$ | `pmod.d[i]` | `[d1, d2, …, dnd]` (leading 1 implicit) |
| $F(q) = 1 + f_1 q^{-1} + \cdots$ | `pmod.f[i]` | `[f1, f2, …, fnf]` (leading 1 implicit) |

**Examples**

```python
# ARMA(3, 0) — AR model of order 3
pmod = pmodel('arma', nc=[0], nd=[3], diff=[0], per=[])

# ARMAX: na=2, one input with nb=2, nc=1, delay=3
pmod = pmodel('armax', na=2, nb=[2], nc=1, delay=[3])

# BJTF: nb=2, nc=0, nd=2, nf=2, delay=3
pmod = pmodel('bjtf', nb=[2], nc=[0], nd=[2], nf=[2], delay=[3], diff=[0], per=[])
```

---

### `estimate`

**Module:** `TimeSeriesSRC.Model.estimate`

```python
from TimeSeriesSRC.Model.estimate import estimate
```

**Purpose.**  Estimates the parameters of a prediction model from data using the Levenberg–Marquardt algorithm.  Applies any differencing or pre-processing specified in `pmod`, then minimises the sum of squared one-step prediction errors.

**Syntax**

```python
pmod_est, trec, stat = estimate(pmod, y, u=np.array([]),
                                show_plot=True, show_output=True)
```

| Argument | Type | Description |
|----------|------|-------------|
| `pmod` | pmodel | Model with orders and initial parameters |
| `y` | array (N,) or (1,N) | Output (desired) time series |
| `u` | array (1,N) or (M,N) | Input time series; omit for ARMA |
| `show_plot` | bool | Display live training-progress plot; default `True` |
| `show_output` | bool | Print per-epoch progress to stdout; default `True` |

| Return | Description |
|--------|-------------|
| `pmod_est` | pmodel with estimated parameter arrays updated in-place |
| `trec` | Training record dict: `trec['index']` = MSE per epoch, `trec['mu']` = adaptive μ per epoch |
| `stat` | Statistics dict: `stat['sigma']` = residual variance, `stat['stdx']` = parameter standard deviations (same order as `pmod.getmX()`) |

**Notes.**
- `y` must be zero-mean (or close to it).  The function prints a warning if the mean exceeds two standard deviations.
- `estimate` pre-differences `y` (and `u`) internally according to `pmod.diff` before fitting.  Pass the original undifferenced data; callers that later call `predict` or `pmodmse` on the same data must pre-difference manually.

---

### `selpmod`

**Module:** `TimeSeriesSRC.Model.selpmod`

```python
from TimeSeriesSRC.Model.selpmod import func_selpmod as selpmod
```

**Purpose.**  Grid-searches over a set of model structures and selects the best model by AIC and BIC.  Prints progress as each model is fitted.

**Syntax**

```python
result = selpmod(spec, y, u=[])
```

| Argument | Type | Description |
|----------|------|-------------|
| `spec` | dict | Model specification (see below) |
| `y` | array | Output time series |
| `u` | array | Input time series; omit for ARMA |

**Specification format.**  `spec` is a dict with a single key `'models'` whose value is a list of model dicts.  Each model dict has a key `'type'` plus lists of orders to search over:

```python
spec = {
    'models': [
        {
            'type': 'arma',
            'nc': [0, 1, 2],    # values to try for nc
            'nd': [1, 2, 3],    # values to try for nd
            'diff': [0]
        },
        {
            'type': 'bjtf',
            'nb': [1, 2], 'nc': [0], 'nd': [2], 'nf': [1, 2],
            'delay': [3], 'diff': [0]
        }
    ]
}
```

| Return key | Description |
|------------|-------------|
| `result[type]['aicmod']` | Best pmodel by AIC |
| `result[type]['bicmod']` | Best pmodel by BIC |
| `result[type]['aicstat']` | `stat` dict from the AIC-best fit |
| `result[type]['bicstat']` | `stat` dict from the BIC-best fit |
| `result[type]['aic']` | Array of AIC values for all combinations tried |
| `result[type]['bic']` | Array of BIC values for all combinations tried |

---

### `pmodel.predict`

**Purpose.**  Computes one-step-ahead predictions $\hat{y}(t \mid t-1)$ using the current parameter values.

**Syntax**

```python
yhat = pmod.predict(y)           # ARMA (no input)
yhat = pmod.predict(y, u)        # ARX, ARMAX, BJTF
```

| Argument | Type | Description |
|----------|------|-------------|
| `y` | array (N,) or (1,N) | Output time series |
| `u` | array (1,N) or (M,N) | Input time series |

| Return | Description |
|--------|-------------|
| `yhat` | Array of one-step predictions, same length as `y` |

---

### `pmodsim`

**Module:** `TimeSeriesSRC.Model.pmodsim`

```python
from TimeSeriesSRC.Model.pmodsim import func_pmodsim as pmodsim
```

**Purpose.**  Simulates the model output by driving it with a supplied white-noise sequence `e` and input `u`.  Unlike `predict`, simulation does not feed back the true output — it propagates `e` through $H(q)$ and `u` through $G(q)$.

**Syntax**

```python
y = pmodsim(pmod, e, u=[])
```

| Argument | Type | Description |
|----------|------|-------------|
| `pmod` | pmodel | Fitted prediction model |
| `e` | array (1,N) | White noise input sequence |
| `u` | array (1,N) or (M,N) | External input; omit for ARMA |

| Return | Description |
|--------|-------------|
| `y` | Simulated output array |

---

## 2. Model Assessment


### `pmodmse`

**Module:** `TimeSeriesSRC.Model.pmodmse`

```python
from TimeSeriesSRC.Model.pmodmse import func_pmodmse as pmodmse
```

**Purpose.**  Computes the mean squared one-step prediction error and the vector of individual errors.

**Syntax**

```python
mse, e = pmodmse(pmod, y, u=[])
```

| Argument | Type | Description |
|----------|------|-------------|
| `pmod` | pmodel | Fitted model |
| `y` | array | Output time series |
| `u` | array | Input time series; omit for ARMA |

| Return | Description |
|--------|-------------|
| `mse` | Scalar mean squared error |
| `e` | Array of prediction errors $y - \hat{y}$ |

---

### `pmodaic`

**Module:** `TimeSeriesSRC.Model.pmodaic`

```python
from TimeSeriesSRC.Model.pmodaic import func_pmodaic as pmodaic
```

**Purpose.**  Computes the Akaike Information Criterion for a fitted model:

$$\text{AIC} = \ln(\text{MSE}) + \frac{2\,n_\theta}{N}$$

where $n_\theta$ is the number of free parameters and $N$ is the number of data points.  Lower is better.

**Syntax**

```python
aic = pmodaic(pmod, y, u=[])
```

| Argument | Type | Description |
|----------|------|-------------|
| `pmod` | pmodel | Fitted model |
| `y` | array | Output time series |
| `u` | array | Input time series; omit for ARMA |

| Return | Description |
|--------|-------------|
| `aic` | Scalar AIC value |

**Notes.**  `pmodaic` pre-differences `y` (and `u`) internally.  Pass the original undifferenced data.

---

### `pmodbic`

**Module:** `TimeSeriesSRC.Model.pmodbic`

```python
from TimeSeriesSRC.Model.pmodbic import func_pmodbic as pmodbic
```

**Purpose.**  Computes the Bayesian Information Criterion:

$$\text{BIC} = \ln(\text{MSE}) + \frac{n_\theta \ln N}{N}$$

BIC penalises model complexity more heavily than AIC and tends to select more parsimonious models.

**Syntax**

```python
bic = pmodbic(pmod, y, u=[])
```

| Argument | Type | Description |
|----------|------|-------------|
| `pmod` | pmodel | Fitted model |
| `y` | array | Output time series |
| `u` | array | Input time series; omit for ARMA |

| Return | Description |
|--------|-------------|
| `bic` | Scalar BIC value |

**Notes.**  `pmodbic` pre-differences `y` (and `u`) internally.  Pass the original undifferenced data.

---

### `pmoddisp`

**Module:** `TimeSeriesSRC.Model.pmoddisp`

```python
from TimeSeriesSRC.Model.pmoddisp import func_pmoddisp as pmoddisp
```

**Purpose.**  Prints a table of parameter estimates with ±2σ confidence intervals and produces a horizontal error-bar plot.  The parameter order matches `pmod.getmX()`.

**Syntax**

```python
pmoddisp(pmod, stat)
```

| Argument | Type | Description |
|----------|------|-------------|
| `pmod` | pmodel | Fitted model |
| `stat` | dict | Statistics dict returned by `estimate` or stored in `selpmod` result as `aicstat`/`bicstat` |

**Output.**  Printed table and a Matplotlib figure.  No return value.

**Example**

```python
pmod, trec, stat = estimate(pmod, y, u)
pmoddisp(pmod, stat)

# or from selpmod:
stat = result['arma']['bicstat']
pmoddisp(result['arma']['bicmod'], stat)
```

---

### `pmodpzplot`

**Module:** `TimeSeriesSRC.Model.pmoddisp`

```python
from TimeSeriesSRC.Model.pmoddisp import func_pmodpzplot as pmodpzplot
```

**Purpose.**  Plots the poles (×) and zeros (○) of the $G$ and $H$ transfer functions on the complex plane with the unit circle for reference.

- **G poles:** roots of $F(q) \cdot A(q)$
- **G zeros:** roots of $B(q)$
- **H poles:** roots of $D(q) \cdot A(q)$
- **H zeros:** roots of $C(q)$

For ARMA models (no input) only the H plot is shown.  For `regr` (static model) the function prints a message and returns.

**Seasonal models.**  Each seasonal component $C_s(B^s)$ / $D_s(B^s)$ is displayed in a separate subplot using the compressed variable $w = B^s$.  This maps each seasonal polynomial of degree $m$ in $B^s$ to $m$ roots in $w$-space, avoiding the $s \times m$ z-plane roots that would otherwise appear (e.g. 12 roots for a single seasonal MA order at period 12).  The unit circle in $w$-space still correctly separates invertible (inside) from non-invertible (outside) roots.

**Syntax**

```python
pmodpzplot(pmod)
```

| Argument | Type | Description |
|----------|------|-------------|
| `pmod` | pmodel | Fitted model |

**Output.**  Matplotlib figure.  No return value.

---

## 3. Time Series Analysis


### `uniAnal`

**Module:** `TimeSeriesSRC.basefunctions.uniAnal`

```python
from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal as uniAnal
```

**Purpose.**  Performs a complete univariate analysis of a single time series: computes and plots the ACF, PACF, and GPAC table.  Used to identify the orders of an ARMA model.

**Syntax**

```python
acf, pacf, gpac = uniAnal(y, na=20, nump=10, nrg=5, ncg=0,
                           diff=[0], per=[], perdsp=1)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `y` | array (1,N) or (N,) | — | Time series |
| `na` | int | 20 | ACF lags computed from −na to +na |
| `nump` | int | 10 | Number of PACF terms |
| `nrg` | int | 5 | GPAC rows |
| `ncg` | int | `nrg` | GPAC columns; pass `0` (default) to match `nrg` |
| `diff` | list[int] | `[0]` | Differencing orders before analysis |
| `per` | list[int] | `[]` | Seasonal periods for differencing |
| `perdsp` | int | 1 | Period at which to display ACF/PACF |

| Return | Description |
|--------|-------------|
| `acf` | ACF array, shape (1, 2·na+1); zero lag at centre |
| `pacf` | PACF array, shape (nump,) |
| `gpac` | GPAC table, shape (nrg, ncg) |

---

### `multiAnal`

**Module:** `TimeSeriesSRC.basefunctions.multiAnal`

```python
from TimeSeriesSRC.basefunctions.multiAnal import func_multiAnal as multiAnal
```

**Purpose.**  Performs a multivariate analysis between an input sequence $u$ and an output sequence $y$.  Estimates the impulse response $G$, the residual ACF (for the noise $v = y - G*u$), and the GPAC tables for both the $G$ and $H$ transfer functions.  The input is pre-whitened with a BIC-selected ARMA model before the impulse response is estimated.

**Syntax**

```python
g, rv, g_gpac, h_gpac = multiAnal(u, y, nng=5, ndg=5, nnh=5, ndh=5,
                                   lg=12, lh=12)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `u` | array (1,N) or (N,) | — | Input sequence |
| `y` | array (1,N) or (N,) | — | Output sequence |
| `nng` | int | 5 | Max numerator order for G GPAC |
| `ndg` | int | 5 | Max denominator order for G GPAC |
| `nnh` | int | 5 | Max numerator order for H GPAC |
| `ndh` | int | 5 | Max denominator order for H GPAC |
| `lg` | int | 12 | Number of impulse response lags |
| `lh` | int | 12 | Reserved; not currently used (see Notes) |

| Return | Description |
|--------|-------------|
| `g` | Estimated impulse response, shape (lg+1,) |
| `rv` | Residual ACF array, shape (1, 2·(nnh+ndh+1)+1); zero lag at centre index `nnh+ndh+1` |
| `g_gpac` | GPAC for $G$ transfer function |
| `h_gpac` | GPAC for $H$ transfer function |

**Notes.**
- `u` and `y` should be zero-mean.
- The function internally pre-whitens `u` with a BIC-selected ARMA model via `selpmod`, then estimates the impulse response by solving the Wiener-Hopf equation.  Progress messages are printed during pre-whitening.
- `rv` lags are determined by `L = nnh + ndh + 1`, not by `lh`.  The `lh` parameter is accepted for compatibility but is not used.
- The G GPAC computation zero-pads `g` if `nng + ndg > lg + 1`, so there is no hard constraint between `lg` and the GPAC orders.

---

## 4. Diagnostic Tests


### `uniChi`

**Module:** `TimeSeriesSRC.basefunctions.uniChi`

```python
from TimeSeriesSRC.basefunctions.uniChi import func_uniChi as uniChi
```

**Purpose.**  Performs a portmanteau (Box–Pierce) lack-of-fit test on the residuals of a univariate model.  A large chi-square statistic indicates residual autocorrelation, meaning the model has not captured all structure in the data.

**Syntax**

```python
passed, q, n, pval = uniChi(pmod, y, k=20, alpha=0.05)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `pmod` | pmodel | — | Fitted ARMA model |
| `y` | array | — | Output time series |
| `k` | int | 20 | Number of ACF lags used in the statistic |
| `alpha` | float | 0.05 | Significance level |

| Return | Description |
|--------|-------------|
| `passed` | 1 if test passes (residuals consistent with white noise), 0 otherwise |
| `q` | Box–Pierce statistic |
| `n` | Degrees of freedom (`k` minus number of model parameters) |
| `pval` | p-value; `pval > alpha` means pass |

---

### `multiChi`

**Module:** `TimeSeriesSRC.basefunctions.multiChi`

```python
from TimeSeriesSRC.basefunctions.multiChi import func_multiChi as multiChi
```

**Purpose.**  Performs two chi-square tests for a transfer function model: (1) a portmanteau test for whiteness of the residuals, and (2) a test for cross-correlation between the pre-whitened input and the residuals.  Both tests should pass for a well-fitted model.

**Syntax**

```python
passed, q, pvalq, s, pvals, nq, ns = multiChi(pmod, y, u,
                                               k1=20, k2=20,
                                               alpha1=0.05, alpha2=0.05)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `pmod` | pmodel | — | Fitted ARX, ARMAX, or BJTF model |
| `y` | array | — | Output time series (1-D) |
| `u` | array | — | Input time series (1-D or (1,N)) |
| `k1` | int | 20 | ACF lags for residual whiteness test |
| `k2` | int | 20 | Lags for cross-correlation test |
| `alpha1` | float | 0.05 | Significance level for residual test |
| `alpha2` | float | 0.05 | Significance level for cross-correlation test |

| Return | Description |
|--------|-------------|
| `passed` | List `[pass1, pass2]`; 1 = pass, 0 = fail |
| `q` | Chi-square statistic for residual whiteness |
| `pvalq` | p-value for residual test |
| `s` | Chi-square statistic for cross-correlation |
| `pvals` | p-value for cross-correlation test |
| `nq` | Degrees of freedom for `q` |
| `ns` | Degrees of freedom for `s` |

**Notes.**  `multiChi` internally pre-whitens `u` with a BIC-selected ARMA model via `selpmod` before forming the cross-correlation test statistic.  Progress messages are printed during pre-whitening.

---

## 5. Theoretical Tools


### `partoacf`

**Module:** `TimeSeriesSRC.basefunctions.partoacf`

```python
from TimeSeriesSRC.basefunctions.partoacf import func_partoacf as partoacf
```

**Purpose.**  Computes the theoretical autocovariance function of a stationary ARMA process $D(q)y(t) = C(q)e(t)$ using the Yule-Walker method.  Useful for comparing a fitted model's theoretical ACF with the sample ACF.

**Syntax**

```python
acf, imp = partoacf(phi, theta, lagmax, var_a)
```

| Argument | Type | Description |
|----------|------|-------------|
| `phi` | array | Full AR polynomial including leading 1: `[1, d1, d2, …]` |
| `theta` | array | Full MA polynomial including leading 1: `[1, c1, c2, …]` |
| `lagmax` | int | Number of lags to compute (output covers lags 0 … lagmax−1) |
| `var_a` | float | Variance of white noise input |

| Return | Description |
|--------|-------------|
| `acf` | Autocovariance function, shape (lagmax,); `acf[0]` = variance of $y$ |
| `imp` | First `p` elements of the impulse response of $C(q)/D(q)$ |

---

### `partoacf_pmod`

**Module:** `TimeSeriesSRC.basefunctions.partoacf`

```python
from TimeSeriesSRC.basefunctions.partoacf import func_partoacf_pmod as partoacf_pmod
```

**Purpose.**  Convenience wrapper around `partoacf` that accepts a fitted `pmodel` directly instead of raw polynomial arrays.  Reads the stationary $C$ and $D$ polynomial coefficients directly from `pmod.c` and `pmod.d` (excluding differencing operators), computes the theoretical autocovariance of $H(q) = C(q)/D(q)$, and — for transfer-function models — computes the impulse response of $G(q) = B(q^{-k})/F(q)$ via `scipy.signal.lfilter`.

**Syntax**

```python
acf, imp, g_ir = partoacf_pmod(pmod, var_a, lagmax)
```

| Argument | Type | Description |
|----------|------|-------------|
| `pmod` | pmodel | Fitted model (any type) |
| `var_a` | float | White noise variance $\sigma_e^2$ |
| `lagmax` | int | Number of lags to compute (output covers lags 0 … lagmax−1) |

| Return | Description |
|--------|-------------|
| `acf` | Theoretical autocovariance of $H(q)e(t)$, shape (lagmax,); `acf[0]` = variance of $v$ |
| `imp` | First $p$ elements of the impulse response of $C(q)/D(q)$ |
| `g_ir` | Theoretical impulse response of $G(q) = B(q^{-k})/F(q)$, shape (lagmax,); pure delay $k$ appears as leading zeros.  Empty array (`np.array([])`) for ARMA models. |

**Notes.**
- For ARMA models `g_ir` is always an empty array.
- The `acf` returned is the autocovariance of $v(t) = H(q)e(t)$, scaled by `var_a`.  It can be compared directly with the residual ACF `rv` returned by `multiAnal` (after normalising both by their lag-0 value).

---

## 6. Lower-Level Utilities

These functions are called internally by `uniAnal`, `multiAnal`, and the diagnostic tests, but can also be used directly.

---

### `xcorr`

**Module:** `TimeSeriesSRC.basefunctions.xcorr`

```python
from TimeSeriesSRC.basefunctions.xcorr import func_xcorr as xcorr
```

**Purpose.**  Computes the cross-correlation (or autocorrelation when `a == b`) between two sequences over a range of lags.

**Syntax**

```python
C = xcorr(a, b, maxlag=20, flag='none')
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `a`, `b` | array (1,N) or (N,) | — | Input sequences |
| `maxlag` | int | 20 | Maximum lag; output covers lags −maxlag … +maxlag |
| `flag` | str | `'none'` | Normalisation: `'biased'` (÷N), `'unbiased'` (÷(N−\|k\|)), `'coeff'` (lag-0 = 1), `'none'` |

| Return | Description |
|--------|-------------|
| `C` | Array shape (1, 2·maxlag+1); zero lag at index `maxlag` |

---

### `gpac`

**Module:** `TimeSeriesSRC.basefunctions.gpac`

```python
from TimeSeriesSRC.basefunctions.gpac import func_gpac as gpac
```

**Purpose.**  Computes the Generalized Partial Autocorrelation (GPAC) table from an ACF sequence.  The GPAC is used to identify the orders of an ARMA model: a constant column $m$ starting at row 0 suggests a pure AR($m$) process; a constant row $k$ starting at column 0 suggests a pure MA($k$) process.

**Syntax**

```python
gpac_array = gpac(acf, nrows, ncols)
```

| Argument | Type | Description |
|----------|------|-------------|
| `acf` | array | ACF with zero lag at the centre (output of `xcorr`) |
| `nrows` | int | Number of GPAC rows to compute |
| `ncols` | int | Number of GPAC columns to compute |

| Return | Description |
|--------|-------------|
| `gpac_array` | GPAC table, shape (nrows, ncols) |

**Notes.**  The ACF half-length $L$ must satisfy `nrows + ncols ≤ L + 1`.  If this is violated the function emits a warning and silently truncates `ncols`.

---

### `parcor`

**Module:** `TimeSeriesSRC.basefunctions.parcor`

```python
from TimeSeriesSRC.basefunctions.parcor import func_parcor as parcor
```

**Purpose.**  Computes the partial autocorrelation function (PACF) using the Levinson–Durbin recursion.  The PACF cuts off after lag $p$ for a pure AR($p$) process.

**Syntax**

```python
pacf, phi, sigma = parcor(acf, nump)
```

| Argument | Type | Description |
|----------|------|-------------|
| `acf` | array | ACF with zero lag at centre |
| `nump` | int | Number of PACF terms to compute |

| Return | Description |
|--------|-------------|
| `pacf` | PACF values at lags 1 … nump |
| `phi` | AR coefficient matrix from the Levinson recursion |
| `sigma` | Residual variance for each AR order 1 … nump |

---

### `impest`

**Module:** `TimeSeriesSRC.basefunctions.impest`

```python
from TimeSeriesSRC.basefunctions.impest import func_impest as impest
```

**Purpose.**  Estimates the impulse response between an input sequence $u$ and an output sequence $y$ by solving the Wiener-Hopf equation.  Called internally by `multiAnal` after pre-whitening.

**Syntax**

```python
g = impest(u, y, k)
```

| Argument | Type | Description |
|----------|------|-------------|
| `u` | array (1,N) | Input sequence |
| `y` | array (1,N) | Output sequence |
| `k` | int | Number of impulse response lags to compute |

| Return | Description |
|--------|-------------|
| `g` | Impulse response estimate, length `k+1` |

---

### `sdiff`

**Module:** `TimeSeriesSRC.basefunctions.sdiff`

```python
from TimeSeriesSRC.basefunctions.sdiff import func_sdiff as sdiff
```

**Purpose.**  Differences a time series $d$ times at period $p$: $\nabla_p^d y(t) = (1 - q^{-p})^d y(t)$.  For non-seasonal differencing use the default `p=1`.

**Syntax**

```python
yd = sdiff(y, d, p=1)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `y` | array (1,N) | — | Input sequence |
| `d` | int | — | Number of differences to take |
| `p` | int | 1 | Period of each difference |

| Return | Description |
|--------|-------------|
| `yd` | Differenced sequence, length `N − d·p` |

---

### `chisqrdf`

**Module:** `TimeSeriesSRC.basefunctions.chisqrdf`

```python
from TimeSeriesSRC.basefunctions.chisqrdf import func_chisqrdf as chisqrdf
```

**Purpose.**  Evaluates the chi-square cumulative distribution function: returns the probability that a chi-square random variable with `n` degrees of freedom falls in $[0, q]$.  Used internally by `uniChi` and `multiChi`.

**Syntax**

```python
p = chisqrdf(q, n)
```

| Argument | Type | Description |
|----------|------|-------------|
| `q` | float | Chi-square statistic |
| `n` | int | Degrees of freedom |

| Return | Description |
|--------|-------------|
| `p` | Cumulative probability $P(\chi^2_n \leq q)$ |

---

### `plotacf`

**Module:** `TimeSeriesSRC.basefunctions.plotacf`

```python
from TimeSeriesSRC.basefunctions.plotacf import func_plotacf as plotacf
```

**Purpose.**  Plots an autocorrelation function (from `xcorr`) as a styled stem plot.  Can either create a new figure or draw into a supplied `Axes` object for embedding in a multi-panel figure.

**Syntax**

```python
ax = plotacf(acf, maxlag=None, gtitle='Autocorrelation Function', ax=None)
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `acf` | array | — | ACF array from `xcorr`, shape (1, 2·maxlag+1) or flat |
| `maxlag` | int or None | `None` | Maximum lag; inferred from `acf` length if `None` |
| `gtitle` | str | `'Autocorrelation Function'` | Plot title |
| `ax` | Axes or None | `None` | Existing Matplotlib axes to draw into; creates a new figure if `None` |

| Return | Description |
|--------|-------------|
| `ax` | The Matplotlib `Axes` object used for the plot |

**Notes.**  When `ax=None` the function calls `plt.show()` automatically.  When an `ax` is supplied the caller is responsible for displaying the figure.

**Example**

```python
acf = xcorr(y, y, 20, 'biased')
plotacf(acf, maxlag=20, gtitle='ACF of residuals')
```

---

## See Also

- [README](../README.md) — Mathematical background and model equations
- Jupyter notebook walkthroughs: `TimeSeriesSRC/Examples/NoteBooks/`
  - `01_ARMA_Model.ipynb`
  - `02_ARIMA_Model.ipynb`
  - `03_Seasonal_ARIMA_Model.ipynb`
  - `04_BJTF_Model.ipynb`
- `TimeSeriesSRC/Examples/PyFiles/` — Python script versions
