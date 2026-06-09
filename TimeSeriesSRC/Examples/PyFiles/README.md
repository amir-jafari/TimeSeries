# Example Scripts

Python script versions of the [example notebooks](../NoteBooks/).  Each script covers the same four-step system identification workflow as its notebook counterpart and produces identical results.

Scripts use `# %%` cell markers, making them runnable as interactive notebooks in **VS Code** (Python Interactive window) and **Spyder**, or as plain scripts from the command line.

---

## Scripts

### [01 — ARMA Model](01_ARMA_Model.py)

**Dataset:** Box-Jenkins Series A — Chemical Concentration (197 observations)

Demonstrates the complete identification workflow for a stationary univariate series with no external input.

| Step | Task | Tool |
|------|------|----|
| 1 | Choose model class | Domain knowledge → ARMA |
| 2 | Select model order | `uniAnal` — ACF, PACF, GPAC |
| 3 | Estimate parameters | `estimate`, `pmoddisp`, `pmodpzplot` |
| 4 | Validate | Theoretical vs experimental ACF, residual `uniAnal`, `uniChi` |

Includes automated model selection with `selpmod` (AIC and BIC grid search over ARMA orders).

---

### [02 — ARIMA Model](02_ARIMA_Model.py)

**Dataset:** Box-Jenkins Series C — Chemical Temperature (226 observations)

Demonstrates identification of a non-stationary series requiring first differencing before an ARMA model can be fitted.

| Step | Task | Tool |
|------|------|----|
| 1 | Stationarity check | `uniAnal` on raw series; raw ACF shows slow decay |
| 2 | Difference and identify | `np.diff` → `uniAnal` on ∇y; GPAC → ARIMA(1, 1, 0) |
| 3 | Estimate parameters | `estimate(pmod, y)` pre-differences internally |
| 4 | Validate on ∇y | Theoretical ACF, residual `uniAnal`, `uniChi(pmod, dy)` |

Includes `selpmod` grid search over ARIMA structures with fixed d=1.

---

### [03 — Seasonal ARIMA Model](03_Seasonal_ARIMA_Model.py)

**Dataset:** Box-Jenkins Series G — Airline Passengers (144 observations, monthly, period s=12)

Demonstrates identification of a series with both a stochastic trend and a seasonal pattern, requiring regular and seasonal differencing.

| Step | Task | Tool |
|------|------|----|
| 1 | Log transform + stationarity | `uniAnal` on log y; slow ACF decay |
| 2 | Double difference + identify | `sdiff` for ∇∇₁₂ log y; GPAC → ARIMA(0,1,1)×(0,1,1)₁₂ |
| 3 | Estimate | `estimate(pmod, ly)` with `per=[12]` |
| 4 | Validate on w=∇∇₁₂ log y | Theoretical ACF, residual `uniAnal`, `uniChi` |

Includes `selpmod` grid search over seasonal ARIMA structures.

---

### [04 — BJTF / ARMAX / ARX Model](04_BJTF_Model.py)

**Dataset:** Box-Jenkins Series J — Gas Furnace (296 observations, input u = gas flow rate, output y = CO₂ concentration)

Demonstrates full transfer function identification for a system with an observed external input.  All three input-output model classes are fitted and compared.

| Step | Task | Tool |
|------|------|----|
| 1 | Identify input model | `uniAnal` on u → AR(3) |
| 2 | Impulse response + delay | `multiAnal` → delay k=3, GPAC → BJTF orders |
| 3 | Estimate BJTF | `estimate(pmod, y, u)` |
| 4 | Validate | `multiChi`, theoretical G impulse response vs experimental, theoretical H ACF vs experimental |

Also fits ARMAX and ARX structures on the same data for comparison.  Includes `selpmod` grid search over all three model classes.

---

## Running the Scripts

### Command line

```bash
python 01_ARMA_Model.py
```

### VS Code Interactive

Open the file and click **Run Cell** above any `# %%` marker, or use **Run Current File in Interactive Window** to execute all cells in sequence.

### Spyder

Open the file and press **F5** to run the whole script, or **Shift+Enter** to run the current cell.

---

## Path Resolution

Each script locates the package root relative to its own file, so it can be run from any working directory:

```python
_script_dir = os.path.dirname(os.path.abspath(__file__))
timeseries_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
```

---

## Related

- **Jupyter notebooks:** [`../NoteBooks/`](../NoteBooks/) — equivalent notebooks with rendered output.
- **User Guide:** [`../../../../docs/UserGuide.md`](../../../../docs/UserGuide.md) — complete function reference.
