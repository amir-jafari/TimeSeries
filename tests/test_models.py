"""Tests for TimeSeriesSRC.Model."""
import pathlib
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

DATA_DIR = pathlib.Path(__file__).parent.parent / "TimeSeriesSRC" / "TestData"


def load_series_a():
    import pandas as pd
    return pd.read_csv(DATA_DIR / "Series_A_Chemical_Concentration.csv").values.flatten()


def load_series_j():
    import pandas as pd
    df = pd.read_csv(DATA_DIR / "Series_J_Gas_Furnace.csv")
    y  = df.iloc[:, 1].values.flatten()
    u  = df.iloc[:, 0].values.flatten()
    return y, u


# ── pmodel construction ───────────────────────────────────────────────────────

def test_pmodel_arma_creates():
    from TimeSeriesSRC.Model.model import pmodel
    pm = pmodel("arma", nc=[2], nd=[1], diff=[0], per=[])
    assert pm.type == "arma"
    assert pm.nc == [2]
    assert pm.nd == [1]


def test_pmodel_arima_creates():
    from TimeSeriesSRC.Model.model import pmodel
    pm = pmodel("arma", nc=[1], nd=[1], diff=[1], per=[])
    assert pm.diff == [1]


def test_pmodel_arx_creates():
    from TimeSeriesSRC.Model.model import pmodel
    pm = pmodel("arx", na=[2], nb=[2], delay=[1])
    assert pm.type == "arx"


def test_pmodel_armax_creates():
    from TimeSeriesSRC.Model.model import pmodel
    pm = pmodel("armax", na=[2], nb=[2], nc=[1], delay=[1])
    assert pm.type == "armax"


def test_pmodel_bjtf_creates():
    from TimeSeriesSRC.Model.model import pmodel
    pm = pmodel("bjtf", nb=[2], nc=[1], nd=[1], nf=[2], delay=[1])
    assert pm.type == "bjtf"


# ── estimate — ARMA ──────────────────────────────────────────────────────────

def test_estimate_arma_runs():
    from TimeSeriesSRC.Model.model    import pmodel
    from TimeSeriesSRC.Model.estimate import estimate
    y  = load_series_a()
    pm = pmodel("arma", nc=[2], nd=[1], diff=[0], per=[])
    pm.estimParams.epochs = 10
    pm_est, trec, stat = estimate(pm, y, show_plot=False, show_output=False)
    assert "sigma" in stat
    assert stat["sigma"] > 0


def test_estimate_arma_returns_finite_params():
    from TimeSeriesSRC.Model.model    import pmodel
    from TimeSeriesSRC.Model.estimate import estimate
    y  = load_series_a()
    pm = pmodel("arma", nc=[1], nd=[1], diff=[0], per=[])
    pm.estimParams.epochs = 10
    pm_est, trec, stat = estimate(pm, y, show_plot=False, show_output=False)
    X = np.array(pm_est.getmX()).ravel()
    assert np.all(np.isfinite(X))


# ── estimate — ARX ───────────────────────────────────────────────────────────

def test_estimate_arx_runs():
    from TimeSeriesSRC.Model.model    import pmodel
    from TimeSeriesSRC.Model.estimate import estimate
    y, u = load_series_j()
    pm   = pmodel("arx", na=[2], nb=[2], delay=[3])
    pm.estimParams.epochs = 10
    pm_est, trec, stat = estimate(pm, y, u=u, show_plot=False, show_output=False)
    assert "sigma" in stat
    assert stat["sigma"] > 0


# ── criteria ─────────────────────────────────────────────────────────────────

def test_pmodmse_is_positive():
    from TimeSeriesSRC.Model.model    import pmodel
    from TimeSeriesSRC.Model.estimate import estimate
    from TimeSeriesSRC.Model.pmodmse  import func_pmodmse
    y  = load_series_a()
    pm = pmodel("arma", nc=[1], nd=[1], diff=[0], per=[])
    pm.estimParams.epochs = 5
    pm_est, trec, stat = estimate(pm, y, show_plot=False, show_output=False)
    # func_pmodmse returns (mse, e) — must unpack
    mse, e = func_pmodmse(pm_est, y)
    assert mse > 0


def test_pmodaic_and_pmodbic_run():
    from TimeSeriesSRC.Model.model    import pmodel
    from TimeSeriesSRC.Model.estimate import estimate
    from TimeSeriesSRC.Model.pmodaic  import func_pmodaic
    from TimeSeriesSRC.Model.pmodbic  import func_pmodbic
    y  = load_series_a()
    pm = pmodel("arma", nc=[1], nd=[1], diff=[0], per=[])
    pm.estimParams.epochs = 5
    pm_est, trec, stat = estimate(pm, y, show_plot=False, show_output=False)
    aic = func_pmodaic(pm_est, y)
    bic = func_pmodbic(pm_est, y)
    assert np.isfinite(aic)
    assert np.isfinite(bic)


def test_pmodbic_less_than_aic_for_small_n():
    """BIC penalises more than AIC when N is small relative to k."""
    from TimeSeriesSRC.Model.model    import pmodel
    from TimeSeriesSRC.Model.estimate import estimate
    from TimeSeriesSRC.Model.pmodaic  import func_pmodaic
    from TimeSeriesSRC.Model.pmodbic  import func_pmodbic
    y  = load_series_a()
    pm = pmodel("arma", nc=[2], nd=[2], diff=[0], per=[])
    pm.estimParams.epochs = 5
    pm_est, _, _ = estimate(pm, y, show_plot=False, show_output=False)
    aic = func_pmodaic(pm_est, y)
    bic = func_pmodbic(pm_est, y)
    # For N=197, k=4: BIC penalty = k*ln(N)/N ≈ 0.107 > AIC penalty = 2k/N ≈ 0.041
    assert bic > aic