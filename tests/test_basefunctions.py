"""Tests for TimeSeriesSRC.basefunctions."""
import pathlib
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

DATA_DIR = pathlib.Path(__file__).parent.parent / "TimeSeriesSRC" / "TestData"


def load_series_a():
    import pandas as pd
    return pd.read_csv(DATA_DIR / "Series_A_Chemical_Concentration.csv").values.flatten()


# ── xcorr ────────────────────────────────────────────────────────────────────

def test_xcorr_shape():
    from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    rng = np.random.default_rng(0)
    a = rng.standard_normal(100)
    c = func_xcorr(a, a, 20, "biased")
    # func_xcorr returns shape (1, 2*maxlag+1)
    assert c.shape == (1, 41)


def test_xcorr_symmetric():
    from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    rng = np.random.default_rng(1)
    a = rng.standard_normal(100)
    c = func_xcorr(a, a, 10, "biased")
    np.testing.assert_allclose(c, c[:, ::-1], atol=1e-10)


def test_xcorr_zero_lag_is_positive():
    from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    rng = np.random.default_rng(2)
    a = rng.standard_normal(100)
    c = func_xcorr(a, a, 10, "biased")
    mid = 10  # index of zero-lag for maxlag=10
    assert c[0, mid] > 0


# ── parcor ───────────────────────────────────────────────────────────────────

def test_parcor_shape():
    from TimeSeriesSRC.basefunctions.xcorr  import func_xcorr
    from TimeSeriesSRC.basefunctions.parcor import func_parcor
    rng = np.random.default_rng(2)
    a   = rng.standard_normal(200)
    acf = func_xcorr(a, a, 20, "biased")
    # func_parcor expects the full two-sided ACF (1, 2*L+1)
    pacf, phi, sigma = func_parcor(acf, 10)
    # func_parcor returns pacf with shape (1, nump)
    assert pacf.shape == (1, 10)


def test_parcor_sigma_positive():
    from TimeSeriesSRC.basefunctions.xcorr  import func_xcorr
    from TimeSeriesSRC.basefunctions.parcor import func_parcor
    rng = np.random.default_rng(3)
    a   = rng.standard_normal(500)
    acf = func_xcorr(a, a, 20, "biased")
    pacf, phi, sigma = func_parcor(acf, 5)
    assert sigma > 0


# ── sdiff ────────────────────────────────────────────────────────────────────

def test_sdiff_reduces_length():
    from TimeSeriesSRC.basefunctions.sdiff import func_sdiff
    y  = np.arange(100, dtype=float)
    # d and p are scalars (int), not lists
    yd = func_sdiff(y, 1, 1)
    assert yd.shape[1] < len(y)


def test_sdiff_removes_trend():
    from TimeSeriesSRC.basefunctions.sdiff import func_sdiff
    rng = np.random.default_rng(3)
    t   = np.arange(200, dtype=float)
    y   = 3.0 * t + 5.0 + rng.standard_normal(200) * 0.01
    yd  = func_sdiff(y, 1, 1)
    assert np.std(yd) < np.std(y)


def test_sdiff_seasonal():
    from TimeSeriesSRC.basefunctions.sdiff import func_sdiff
    y  = np.arange(24, dtype=float)
    yd = func_sdiff(y, 1, 12)
    # seasonal difference of period 12 removes 12 observations
    assert yd.shape[1] == 12


# ── uniAnal ──────────────────────────────────────────────────────────────────

def test_uniAnal_returns_three_outputs():
    from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal
    y = load_series_a()
    result = func_uniAnal(y, na=20, nump=10, nrg=3, ncg=0,
                          diff=[0], per=[], perdsp=1)
    assert len(result) == 3


def test_uniAnal_gpac_shape():
    from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal
    y = load_series_a()
    yacf, ypacf, ygpac = func_uniAnal(y, na=15, nump=5, nrg=4, ncg=4,
                                       diff=[0], per=[])
    assert ygpac.shape == (4, 4)


# ── uniChi ───────────────────────────────────────────────────────────────────

def test_uniChi_returns_four_values():
    from TimeSeriesSRC.Model.model    import pmodel
    from TimeSeriesSRC.Model.estimate import estimate
    from TimeSeriesSRC.basefunctions.uniChi import func_uniChi
    y  = load_series_a()
    pm = pmodel("arma", nc=[2], nd=[1], diff=[0], per=[])
    pm.estimParams.epochs = 10
    pm_est, _, _ = estimate(pm, y, show_plot=False, show_output=False)
    # func_uniChi(pmod, y, k=20, alpha=0.05) → (passed, q, n, pval)
    result = func_uniChi(pm_est, y, k=20, alpha=0.05)
    assert len(result) == 4
    passed, q, n, pval = result
    assert 0 <= pval <= 1
    assert q >= 0


# ── gpac ─────────────────────────────────────────────────────────────────────

def test_gpac_shape():
    from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    from TimeSeriesSRC.basefunctions.gpac  import func_gpac
    rng = np.random.default_rng(5)
    a   = rng.standard_normal(200)
    acf = func_xcorr(a, a, 20, "biased")
    # func_gpac(acf, nrows, ncols) — positional args
    G = func_gpac(acf, 5, 5)
    assert G.shape == (5, 5)