"""Tests for TimeSeriesSRC.basefunctions."""
import pathlib
import numpy as np
import pytest

DATA_DIR = pathlib.Path(__file__).parent.parent / "TimeSeriesSRC" / "TestData"


def load_series_a():
    import pandas as pd
    return pd.read_csv(DATA_DIR / "Series_A_Chemical_Concentration.csv").values.flatten()


def load_series_j():
    import pandas as pd
    df = pd.read_csv(DATA_DIR / "Series_J_Gas_Furnace.csv")
    y = df.iloc[:, 1].values.flatten()
    u = df.iloc[:, 0].values.flatten()
    return y, u


# ── xcorr ────────────────────────────────────────────────────────────────────

def test_xcorr_shape():
    from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    rng = np.random.default_rng(0)
    a = rng.standard_normal(100)
    c = func_xcorr(a, a, 20, "biased")
    assert c.shape[0] == 41  # lags -20 … +20


def test_xcorr_symmetric():
    from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    rng = np.random.default_rng(1)
    a = rng.standard_normal(100)
    c = func_xcorr(a, a, 10, "biased")
    np.testing.assert_allclose(c, c[::-1], atol=1e-10)


# ── parcor ───────────────────────────────────────────────────────────────────

def test_parcor_shape():
    from TimeSeriesSRC.basefunctions.xcorr  import func_xcorr
    from TimeSeriesSRC.basefunctions.parcor import func_parcor
    rng = np.random.default_rng(2)
    a   = rng.standard_normal(200)
    acf = func_xcorr(a, a, 20, "biased")
    mid = len(acf) // 2
    acf_half = acf[mid:]
    pacf, phi, sigma = func_parcor(acf_half, 10)
    assert len(pacf) == 10


# ── sdiff ────────────────────────────────────────────────────────────────────

def test_sdiff_reduces_length():
    from TimeSeriesSRC.basefunctions.sdiff import func_sdiff
    y = np.arange(100, dtype=float)
    yd = func_sdiff(y, [1], [1])
    assert len(yd) < len(y)


def test_sdiff_removes_trend():
    from TimeSeriesSRC.basefunctions.sdiff import func_sdiff
    t  = np.arange(200, dtype=float)
    y  = 3.0 * t + 5.0 + np.random.default_rng(3).standard_normal(200) * 0.01
    yd = func_sdiff(y, [1], [1])
    assert np.std(yd) < np.std(y)


# ── uniAnal ──────────────────────────────────────────────────────────────────

def test_uniAnal_returns_three_outputs():
    import matplotlib
    matplotlib.use("Agg")
    from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal
    y = load_series_a()
    result = func_uniAnal(y, na=20, nump=10, nrg=3, ncg=0,
                          diff=[0], per=[], perdsp=1)
    assert len(result) == 3


def test_uniAnal_acf_at_lag_zero_is_one():
    import matplotlib
    matplotlib.use("Agg")
    from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal
    from TimeSeriesSRC.basefunctions.xcorr  import func_xcorr
    y   = load_series_a()
    acf = func_xcorr(y, y, 20, "biased")
    mid = len(acf) // 2
    assert pytest.approx(acf[mid], abs=1e-6) == 1.0


# ── uniChi ───────────────────────────────────────────────────────────────────

def test_uniChi_runs_on_white_noise():
    import matplotlib
    matplotlib.use("Agg")
    from TimeSeriesSRC.basefunctions.uniChi import func_uniChi
    rng = np.random.default_rng(4)
    e   = rng.standard_normal(200)
    func_uniChi(e, na=20, npar=2, alpha=0.05)


# ── gpac ─────────────────────────────────────────────────────────────────────

def test_gpac_shape():
    from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    from TimeSeriesSRC.basefunctions.gpac  import func_gpac
    rng = np.random.default_rng(5)
    a   = rng.standard_normal(200)
    acf = func_xcorr(a, a, 20, "biased")
    mid = len(acf) // 2
    G   = func_gpac(acf[mid:], nrg=5, ncg=5)
    assert G.shape == (5, 5)