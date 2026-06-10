import numpy as np
from scipy.signal import lfilter
from .xcorr import func_xcorr as xcorr
from .chisqrdf import func_chisqrdf as chisqrdf

def func_multiChi(pmod, y, u, k1=20, k2=20, alpha1=0.05, alpha2=0.05):
	"""Multivariate chi-square tests for transfer function model validation.

    Performs two hypothesis tests on a fitted transfer function model:

    1. **Residual whiteness** — Box-Pierce Q statistic on the one-step-ahead
       residuals ``e``.  Tests whether the noise model H(q) is adequate.
    2. **Residual/input independence** — cross-correlation statistic S between
       the pre-whitened input ``u`` and the residuals ``e``.  Tests whether
       the transfer function G(q) is adequate.

    Parameters
    ----------
    pmod : pmodel
        Estimated prediction model (ARX, ARMAX, or BJTF).
    y : array-like
        1-D desired output sequence.
    u : array-like
        1-D or 2-D input sequence, shape ``(N,)`` or ``(1, N)``.
    k1 : int, optional
        Residual-ACF lags for the Q statistic. Default 20.
    k2 : int, optional
        Cross-correlation lags for the S statistic. Default 20.
    alpha1 : float, optional
        Significance level for the residual whiteness test. Default 0.05.
    alpha2 : float, optional
        Significance level for the cross-correlation test. Default 0.05.

    Returns
    -------
    pass_arr : list of int
        ``[pass_Q, pass_S]`` — 1 if the test passes, 0 otherwise.
    q : float
        Box-Pierce Q statistic (residual autocorrelation).
    pvalq : float
        p-value for Q.
    s : float
        Cross-correlation chi-square statistic.
    pvals : float
        p-value for S.
    nq : int
        Degrees of freedom for Q.
    ns : int
        Degrees of freedom for S.

    Examples
    --------
    >>> import pathlib, pandas as pd
    >>> import TimeSeriesSRC as ts
    >>> data_dir = pathlib.Path(ts.__file__).parent / 'TestData'
    >>> df  = pd.read_csv(data_dir / 'Series_J_Gas_Furnace.csv')
    >>> u   = df.iloc[:, 0].values
    >>> y   = df.iloc[:, 1].values
    >>> pm  = ts.pmodel('arx', na=[2], nb=[2], delay=[3])
    >>> pm_est, trec, stat = ts.estimate(pm, y, u=u,
    ...                                  show_plot=False, show_output=False)
    >>> pass_arr, q, pvalq, s, pvals, nq, ns = ts.multiChi(pm_est, y, u)

    See Also
    --------
    uniChi : Univariate chi-square test (ARMA models without input).
    multiAnal : Impulse response and GPAC analysis for input-output data.
	"""
	from ..Model.selpmod import func_selpmod as selpmod

	y_1d = np.asarray(y).ravel()
	u_1d = np.asarray(u).ravel()
	u_2d = u_1d.reshape(1, -1)

	# Prewhiten the input with an ARMA model
	# nd starts at 1 to avoid the degenerate ARMA(0,0) case (no free parameters)
	print('Prewhitening input, please wait...')
	arma_spec = {
		'models': [{
			'type': 'arma',
			'nc': [0, 1, 2, 3],
			'nd': [1, 2, 3],
			'diff': [0]
		}]
	}
	estpmodu = selpmod(arma_spec, u_1d)
	bicmod = estpmodu['arma']['bicmod']

	# Build MA (C) and AR (D) polynomials with leading 1
	c_coef = np.asarray(bicmod.c[0]).ravel() if bicmod.c and len(bicmod.c) > 0 else np.array([])
	d_coef = np.asarray(bicmod.d[0]).ravel() if bicmod.d and len(bicmod.d) > 0 else np.array([])
	Rq = np.concatenate([[1.0], c_coef])   # C polynomial (MA numerator)
	Sq = np.concatenate([[1.0], d_coef])   # D polynomial (AR denominator)

	# Prewhitened input: al = D(z)/C(z) * u
	al = lfilter(Sq, Rq, u_1d)

	# AUTO CORRELATION TEST — residuals from the fitted model
	e = y_1d - pmod.predict(y_1d, u_2d)

	e_2d = e.reshape(1, -1)
	acf_full = xcorr(e_2d, e_2d, k1, 'unbiased')

	# Keep positive lags (0..k1), normalize by lag-0, drop lag-0
	acf = acf_full[0, k1:]      # lags 0..k1
	acf = acf / acf[0]          # normalize → ρ_0 = 1
	acf = acf[1:]               # drop lag-0, now lags 1..k1

	q = len(y_1d) * np.sum(acf ** 2)

	# Degrees of freedom for Q: lags minus noise model free parameters
	nc = len(pmod.c[0]) if (hasattr(pmod, 'c') and len(pmod.c) > 0 and len(pmod.c[0]) > 0) else 0
	nd = len(pmod.d[0]) if (hasattr(pmod, 'd') and len(pmod.d) > 0 and len(pmod.d[0]) > 0) else 0
	nq = k1 - (nc + nd)

	prq = chisqrdf(q, nq)
	pvalq = 1.0 - prq
	pass_arr = [0, 0]
	pass_arr[0] = 1 if pvalq > alpha1 else 0

	# CROSS CORRELATION TEST — xcorr between prewhitened input and residuals
	al_2d = al.reshape(1, -1)
	ccf_full = xcorr(al_2d, e_2d, k2, 'unbiased')

	# Keep positive lags (0..k2)
	ccf = ccf_full[0, k2:]      # lags 0..k2 (k2+1 elements)

	# Normalize by sqrt(var(al) * var(e))
	alvar = np.var(al, ddof=1)
	epsvar = np.var(e, ddof=1)
	ccf = ccf / np.sqrt(alvar * epsvar)

	s = len(y_1d) * np.sum(ccf ** 2)

	# Degrees of freedom for S: transfer function free parameters
	nb = (len(pmod.b[0]) - 1) if (hasattr(pmod, 'b') and len(pmod.b) > 0 and len(pmod.b[0]) > 0) else 0
	nf = len(pmod.f[0]) if (hasattr(pmod, 'f') and len(pmod.f) > 0 and len(pmod.f[0]) > 0) else 0
	ns = k2 + 1 - (nb + nf + 1)

	prs = chisqrdf(s, ns)
	pvals = 1.0 - prs
	pass_arr[1] = 1 if pvals > alpha2 else 0

	return pass_arr, q, pvalq, s, pvals, nq, ns
