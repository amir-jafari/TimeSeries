import numpy as np
from .xcorr import func_xcorr as xcorr
from .chisqrdf import func_chisqrdf as chisqrdf

def func_uniChi(pmod, y, k=20, alpha=0.05):
	"""Box-Pierce portmanteau chi-square test on one-step-ahead residuals.

    Tests the null hypothesis that the residuals of a fitted model are white
    noise.  Large values of the Q statistic (small p-values) indicate that
    the residuals are autocorrelated and the model does not adequately fit the
    data.

    Parameters
    ----------
    pmod : pmodel
        Estimated prediction model (returned by :func:`estimate`).
    y : array-like
        1-D desired output sequence used to compute residuals.
    k : int, optional
        Number of residual-ACF lags used in the Q statistic. Default 20.
    alpha : float, optional
        Significance level for the test (probability of Type I error).
        Default 0.05.

    Returns
    -------
    passed : int
        1 if the test is passed (residuals appear white), 0 otherwise.
    q : float
        Box-Pierce Q statistic.
    n : int
        Degrees of freedom (``k`` minus the number of free parameters).
    pval : float
        p-value of the Q statistic under the chi-square distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.Model.model import pmodel
    >>> from TimeSeriesSRC.Model.estimate import estimate
    >>> from TimeSeriesSRC.basefunctions.uniChi import func_uniChi
    >>> e = np.random.default_rng(0).standard_normal(500)
    >>> y = lfilter([1, 0.5], [1, -0.8], e)
    >>> pm = pmodel('arma', nc=[1], nd=[1], diff=[0], per=[])
    >>> pm_est, trec, stat = estimate(pm, y, show_plot=False, show_output=False)
    >>> passed, q, n, pval = func_uniChi(pm_est, y)

    See Also
    --------
    multiChi : Chi-square test for transfer function models (residual + cross-correlation).
    uniAnal : ACF/PACF/GPAC analysis for order identification.
	"""

	y_1d = np.asarray(y).ravel()

	# Compute residuals
	e = y_1d - pmod.predict(y_1d)

	# ACF of residuals at lags 0..k
	e_2d = e.reshape(1, -1)
	acf_full = xcorr(e_2d, e_2d, k, 'unbiased')

	# Keep positive lags (0..k), normalize by lag-0 value, drop lag-0
	acf = acf_full[0, k:]       # lags 0..k
	acf = acf / acf[0]          # normalize → ρ_0 = 1
	acf = acf[1:]               # drop lag-0, now lags 1..k

	# Box-Pierce Q statistic
	q = len(y_1d) * np.sum(acf ** 2)

	# Degrees of freedom: lags minus number of free parameters
	X = pmod.getmX()
	num_para = len(X)
	n = k - num_para

	# Chi-square cumulative probability
	pr = chisqrdf(q, n)
	pval = 1.0 - pr
	passed = 1 if pval > alpha else 0
	print('pval:', pval)
	print('alpha:', alpha)
	print('pr:',pr)
	print('q:',q)

	return passed, q, n, pval
