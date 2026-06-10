from . import makerow
from . import xcorr
import numpy as np

def func_impest (u,y,k) :
	"""Estimate the impulse response between two sequences (Wiener-Hopf method).

    Solves the Wiener-Hopf equations :math:`R_{uu}\\, g = R_{uy}` for the
    finite impulse response ``g`` of length ``k+1``.

    Parameters
    ----------
    u : array-like, shape (1, N)
        Input (exogenous) sequence.
    y : array-like, shape (1, N)
        Output sequence (same length as ``u``).
    k : int
        Number of lags of the impulse response to compute; the result covers
        lags 0 through ``k``.

    Returns
    -------
    g : ndarray, shape (k+1,)
        Estimated impulse response coefficients g[0], g[1], ..., g[k].

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.basefunctions.impest import func_impest
    >>> rng = np.random.default_rng(0)
    >>> u = rng.standard_normal((1, 300))
    >>> e = rng.standard_normal((1, 300)) * 0.1
    >>> y = lfilter([1], [1, 0.5], u[0]).reshape(1, -1) + e
    >>> g = func_impest(u, y, k=10)
    >>> g.shape
    (11,)

    See Also
    --------
    multiAnal : Higher-level multivariate analysis that calls this function.
	"""

	u = makerow.func_makerow(u);
	y = makerow.func_makerow(y);

	ru = xcorr.func_xcorr(u, u, k, 'biased');
	ruy = xcorr.func_xcorr(u, y, k, 'biased');

	# This was for the original version of xcorr
	# ruy = xcorr(y, u, k, 'unbiased');
	# This is for the updated xcorr

	l = k + 1
	r1 = np.array([])
	for n in range(l):
		xru = ru[0,l - 1 + n - k: l + n]
		if len(r1) ==0:
			r1= xru[::-1]
			r1= r1.transpose()
		else:
			r1 = np.vstack((r1, xru[::-1].transpose()))

		#r1 = [r1;ru(: -1:l - 1 + n - k)];

	rr = ruy[0,l-1:l + k]


	g= np.linalg.solve(r1,rr)


	return g
