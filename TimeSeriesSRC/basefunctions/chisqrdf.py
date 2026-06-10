import numpy as np
from scipy.special import gamma

def func_chisqrdf (q,n) :
	"""Compute the chi-square cumulative distribution function (CDF).

    Returns the probability that a chi-square random variable with ``n``
    degrees of freedom falls in the interval ``[0, q]``.

    Parameters
    ----------
    q : float
        Chi-square statistic (upper integration limit), must be > 0.
    n : int
        Degrees of freedom.

    Returns
    -------
    pr : float
        :math:`P(X \\leq q)` where :math:`X \\sim \\chi^2(n)`.

    Examples
    --------
    >>> from TimeSeriesSRC.basefunctions.chisqrdf import func_chisqrdf
    >>> func_chisqrdf(18.3, 10)   # should be ≈ 0.95
    0.9499...

    See Also
    --------
    uniChi : Uses this function to convert Q statistic to p-value.
	"""

	t = np.round(np.arange(0, q, q/5000),4)
	x = gamma(n/2)

	y = (t**((n - 2) / 2)) * (np.exp((t * - 1)/2)) / ((2 ** (n / 2)) * gamma(n / 2))
	_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
	pr = _trapz(y, t)

	#y = (t.^ ((n - 2) / 2)).* (exp(-t. / 2)) / ((2 ^ (n / 2)) * gamma(n / 2));

	return pr
