import numpy as np
from ..basefunctions import makerow

def func_xcorr (a,b,maxlag=20,flag='none') :
	"""Compute the cross-correlation (or autocorrelation) between two sequences.

    Produces a symmetric array of length ``2*maxlag + 1`` covering lags
    ``-maxlag`` to ``+maxlag``.

    Parameters
    ----------
    a : array-like
        First 1-D sequence (length N).
    b : array-like
        Second 1-D sequence (same length as ``a``).
    maxlag : int, optional
        Maximum lag to compute; must be less than N. Default 20.
    flag : {'biased', 'unbiased', 'coeff', 'none'}, optional
        Normalization method:

        - ``'biased'``   — divide by N (biased estimate).
        - ``'unbiased'`` — divide by N - |k| (unbiased estimate).
        - ``'coeff'``    — normalize so that the zero-lag value is 1.0.
        - ``'none'``     — no scaling (raw inner product). Default.

    Returns
    -------
    c : ndarray, shape (1, 2*maxlag+1)
        Cross-correlation values at lags ``-maxlag, ..., 0, ..., +maxlag``.
        The zero-lag value is at index ``maxlag``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    >>> e = np.random.default_rng(0).standard_normal(500)
    >>> y = lfilter([1], [1, -0.8], e)
    >>> acf = func_xcorr(y, y, maxlag=20, flag='biased')
    >>> acf.shape
    (1, 41)

    See Also
    --------
    parcor : Partial autocorrelation via Levinson-Durbin.
    gpac   : Generalized partial autocorrelation table.
	"""


	error = ''

	if type(flag) != str:
		error='FLAG should be a string'
		raise Exception(error)

	a = makerow.func_makerow(a)
	b = makerow.func_makerow(b)
	n = len(a[0])
	n1 = len(b[0])

	if maxlag >= n:
		error= 'MAXLAG should be less than the lengths of A and B.'
		raise Exception(error)

	if n != n1:
		error = 'The lengths of A and B should be equal.'
		raise Exception(error)

	c = np.zeros([1, (2 * maxlag + 1)])

	# for i=0:maxlag

	for i in range(maxlag+1):

		ac = np.dot(a[0,0:(n-i)], np.transpose(b[0,i:n]))

		#ac = a(1:(n - i))*b(1 + i: n)';

		if i == 0:
			ac0 = ac

		# if i == 0,
		# ac0 = ac;

		if flag == 'unbiased':
			ac = ac / (n-i)
		elif flag == 'biased':
			ac = ac / n
		elif flag == 'coeff':
			ac = ac / ac0

		c[0, maxlag + i] = ac
		c[0, maxlag - i] = ac

	return c
