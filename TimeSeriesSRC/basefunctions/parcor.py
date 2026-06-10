import numpy as np
from .makerow import func_makerow

def func_parcor (acf,nump) :
	"""Compute the partial autocorrelation function via the Levinson-Durbin algorithm.

    Parameters
    ----------
    acf : array-like
        Full (two-sided) autocorrelation sequence with the zero-lag value at
        the center — i.e., the output of :func:`func_xcorr`.
        Shape ``(1, 2*L+1)`` or ``(2*L+1,)``.
    nump : int
        Number of PACF terms to compute (orders 1 through ``nump``).

    Returns
    -------
    pacf : ndarray, shape (1, nump)
        Partial autocorrelation values at orders 1 through ``nump``.
    phi : ndarray, shape (nump+1, 1)
        Final AR parameter vector from the Levinson recursion.
    sigma : float
        Residual variance of the AR(nump) model.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    >>> from TimeSeriesSRC.basefunctions.parcor import func_parcor
    >>> e = np.random.default_rng(0).standard_normal(500)
    >>> y = lfilter([1], [1, -0.8], e)
    >>> acf = func_xcorr(y, y, 20, 'biased')
    >>> pacf, phi, sigma = func_parcor(acf, 10)

    See Also
    --------
    xcorr : Compute the autocorrelation sequence.
    gpac  : Generalized partial autocorrelation table.
	"""

	acf = func_makerow(acf)
	xlen = len(acf[0])

	# Take	off	the negative lags of the acf(assume zero lag is in the middle)

	acf = acf[0,int((xlen) / 2): ]

   # Computethe partial autocorrelation function
	sigma = acf[0];
	phi = [1];
	q = acf[1];
	pacf = np.zeros([nump])
	for i in range(nump):
		pacf[i] = float(-q/sigma)
		phi = np.append(phi,0)+ pacf[i] * np.append(0,phi[i+1::-1])
		q = np.dot(acf[i+2 :0: -1],phi)
		sigma = sigma * (1 - pacf[i] ** 2)

	pacf = pacf.reshape(1,-1)
	phi = phi.reshape(1,-1).transpose()

	return pacf,phi,sigma
