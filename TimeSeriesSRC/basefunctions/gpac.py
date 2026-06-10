import warnings
import numpy as np

def func_gpac (acf,nrows,ncols) :
	"""Compute the Generalized Partial Autocorrelation (GPAC) table.

    Each cell ``(j, m)`` of the GPAC is the ratio of two determinants built
    from the ACF, following the Woodward & Gray (1981) construction.  The
    table simultaneously identifies the MA order (row index + 1) and AR order
    (column index + 1) of an ARMA process: cells in an MA(q) column pattern
    or AR(p) row pattern help select model orders.

    Parameters
    ----------
    acf : array-like, shape (1, 2*L+1) or (2*L+1,) or (2*L,)
        Autocorrelation sequence.  If two-sided (zero lag at center), the
        positive-lag half is extracted automatically.  If one-sided
        (zero lag first), it is used directly.
    nrows : int
        Number of GPAC rows (numerator orders 1 .. nrows).
    ncols : int
        Number of GPAC columns (denominator orders 1 .. ncols).

    Returns
    -------
    gpac_array : ndarray, shape (nrows, ncols)
        GPAC table.  Row ``j`` corresponds to MA order ``j+1``; column ``m``
        corresponds to AR order ``m+1``.

    Notes
    -----
    If ``nrows + ncols`` exceeds the half-length of the ACF, ``ncols`` is
    silently truncated and a ``UserWarning`` is issued.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    >>> from TimeSeriesSRC.basefunctions.gpac import func_gpac
    >>> e = np.random.default_rng(0).standard_normal(500)
    >>> y = lfilter([1], [1, -0.8], e)
    >>> acf = func_xcorr(y, y, 25, 'biased')
    >>> G = func_gpac(acf, nrows=5, ncols=5)
    >>> G.shape
    (5, 5)

    See Also
    --------
    plotgpac : Visualize the GPAC table.
    uniAnal  : Computes and plots ACF, PACF, and GPAC together.
	"""

	if len(acf.shape) == 1:
		acf = acf.reshape(1,-1)
	elif acf.shape[0] != 1:
		acf = np.transpose(acf)

	l = int(acf.shape[1] / 2) + 1
	L_half = acf.shape[1] // 2   # (acf.shape[1]-1)//2 for odd-length, same value

	# gpac needs: (nrows-1) + (ncols-1) <= L_half - 1, i.e. nrows+ncols <= L_half+1
	if nrows + ncols > L_half + 1:
		ncols_safe = max(1, L_half + 1 - nrows)
		warnings.warn(
			f'gpac: ACF length {acf.shape[1]} (half-length {L_half}) is too short for '
			f'nrows={nrows}, ncols={ncols} (need nrows+ncols <= {L_half+1}); '
			f'ncols truncated to {ncols_safe}.',
			stacklevel=2)
		ncols = ncols_safe

	gpac_array = np.zeros([nrows, ncols])

	for j in range(nrows):
		for m in range(ncols):
			r1 = np.array([])
			for n in range(m+1):
				xacf = acf[0, l+j-1+n-m:  l+j+n]
				r1 = np.append(r1, xacf[::-1])
				# r1 = [r1; acf(l + j - 2 + n : -1 :l + j - 1 + n - m)];

			xlen = len(r1)
			if xlen > m+1:
				xrows = int(xlen/(m+1))
				r1 = r1.reshape(xrows,-1)
			else:
				r1 = r1.reshape(1,-1)
			rr = np.transpose(acf[0,j + l:j + l + m+1].reshape(1,-1))
			r2 = r1.copy();
			r2[:, m] = rr[:,0];
			gpac_array[j, m] = float(np.linalg.det(r2)) / float(np.linalg.det(r1));

	return gpac_array

