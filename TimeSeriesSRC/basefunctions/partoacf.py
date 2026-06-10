import numpy as np
from scipy.signal import lfilter


def func_partoacf(phi, theta, lagmax, var_a):
	"""Compute the theoretical autocovariance function of an ARMA process.

    Uses the Yule-Walker method: solves a linear system for the first ``p``
    lags, then extends the autocovariance sequence by the AR recursion.

    The ARMA convention is:

    .. math::

        y(t) + \\phi_1 y(t-1) + \\cdots + \\phi_p y(t-p) =
        a(t) + \\theta_1 a(t-1) + \\cdots + \\theta_q a(t-q)

    Parameters
    ----------
    phi : array-like
        AR polynomial **including** the leading 1:
        ``[1, phi1, phi2, ...]``.
    theta : array-like
        MA polynomial **including** the leading 1:
        ``[1, theta1, theta2, ...]``.
    lagmax : int
        Number of lags to compute; output covers lags 0 through
        ``lagmax - 1``.
    var_a : float
        Variance :math:`\\sigma_a^2` of the white-noise input ``a(t)``.

    Returns
    -------
    acf : ndarray, shape (lagmax,)
        Theoretical autocovariance function.  ``acf[0]`` is the variance
        of ``y``.
    imp : ndarray, shape (p,)
        First ``p`` coefficients of the impulse response of
        :math:`C(B) / D(B)`.

    Examples
    --------
    >>> from TimeSeriesSRC.basefunctions.partoacf import func_partoacf
    >>> acf, imp = func_partoacf([1, 0.8], [1], 10, 1.0)
    >>> round(acf[0], 3)   # variance = 1 / (1 - 0.64) ≈ 2.778
    2.778

    See Also
    --------
    func_partoacf_pmod : Wrapper that reads polynomials from a ``pmodel``.
    uniAnal : Computes the sample ACF from data.
	"""

	phi   = np.asarray(phi,   dtype=float).ravel()
	theta = np.asarray(theta, dtype=float).ravel()

	p = len(phi)
	q = len(theta)

	# Pad the shorter polynomial with zeros so both have the same length
	if p > q:
		theta = np.concatenate([theta, np.zeros(p - q)])
	elif p < q:
		phi = np.concatenate([phi, np.zeros(q - p)])
		p = q

	# a1[i,j] = phi[i+j] for i+j < p, else 0  (anti-diagonal Toeplitz, upper-triangular feel)
	a1 = np.zeros((p, p))
	for i in range(p):
		for j in range(p - i):
			a1[i, j] = phi[i + j]
	a1[:, 0] /= 2

	# a2[i,j] = phi[i-j] for i >= j, else 0  (lower-triangular Toeplitz)
	a2 = np.zeros((p, p))
	for i in range(p):
		for j in range(i + 1):
			a2[i, j] = phi[i - j]

	# Solve a2 * imp = theta for the first p values of the impulse response of C/D
	imp = np.linalg.solve(a2, theta)

	# b1: same anti-diagonal structure as a1 but built from theta
	b1 = np.zeros((p, p))
	for i in range(p):
		for j in range(p - i):
			b1[i, j] = theta[i + j]

	# Halve the first column of a2, then solve (a1+a2)*acf = b1*imp
	a2[:, 0] /= 2
	acf = np.linalg.solve(a1 + a2, b1 @ imp)

	# Extend via the AR Yule-Walker recursion for lags p .. lagmax-1
	phi_ar = phi[1:]   # AR coefficients without the leading 1  (length p-1)
	for k in range(p, lagmax):
		# acf[k] = -phi_ar[0]*acf[k-1] - ... - phi_ar[p-2]*acf[k-p+1]
		acf = np.append(acf, -phi_ar @ acf[k - 1 : k - p : -1])

	acf = acf[:lagmax]
	acf *= var_a
	return acf, imp


def func_partoacf_pmod(pmod, var_a, lagmax):
	"""Compute the theoretical autocovariance from a fitted ``pmodel``.

    Extracts the AR (D) and MA (C) polynomials from ``pmod``, composes
    seasonal factors when present, and calls :func:`func_partoacf`.

    Parameters
    ----------
    pmod : pmodel
        Fitted ARMA prediction model.  Seasonal components are handled
        automatically using ``pmod.period``.
    var_a : float
        Variance :math:`\\sigma_a^2` of the white-noise driving process.
    lagmax : int
        Number of autocovariance lags to compute (lags 0 .. lagmax-1).

    Returns
    -------
    acf : ndarray, shape (lagmax,)
        Theoretical autocovariance function. ``acf[0]`` is the variance of
        ``y``.
    imp : ndarray, shape (p,)
        First ``p`` impulse-response coefficients of :math:`C(B)/D(B)`.
    g_ir : ndarray
        Impulse response of the G transfer function (empty for ARMA models).

    Examples
    --------
    >>> import numpy as np
    >>> from TimeSeriesSRC.Model.model import pmodel
    >>> from TimeSeriesSRC.basefunctions.partoacf import func_partoacf_pmod
    >>> pm = pmodel('arma', nc=[1], nd=[1], diff=[0], per=[])
    >>> pm.c[0] = np.array([-0.5])
    >>> pm.d[0] = np.array([-0.8])
    >>> acf, imp, g_ir = func_partoacf_pmod(pm, 1.0, 10)

    See Also
    --------
    func_partoacf : Lower-level function that takes explicit polynomials.
	"""

	# Read the stationary ARMA polynomials directly from the model coefficients.
	# getGH() now includes differencing operators (for ARIMA), which would
	# introduce unit roots and break the Yule-Walker equations here.
	theta = np.array([1.0]) if len(pmod.c) == 0 else np.concatenate([[1.0], np.asarray(pmod.c[0]).ravel()])
	phi   = np.array([1.0]) if len(pmod.d) == 0 else np.concatenate([[1.0], np.asarray(pmod.d[0]).ravel()])

	# Compose seasonal ARMA components if present (differencing excluded)
	for i, per in enumerate(pmod.period):
		if i + 1 < len(pmod.c):
			ctot = int(per) * len(pmod.c[i + 1])
			nh1 = np.zeros(ctot)
			for j, cj in enumerate(pmod.c[i + 1]):
				nh1[(j + 1) * int(per) - 1] = cj
			theta = np.convolve(theta, np.concatenate([[1.0], nh1]))
		if i + 1 < len(pmod.d):
			dtot = int(per) * len(pmod.d[i + 1])
			dh1 = np.zeros(dtot)
			d_i = np.asarray(pmod.d[i + 1]).ravel()
			for j in range(len(d_i)):
				dh1[(j + 1) * int(per) - 1] = d_i[j]
			phi = np.convolve(phi, np.concatenate([[1.0], dh1]))

	acf, imp = func_partoacf(phi, theta, lagmax, var_a)

	# G impulse response — only for models that have an input transfer function
	xtype = getattr(pmod, 'type', 'arma')
	if xtype == 'arma':
		g_ir = np.array([])
	else:
		# B coefficients — no leading 1 (b0, b1, ..., b_nb)
		try:
			b_coefs = np.asarray(pmod.b[0]).ravel()
		except (IndexError, TypeError):
			b_coefs = np.array([1.0])

		# Pure delay k: prepend k zeros to the numerator
		try:
			k = int(np.asarray(pmod.delay).ravel()[0])
		except (IndexError, TypeError, ValueError):
			k = 0
		b_delayed = np.concatenate([np.zeros(k), b_coefs])

		# Denominator: F polynomial for bjtf, A polynomial for arx/armax
		if xtype == 'bjtf':
			try:
				den_coefs = np.asarray(pmod.f[0]).ravel()
			except (IndexError, TypeError):
				den_coefs = np.array([])
		else:
			try:
				den_coefs = np.asarray(pmod.a[0]).ravel()
			except (IndexError, TypeError):
				den_coefs = np.array([])

		den_poly = np.concatenate([[1.0], den_coefs]) if den_coefs.size > 0 else np.array([1.0])

		impulse = np.zeros(lagmax)
		impulse[0] = 1.0
		g_ir = lfilter(b_delayed, den_poly, impulse)

	return acf, imp, g_ir
