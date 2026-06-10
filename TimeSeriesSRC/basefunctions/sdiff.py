from . import makerow
import numpy as np

def func_sdiff (y,d,p=1) :
	"""Apply seasonal (or regular) differencing to a time series.

    Computes the ``d``-th order difference at period ``p``:

    .. math::

        \\nabla_p^d\\, y(t) = \\nabla_p^{d-1}\\, y(t) - \\nabla_p^{d-1}\\, y(t-p)

    For ``p = 1`` this reduces to ordinary differencing (``numpy.diff``).

    Parameters
    ----------
    y : array-like
        1-D or row-vector input series.
    d : int
        Number of differences to apply. ``d = 0`` returns ``y`` unchanged.
    p : int, optional
        Seasonal period. Default 1 (ordinary differencing).

    Returns
    -------
    yd : ndarray, shape (1, N - d*p)
        Differenced series.  Each application of the operator reduces the
        length by ``p``.

    Raises
    ------
    Exception
        If ``d * p`` is larger than or equal to the length of ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> from TimeSeriesSRC.basefunctions.sdiff import func_sdiff
    >>> y = np.arange(1.0, 21.0)
    >>> yd = func_sdiff(y, d=1, p=4)   # seasonal difference at lag 4
    >>> yd.shape
    (1, 16)

    See Also
    --------
    uniAnal : Passes ``diff`` / ``per`` arguments directly to this function.
	"""


	# Make y into row format

	y = makerow.func_makerow(y)
	m, num_pts = y.shape

	if num_pts <= (d * p):
		raise Exception('d*p is larger than the number of points.')

	if p == 1:
		yd = np.diff(y, d)
	else:
		yd = y

		for i in range(d):
			yd = yd[:,  p: num_pts] - yd[:, : num_pts - p]
			num_pts = num_pts - p

	return yd
