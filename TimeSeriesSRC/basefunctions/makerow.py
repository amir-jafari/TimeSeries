def func_makerow (y):
	"""Ensure a numpy array has shape ``(1, N)`` (row vector).

    If the array is 1-D it is reshaped to ``(1, N)``.  If it is already
    2-D with more rows than columns it is transposed.  Otherwise it is
    returned unchanged.

    Parameters
    ----------
    y : ndarray
        Input array of shape ``(N,)``, ``(N, 1)``, or ``(1, N)``.

    Returns
    -------
    yr : ndarray, shape (1, N)
        Row-vector form of ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> from TimeSeriesSRC.basefunctions.makerow import func_makerow
    >>> func_makerow(np.array([1, 2, 3, 4])).shape
    (1, 4)
	"""

	## what happens if the array size == 0 or grearter than 2

	xshape = y.shape
	if len(xshape) == 1:
		yr = y.reshape(1,-1)
	elif xshape[0] > xshape[1]:
		yr = y.transopose()
	else:
		yr = y

	return yr
