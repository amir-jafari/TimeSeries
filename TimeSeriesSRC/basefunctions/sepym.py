import numpy as np

def func_sepym (y):
	"""Unpack a (possibly weighted) output argument into its components.

    Accepts either a plain array or a dict with keys ``'y'`` and ``'m'``.
    Always returns the array, the weight vector, and a canonical structured
    form so downstream code can use a uniform interface.

    Parameters
    ----------
    y : array-like or dict
        Output data.  If a dict, must have:

        - ``y['y']`` — output array.
        - ``y['m']`` — sample-weight vector (same length as ``y['y']``).

        If a plain array, all weights default to 1.

    Returns
    -------
    ystru : dict
        Dict with keys ``'y'`` (array) and ``'m'`` (weights).
    y : ndarray
        Output array (extracted from ``ystru`` if input was a dict).
    m : ndarray, shape (1, N)
        Row vector of sample weights.

    See Also
    --------
    func_estimlm  : Passes ``y`` through this before optimisation.
    func_jacobian : Also unpacks ``y`` via this function.
	"""


	if type(y) is dict:
		ystru = y
		if 'm' in y:
			m = y['m']
		else:
			error='m is not a field of input y'
			raise Exception(error)

		if 'y' in y:
			y = y['y']
		else:
			error='y is not a field of input y'
			raise Exception(error)

		# Handle both 1D and 2D arrays
		y_len = len(y) if y.ndim == 1 else len(y[0])
		m_len = m.shape[1] if len(m.shape) > 1 else len(m)
		if y_len != m_len:
			error='y and m should have the same length'
			raise Exception(error)

	else:
		m = np.ones((1, y.shape[-1]))
		ystru = {
			'y':y,
			'm':m
		}

	return ystru, y, m
