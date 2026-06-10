import numpy as np

def func_gcombvec (a1, *argv) :
	"""Return all column-vector combinations of two or more input matrices.

    Given matrices ``a1`` (shape ``r1 × c1``) and ``a2`` (shape ``r2 × c2``),
    returns a matrix of shape ``(r1+r2) × (c1*c2)`` whose columns are every
    possible concatenation of a column from ``a1`` with a column from ``a2``.
    Additional positional arguments are combined recursively.

    Parameters
    ----------
    a1 : ndarray, shape (r1, c1)
        First matrix of column vectors.
    *argv : ndarray
        One or more additional matrices to combine with ``a1``.

    Returns
    -------
    y : ndarray
        Matrix of all column combinations.

    Examples
    --------
    >>> import numpy as np
    >>> from TimeSeriesSRC.basefunctions.gcombvec import func_gcombvec
    >>> a1 = np.array([[7], [9]])
    >>> a2 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> func_gcombvec(a1, a2)
    array([[7., 7., 7.],
           [9., 9., 9.],
           [1., 2., 3.],
           [4., 5., 6.]])

    See Also
    --------
    func_selpmod : Uses this to enumerate model-order grids.
	"""

   #=============================================================
	def nncpy(m,n):
		return np.repeat(m, n, axis=1)

	#==============================================================

	def nncpyi(m, n):

		mr, mc = m.shape

		b = np.zeros((mr * n, mc))

		ind = np.array([i for i in range(mr)])

		for i in range(n):
			ii = i * mr
			b[ind + ii,:] = m

		b = b.reshape( mr, n * mc)

		return b

	narg = len(argv)

	if narg == 0:
		y = a1

	elif narg == 1:

		a2 = argv[0]

		len1 = a1.shape[1]
		len2 = a2.shape[1]

		x1 = nncpy(a1, len2)
		x2 = nncpyi(a2, len1)

		y = np.vstack((x1 ,x2 ))

	else:
		y = func_gcombvec(func_gcombvec(a1, argv[0]), *argv[1:])

	return y
