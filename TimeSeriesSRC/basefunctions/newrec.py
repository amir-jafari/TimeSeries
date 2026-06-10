import numpy as np

def func_newrec (epochs, *argv) :
	"""Allocate a training record with pre-zeroed fields.

    Parameters
    ----------
    epochs : int
        Number of epochs to pre-allocate.
    *argv : str
        Field names for the record (e.g. ``'index'``, ``'mu'``).

    Returns
    -------
    tr : dict
        Dict mapping each field name to a list of ``epochs`` zeros.

    Examples
    --------
    >>> from TimeSeriesSRC.basefunctions.newrec import func_newrec
    >>> tr = func_newrec(5, 'index', 'mu')
    >>> list(tr.keys())
    ['index', 'mu']
    >>> len(tr['index'])
    5

    See Also
    --------
    func_cliprec : Trims the record to the actual number of epochs used.
	"""

	names = argv

	tr = {}
	for i in range(len(names)):
		tr[names[i]] = [0 for _ in range(epochs)]  # Create a NEW list for each field

	return tr