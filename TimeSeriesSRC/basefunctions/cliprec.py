def func_cliprec (tr,epochs) :
	"""Trim a training record to the last completed epoch.

    Parameters
    ----------
    tr : dict
        Training record as returned by :func:`func_newrec`.  Each value is
        a list of length ``max_epochs``.
    epochs : int
        Index of the final completed epoch (0-based).

    Returns
    -------
    tr : dict
        Same dict with each list sliced to ``epochs`` entries.

    See Also
    --------
    func_newrec  : Allocates a training record.
    func_estimlm : Calls this at the end of optimisation.
	"""

	indexes = list(range(epochs + 1))
	for name in tr:
		tr[name]  = tr[name][indexes[0]:indexes[-1]]

	return tr
