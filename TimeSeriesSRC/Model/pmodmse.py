import numpy as np

from ..basefunctions.makerow import  func_makerow as makerow
from ..basefunctions.sepym import func_sepym as sepym

def func_pmodmse (pmod,y,u=[]):
	"""Compute the mean squared prediction error (MSE) for a fitted model.

    Parameters
    ----------
    pmod : pmodel
        Fitted prediction model.
    y : array-like
        Desired output sequence.
    u : array-like, optional
        Input sequence (required for ARX, ARMAX, BJTF models).
        Default ``[]`` (univariate models).

    Returns
    -------
    mse : float
        Mean squared one-step-ahead prediction error.
    e : ndarray
        Prediction error sequence ``y - yhat``.

    Examples
    --------
    >>> import pathlib, pandas as pd
    >>> import TimeSeriesSRC as ts
    >>> data_dir = pathlib.Path(ts.__file__).parent / 'TestData'
    >>> y = pd.read_csv(data_dir / 'Series_A_Chemical_Concentration.csv').values.flatten()
    >>> pm = ts.pmodel('arma', nc=[2], nd=[1], diff=[0], per=[])
    >>> pm_est, trec, stat = ts.estimate(pm, y, show_plot=False, show_output=False)
    >>> mse, e = ts.pmodmse(pm_est, y)

    See Also
    --------
    pmodaic : Akaike Information Criterion.
    pmodbic : Bayesian Information Criterion.
	"""

	uflag = len(u) > 0

	ystru, y, m = sepym(y);


	if uflag:
		yhat = pmod.predict(y, u)

	else:
		yhat = pmod.predict(y)

	e = y - yhat

	m = makerow(m)

	# Overflow is expected when LM tries unstable filter parameters; suppress it and
	# cap at a large finite value so the optimizer can still rank trial steps correctly.
	with np.errstate(over='ignore', invalid='ignore'):
		res = e * m * e

	mse = np.sum(res) / e.size
	if not np.isfinite(mse):
		mse = np.finfo(np.float64).max / 2

	return mse, e
