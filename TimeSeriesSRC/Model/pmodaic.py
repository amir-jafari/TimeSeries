import numpy as np

from ..basefunctions.sdiff import func_sdiff as sdiff

def func_pmodaic (pmod,y,u=[]) :
	"""Compute the Akaike Information Criterion (AIC) for a fitted model.

    .. math::

        \\text{AIC} = \\ln(\\text{MSE}) + \\frac{2 k}{N}

    where :math:`k` is the number of free parameters and :math:`N` is the
    number of data points.  Lower AIC indicates a better trade-off between
    fit and model complexity.

    Parameters
    ----------
    pmod : pmodel
        Fitted prediction model.
    y : array-like
        Desired output sequence.
    u : array-like, optional
        Input sequence. Default ``[]``.

    Returns
    -------
    aic : float
        Akaike Information Criterion.

    Examples
    --------
    >>> import pathlib, pandas as pd
    >>> import TimeSeriesSRC as ts
    >>> data_dir = pathlib.Path(ts.__file__).parent / 'TestData'
    >>> y = pd.read_csv(data_dir / 'Series_A_Chemical_Concentration.csv').values.flatten()
    >>> pm = ts.pmodel('arma', nc=[2], nd=[1], diff=[0], per=[])
    >>> pm_est, trec, stat = ts.estimate(pm, y, show_plot=False, show_output=False)
    >>> aic = ts.pmodaic(pm_est, y)

    See Also
    --------
    pmodbic : Bayesian Information Criterion (penalises complexity more strongly).
    selpmod : Automatic model selection using AIC/BIC grid search.
	"""

	uflag = (len(u) > 0)

	period = np.append([1], pmod.period)
	diff = pmod.diff
	for i in range(len(diff)):
		d = diff[i]
		if d != 0:
			if uflag:
				u = sdiff(u, d, period[i])
			y = sdiff(y, d, period[i])

	if uflag:
		yhat = pmod.predict( y, u)
	else:
		yhat = pmod.predict(y)


	e = y - yhat;

	N = e.size

	mse = np.sum(e ** 2) / N

	X = pmod.getmX()
	numparams = len(X)

	aic = np.log(mse) + 2 * numparams / N

	return aic
