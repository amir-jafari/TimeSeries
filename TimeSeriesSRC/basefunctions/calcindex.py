from ..Model.pmodaic import func_pmodaic as pmodaic
from ..Model.pmodbic import func_pmodbic as pmodbic
from ..Model.pmodmse import func_pmodmse as pmodmse
from ..Model.pmodsim import func_pmodsim as pmodsim

def func_calcindex (pmod,y,u=[]):
	"""Evaluate the performance index and prediction errors for a model.

    Dispatches to the criterion function named by ``pmod.indexFcn``
    (``'pmodmse'``, ``'pmodaic'``, ``'pmodbic'``, or ``'pmodsim'``).

    Parameters
    ----------
    pmod : pmodel
        Prediction model.  ``pmod.indexFcn`` selects the criterion.
    y : array-like or dict
        Desired output sequence.  If a dict, must contain keys ``'y'``
        and ``'m'`` (sample weights).
    u : array-like, optional
        Input sequence.  Default ``[]``.

    Returns
    -------
    index : float
        Scalar performance index (e.g. MSE, AIC, or BIC).
    e : ndarray
        One-step-ahead prediction error sequence.

    See Also
    --------
    func_pmodmse : Mean squared prediction error.
    func_pmodaic : Akaike Information Criterion.
    func_pmodbic : Bayesian Information Criterion.
    func_estimlm : Calls this inside the Levenberg-Marquardt loop.
	"""

	indexFcn = pmod.indexFcn


	if len(u) > 0:

		if indexFcn == 'pmodsim':

			index, e = pmodsim(pmod, y, u)

		elif indexFcn == 'pmodmse':

			index, e = pmodmse(pmod, y, u)

		elif indexFcn == 'pmodbic':

			index, e = pmodbic(pmod, y, u)

		elif indexFcn == 'pmodaic':

			index, e = pmodaic(pmod,y, u)

	else:

		if indexFcn == 'pmodsim':
			index, e = pmodsim(pmod, y)
		elif indexFcn == 'pmodmse':
			index, e = pmodmse(pmod, y)
		elif indexFcn == 'pmodbic':
			index, e = pmodbic(pmod, y)
		elif indexFcn == 'pmodaic':
			index, e = pmodaic(pmod, y)

	return index, e