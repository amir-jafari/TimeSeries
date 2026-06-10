import numpy as np
import math
from scipy.signal import lfilter

from ..basefunctions.makerow import func_makerow as makerow


def func_pmodsim (pmod,e,u=[]):
	"""Simulate the output of a prediction model driven by white noise.

    Implements the system:

    .. math::

        y(t) = G(q)\\,u(t) + H(q)\\,e(t)

    where :math:`G(q)` and :math:`H(q)` are the transfer functions of the
    fitted model, and :math:`e(t)` is a user-supplied white noise sequence.

    Parameters
    ----------
    pmod : pmodel
        Fitted or manually constructed prediction model.
    e : array-like
        White noise input sequence with the same length as ``u``.
    u : array-like, optional
        External input sequence.  Required for ARX, ARMAX, BJTF, and
        regression models.  Default ``[]``.

    Returns
    -------
    y : ndarray
        Simulated output sequence.

    Examples
    --------
    >>> import numpy as np
    >>> import TimeSeriesSRC as ts
    >>> pm = ts.pmodel('arma', nc=[1], nd=[1], diff=[0], per=[])
    >>> pm.c[0] = np.array([-0.5])
    >>> pm.d[0] = np.array([-0.8])
    >>> e = np.random.default_rng(0).standard_normal(200)
    >>> y = ts.pmodsim(pm, e)

    See Also
    --------
    estimate : Fit model parameters from data.
    pmodel   : Define model structure.
	"""

	math_functions = dir(math)
	# preprocessing
	uflag = len(u)>0
	if uflag:
		u = makerow(u)

	upreproc = pmod.upreproc
	pr = len(upreproc)
	if (uflag & pr!=0):
		#if (pr!=1 & pr!= u.shape[0] ):
		#	xerror='rows of upreproc should either equals 1 or the number of inputs. '
		#	raise Exception (xerror)

		if pr == 1:
			pc = upreproc[pr,:]
			for i in range(pc):
				u = eval(upreproc[pr, i], u)

		for i in range(pr):
			if upreproc[i] in math_functions:
				code = 'math.{}(x)'.format(upreproc[i])
			else:
				code = '{}(x)'.format(upreproc[i])

			for j in range(len(u)):
				uj = list(u[j])
				uj = list(map(lambda x: eval(code, globals(), {'x': x}), uj))
				uj = np.array(uj)
				u[j] = uj


	if pmod.type== 'regr':

		# regression model
		ru, cu = u.shape
		udelay=np.zeros((ru,cu))
		idel = 0
		for i in range(ru):
			if len(pmod.delay)==0:
				idel = 0
			else:
				idel = int(pmod.delay[i])

			udelay[i,:] = np.append( np.zeros(idel) , u[i, : (cu - idel)])

		u1 = np.append(np.ones((1, e.shape[1])), udelay, axis=0)
		b = np.array(pmod.b).reshape(1,-1)
		y = b @ u1 + e

	else: # bjtf, arma, armax, arx model

		# Expand the parameter vectors into g and h form
		ng, dg, nh, dh = pmod.getGH()

		num_inputs = len(ng)
		diff = pmod.diff
		period = pmod.period

		# Add the differencing terms into H
		for i in range(diff[0]):
			dh = np.convolve(dh, [1, -1])

		lp = len(period)

		for i in range(lp):
			per = period[i]
			# Fixed: Matlab code [1 zeros(1,per-1) -1] creates [1, 0, 0, ..., 0, -1]
			ddh = np.concatenate([[1], np.zeros(per - 1), [-1]])
			for j in range(diff[i + 1]):
				dh = np.convolve(dh, ddh)

		# Simulate the prediction model
		y = lfilter(nh, dh, e)

		for i in range(num_inputs):
			y = y + lfilter(ng[i], dg[i], u[i,:])

	# post processing
	ypostproc = pmod.ypostproc;
	pc = len(ypostproc)# only one output is possible
	'''
	if (pc>0 & type(ypostproc[0]) == 'list' ):
		xerror='ypostproc should have only one row. '[i]
		raise Exception (xerror)
    '''

	if pc!=0:

		for i in range(pc):
			if ypostproc[i] in math_functions:
				code = 'math.{}(x)'.format(ypostproc[i])
			else:
				code = '{}(x)'.format(ypostproc[i])

			for j in range(len(y)):
				yj = list(y[j])
				yj = list(map(lambda x: eval(code, globals(), {'x': x}), yj))
				yj = np.array(yj)
				y[j] = yj

	return y