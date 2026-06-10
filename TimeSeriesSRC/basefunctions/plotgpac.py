import matplotlib.pyplot as plt
# importing the style package
from matplotlib import style
import numpy as np
import math

def func_plotgpac (gpac,gtitle="Gpac Array",  ax=None) :
	"""Visualize a GPAC table as a grid of colored squares.

    Each cell's **area** encodes the magnitude of the GPAC value and its
    **color** encodes the sign: green for positive, red for negative.

    Parameters
    ----------
    gpac : ndarray, shape (nrows, ncols)
        GPAC table as returned by :func:`func_gpac`.
    gtitle : str, optional
        Plot title. Default ``'Gpac Array'``.
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to draw.  If ``None`` a new figure is created and
        displayed. Default ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    >>> from TimeSeriesSRC.basefunctions.gpac import func_gpac
    >>> from TimeSeriesSRC.basefunctions.plotgpac import func_plotgpac
    >>> e = np.random.default_rng(0).standard_normal(500)
    >>> y = lfilter([1], [1, -0.8], e)
    >>> acf = func_xcorr(y, y, 25, 'biased')
    >>> G = func_gpac(acf, 5, 5)
    >>> ax = func_plotgpac(G, 'My GPAC')

    See Also
    --------
    gpac    : Compute the GPAC table.
    uniAnal : Calls this function automatically.
	"""

	limit = 5
	ind = np.where(gpac > limit)
	if (len(ind)>0):
		gpac[ind] = limit

	ind = np.where(gpac < (limit * -1))
	if (len(ind)>0):
		gpac[ind] = limit * -1

	# max_m = max(max(abs(gpac)));
	max_m = limit
	min_m = max_m / 300

	# DEFINE BOX EDGES
	xn1 = np.array([-1.,-1.,1.]) * 0.5;
	xn2 = np.array([ 1., 1., -1.]) * 0.5;

	yn1 = np.array([1.,-1.,-1.]) * 0.5;
	yn2 = np.array([-1., 1. ,1.]) * 0.5;

	# DEFINE POSITIVE BOX
	xn = np.array([-1., -1., 1., 1., -1.]) * 0.5;
	yn = np.array([-1., 1., 1., -1., -1.]) * 0.5;

	# DEFINE POSITIVE BOX
	xp = np.append(xn,np.array([-1.,1.,1.,0., 0.]) * 0.5)
	yp = np.append(yn,np.array([ 0.,0.,1.,1.,-1.]) * 0.5)

	S, R = gpac.shape

	display=False

	# Try to use seaborn style if available, otherwise use default
	try:
		style_context = plt.style.context('seaborn-v0_8-darkgrid')
	except:
		try:
			style_context = plt.style.context('seaborn')
		except:
			# If seaborn styles not available, use a null context
			import contextlib
			style_context = contextlib.nullcontext()

	with style_context:
		if ax == None:
			fig, ax = plt.subplots()
			display = True
		xxlim = np.array([0,R]) + 0.5
		xylim = np.array([0,S]) - 0.5
		ax.xlim = xxlim
		ax.ylim = xylim

		new_x = range(math.floor(min(xxlim)), math.ceil(max(xxlim)) + 1)
		new_y = range(math.floor(min(xylim)), math.ceil(max(xylim)) + 1)

		ax.set_xticks(new_x)
		ax.set_yticks(new_y)
		ax.invert_yaxis()

		for i in range(S):
			i1 = i;
			for j in range(R):
				xval = (np.abs(gpac[i, j]) - min_m) / max_m
				if xval >= 0:
					m = round(np.sqrt(xval),2)
					m = min(m, max_m) * 0.95

					if gpac[i, j] >= 0:
						ax.fill(xn * m + j+1, yn * m + i1, color='green')
						ax.plot(xn1 * m + j+1, yn1 * m + i1, 'w' ,xn2 * m + j+1, yn2 * m + i1, 'k', linewidth=3)
					elif gpac[i, j] < 0:
						ax.fill(xn * m + j+1, yn * m + i1,color= 'red');
						ax.plot(xn1 * m + j+1, yn1 * m + i1, 'k', xn2 * m + j+1, yn2 * m + i1, 'w',linewidth=3);


		ax.plot(np.array([0, R, R, 0, 0]) + 0.5, np.array([0, 0, S, S, 0]) - 0.5, 'w');
		ax.set_xlabel('Denominator Order')
		ax.set_ylabel('Numerator Order')
		ax.set_title(gtitle)


		plt.tight_layout()

		if display:
			plt.tight_layout()
			plt.show()

		return ax