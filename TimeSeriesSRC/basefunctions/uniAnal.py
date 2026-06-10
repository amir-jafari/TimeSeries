import numpy as np
import matplotlib.pyplot as plt
from .sdiff import func_sdiff
from .xcorr import func_xcorr
from .plotgpac import func_plotgpac
from .parcor import func_parcor
from .gpac import func_gpac
from .makerow import func_makerow

def func_uniAnal (y,na=20,nump=10,nrg=5,ncg=0,diff=[0],per=[],perdsp=1) :
	"""Compute and plot the ACF, PACF, and GPAC for a univariate time series.

    Differences the series as requested, computes the autocorrelation (ACF),
    partial autocorrelation (PACF), and generalized partial autocorrelation
    (GPAC) functions, and produces stem/GPAC plots for order identification.

    Parameters
    ----------
    y : array-like
        1-D time series.
    na : int, optional
        Maximum lag for ACF; lags range from ``-na`` to ``+na``. Default 20.
    nump : int, optional
        Number of PACF terms to compute (lags 1 .. nump). Default 10.
    nrg : int, optional
        Number of GPAC rows (numerator orders). Default 5.
    ncg : int, optional
        Number of GPAC columns (denominator orders); 0 sets it equal to
        ``nrg``. Default 0.
    diff : list of int, optional
        Differencing orders to apply before analysis.
        Must satisfy ``len(diff) == len(per) + 1``. Default ``[0]``.
    per : list of int, optional
        Seasonal periods.  Used with ``diff`` — must have one fewer element.
        Default ``[]``.
    perdsp : int, optional
        Display period — ACF and PACF are sampled at every ``perdsp``-th lag.
        Default 1.

    Returns
    -------
    yacf : ndarray, shape (1, 2*na+1)
        Biased ACF from lag ``-na`` to ``+na``.
    ypacf : ndarray, shape (1, nump)
        Partial ACF from lag 1 to ``nump``.
    ygpac : ndarray, shape (nrg, ncg)
        GPAC table (row = numerator order, column = denominator order).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.basefunctions.uniAnal import func_uniAnal
    >>> e = np.random.default_rng(0).standard_normal(500)
    >>> y = lfilter([1], [1, -0.8], e)
    >>> yacf, ypacf, ygpac = func_uniAnal(y, na=20, nump=10, nrg=5)

    See Also
    --------
    multiAnal : Analysis for input-output (transfer function) data.
    uniChi : Box-Pierce chi-square whiteness test on residuals.
	"""


	y = func_makerow(y)

	if ncg == 0:
		ncg = nrg

	numpts = len(y[0])

	if (len(per) + 1) != len(diff):
		error='per must have one less term than diff.'
		raise Exception(error)

	# Difference the sequence

	period = np.append([1],diff)

	for i in range(len(diff)):

		d = diff[i]
		if (d!=0):
			y = func_sdiff(y, d, period[i]);

	tot = max([nump + 1, nrg + ncg])

	if na < tot:
		L = tot * perdsp
	else:
		L = na * perdsp

	if L > numpts:
		error='Not enough data to compute the acf for sufficient lags.'
		raise Exception(error)


	yacf = func_xcorr(y, y, L, 'biased');

	# Select data only at multiples of the display period

	xlen = len(yacf[0])
	if perdsp > 1:
		m =yacf[0,(L - perdsp):: -perdsp]
		n =yacf[0,L :xlen:perdsp]
		yacf =  np.append(m[::-1],n)
		yacf = yacf.reshape(1,-1)
		# yacf = [fliplr(yacf((L + 1 - perdsp):-perdsp: 1)) yacf((L + 1): perdsp:xlen)];


	# Calculate pacf and gpac

	ypacf, phi, sigma = func_parcor(yacf[0], nump);
	ygpac = func_gpac(yacf, nrg, ncg);


	# Convert acf to proper length for plotting

	xlen = len(yacf[0]);

	L = round((xlen - 1) / 2);


	if L != na:
		dif = L - na
		yacf = yacf[0,dif:xlen - dif]

	lag = np.arange(-na,na+1)

	confint = yacf[0,na] * 2 / np.sqrt(numpts)

	# Getfigure if it exists, or create new figure

	me = 'ACF and PACF'

	fig, (ax1, ax2) = plt.subplots(2)
	fig.suptitle(me)
	ax1.stem(lag, yacf[0])


	ax1.set_title('Autocorrelation Function')
	ax1.set_xlabel('Lag')
	xlim = ax1.get_xlim()
	ax1.plot([xlim[0], xlim[1]], [0 ,0], 'k');
	ax1.plot([xlim[0], xlim[1]], [ confint, confint], ':r');
	ax1.plot([xlim[0], xlim[1]], [-confint,-confint], ':r');

	ind = [i for i in range(1,nump+1)]
	ax2.stem(ind,ypacf[0])
	ax2.set_title('Partial Autocorrelation Function')
	ax2.set_xlabel('Lag')
	xlim = ax2.get_xlim()
	ax2.plot([xlim[0], xlim[1]], [0 ,0], 'k');

	# Get figure if it exists, or create new figure

	plt.tight_layout()
	plt.show()

	func_plotgpac(ygpac, 'GPAC Array')

	return yacf, ypacf, ygpac
