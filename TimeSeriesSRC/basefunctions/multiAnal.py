import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from .gpac import func_gpac
from .impest import func_impest
from .plotgpac import func_plotgpac
from .xcorr import func_xcorr
from .makerow import func_makerow

def func_multiAnal (u,y,nng=5,ndg=5,nnh=5,ndh=5,lg=12,lh=12) :
	"""Multivariate analysis: impulse response, residual ACF, and GPAC tables.

    Pre-whitens the input ``u`` with a BIC-selected ARMA model, estimates the
    impulse response between ``u`` and ``y`` via the Wiener-Hopf equations,
    and computes GPAC tables for both the G (transfer function) and H (noise)
    components.  Produces plots of the impulse response, residual ACF, and the
    two GPAC arrays.

    Parameters
    ----------
    u : array-like
        1-D input (exogenous) sequence.
    y : array-like
        1-D output sequence (same length as ``u``).
    nng : int, optional
        Maximum numerator order for the G GPAC table. Default 5.
    ndg : int, optional
        Maximum denominator order for the G GPAC table. Default 5.
    nnh : int, optional
        Maximum numerator order for the H GPAC table. Default 5.
    ndh : int, optional
        Maximum denominator order for the H GPAC table. Default 5.
    lg : int, optional
        Number of impulse-response lags to compute. Default 12.
    lh : int, optional
        Number of residual-ACF lags to compute. Default 12.

    Returns
    -------
    g : ndarray, shape (lg+1,)
        Estimated impulse response from ``u`` to ``y``.
    rv : ndarray, shape (1, 2*lh+1)
        Residual autocorrelation function.
    g_gpac : ndarray, shape (nng, ndg)
        GPAC table for the G transfer function.
    h_gpac : ndarray, shape (nnh, ndh)
        GPAC table for the H noise model.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.basefunctions.multiAnal import func_multiAnal
    >>> rng = np.random.default_rng(0)
    >>> u = rng.standard_normal(300)
    >>> e = rng.standard_normal(300) * 0.2
    >>> y = lfilter([1], [1, 0.5], u) + lfilter([1], [1, -0.8], e)
    >>> g, rv, g_gpac, h_gpac = func_multiAnal(u, y, lg=10)

    See Also
    --------
    uniAnal : Univariate ACF/PACF/GPAC analysis.
    multiChi : Chi-square test for residual/input independence.
    impest : Lower-level impulse-response estimator.
	"""

	u = func_makerow(u)
	y = func_makerow(y)

	u_1d = u[0]
	y_1d = y[0]

	# Prewhiten u with a BIC-selected ARMA model; apply same filter to y.
	# This replaces the ill-conditioned direct Wiener-Hopf for correlated inputs.
	from ..Model.selpmod import func_selpmod as selpmod
	arma_spec = {
		'models': [{
			'type': 'arma',
			'nc': [0, 1, 2, 3],
			'nd': [1, 2, 3],
			'diff': [0]
		}]
	}
	print('Prewhitening input for impulse response estimation, please wait...')
	estpmodu = selpmod(arma_spec, u_1d)
	bicmod = estpmodu['arma']['bicmod']

	c_coef = np.asarray(bicmod.c[0]).ravel() if bicmod.c and len(bicmod.c) > 0 else np.array([])
	d_coef = np.asarray(bicmod.d[0]).ravel() if bicmod.d and len(bicmod.d) > 0 else np.array([])
	Rq = np.concatenate([[1.0], c_coef])   # C polynomial (MA)
	Sq = np.concatenate([[1.0], d_coef])   # D polynomial (AR)

	al = lfilter(Sq, Rq, u_1d)   # prewhitened input α
	be = lfilter(Sq, Rq, y_1d)   # filtered output β

	# K = nng + ndg + 1
	K = lg
	g = func_impest(al.reshape(1, -1), be.reshape(1, -1), K)
	index = range(0, K+1)

	# Get figure if it	exists, or create	new	figure
	me = 'Impulse Response and Residual ACF'

	fig, ax= plt.subplots(2)
	fig.suptitle(me)

	ax[0].stem(index,g)
	ax[0].set_title('Impulse Response')
	ax[0].set_xlabel('Lag')
	xlim = ax[0].get_xlim()
	ax[0].plot([xlim[0], xlim[1]], [0, 0], 'k')

	# Get figure if it exists, or create new figure

	me = 'G & H GPAC Arrays'

	# g_aug must have half-length >= nng+ndg-1 for gpac to index safely.
	# If lg < nng+ndg-1 we zero-pad g (equivalent to assuming g[k]=0 for k>K).
	K_gpac = max(K, nng + ndg - 1)
	g_padded = np.concatenate([g, np.zeros(K_gpac - K)]) if K_gpac > K else g
	g_aug = np.hstack((np.zeros(K_gpac), g_padded))
	g_gpac = func_gpac(g_aug, nng, ndg)


	y1 = np.convolve(g, u[0])

	#yy1 = filter(g, 1, u)  # is not used in the original code

	y1 = y1[0:len(y[0])];
	v = y[0] - y1

	L = nnh + ndh + 1
	rv = func_xcorr(v, v, L, 'biased')
	lag = range(-L,L+1)
	confint = rv[0,L] * 2 / np.sqrt(len(v))

	ax[1].stem(lag, rv[0])
	ax[1].set_title('Autocorrelation Function for V')
	ax[1].set_xlabel('Lag')

	xlim = ax[1].get_xlim()

	ax[1].plot([xlim[0],xlim[1]], [0, 0], 'k')
	ax[1].plot([xlim[0],xlim[1]], [confint, confint], ':r')
	ax[1].plot([xlim[0],xlim[1]], [-confint, -confint], ':r')

	plt.tight_layout()
	plt.show()
	plt.close()

	h_gpac = func_gpac(rv, nnh, ndh)





	fig, ax= plt.subplots(2)

	ax[0] = func_plotgpac(g_gpac, 'GPAC for G Transfer Function', ax[0])
	ax[1] = func_plotgpac(h_gpac, 'GPAC for H Transfer Function', ax[1])

	plt.tight_layout()
	plt.show()
	plt.close()

	return g,rv,g_gpac,h_gpac

