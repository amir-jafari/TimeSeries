import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

def func_plotacf(acf, maxlag=None, gtitle="Autocorrelation Function", ax=None):
	"""Plot the autocorrelation function as a stem plot.

    Parameters
    ----------
    acf : array-like
        Autocorrelation sequence as returned by :func:`func_xcorr`, shape
        ``(1, 2*maxlag+1)`` or ``(2*maxlag+1,)``.
    maxlag : int or None, optional
        Maximum lag displayed.  If ``None``, inferred from the length of
        ``acf`` as ``(len(acf) - 1) // 2``. Default ``None``.
    gtitle : str, optional
        Plot title. Default ``'Autocorrelation Function'``.
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to draw.  A new figure is created when ``None``.
        Default ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the stem plot.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.signal import lfilter
    >>> from TimeSeriesSRC.basefunctions.xcorr import func_xcorr
    >>> from TimeSeriesSRC.basefunctions.plotacf import func_plotacf
    >>> e = np.random.default_rng(0).standard_normal(500)
    >>> y = lfilter([1], [1, -0.8], e)
    >>> acf = func_xcorr(y, y, 20, 'biased')
    >>> ax = func_plotacf(acf, maxlag=20)

    See Also
    --------
    xcorr   : Compute the autocorrelation sequence.
    uniAnal : Higher-level function that plots ACF, PACF, and GPAC together.
	"""

	# Handle input shape - xcorr returns [1, 2*maxlag+1] array
	acf = np.array(acf).flatten()

	# Infer maxlag from acf length if not provided
	if maxlag is None:
		maxlag = (len(acf) - 1) // 2

	# Create lag axis from -maxlag to +maxlag
	lags = np.arange(-maxlag, maxlag + 1)

	display = False

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
		if ax is None:
			fig, ax = plt.subplots()
			display = True

		# Create stem plot
		markerline, stemlines, baseline = ax.stem(lags, acf, basefmt='k-')

		# Style the stem plot
		plt.setp(stemlines, 'color', 'steelblue', 'linewidth', 1.5)
		plt.setp(markerline, 'color', 'steelblue', 'markersize', 5)
		plt.setp(baseline, 'color', 'black', 'linewidth', 0.5)

		# Add horizontal line at y=0
		ax.axhline(y=0, color='black', linewidth=0.5)

		# Set labels and title
		ax.set_xlabel('Lag')
		ax.set_ylabel('Correlation')
		ax.set_title(gtitle)

		# Set x-axis limits with small padding
		ax.set_xlim(-maxlag - 0.5, maxlag + 0.5)

		plt.tight_layout()

		if display:
			plt.show()

	return ax