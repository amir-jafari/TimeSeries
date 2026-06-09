import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

def func_plotacf(acf, maxlag=None, gtitle="Autocorrelation Function", ax=None):
	'''
		PLOTACF Plot the autocorrelation function using a stem plot.

			Syntax

			  PLOTACF(ACF,MAXLAG,GTITLE)

			Description

			  PLOTACF(ACF,MAXLAG,GTITLE) takes these inputs,
			    ACF    - Autocorrelation array from xcorr function.
			    MAXLAG - Maximum lag value (optional, inferred from ACF length).
			    GTITLE - Title for the plot.
			  and displays the autocorrelation function as a stem plot.

			Examples

			  This code generates an autoregressive sequence.

			    e = randn(1,2000);
			    y = filter(1,[1 -.8],e);

			  The following command generates the autocorrelation function.
			  The acf will be computed from lag -20 to lag 20.

			    acf = xcorr(y,y,20,'unbiased');
			    plotacf(acf,20);


		 Martin Hagan, 2-4-26
		 $Revision: 1.0 $ $Date: 04-Feb-2026 $

	'''

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