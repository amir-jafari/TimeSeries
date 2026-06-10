import numpy as np
import copy
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from ..basefunctions.sepym import func_sepym as sepym
from ..basefunctions.calcindex import func_calcindex
from ..basefunctions.newrec import func_newrec
from ..basefunctions.cliprec import func_cliprec

from .jacobian import func_jacobian as jacobian
from .pmodaic import func_pmodaic as pmodaic
from .pmodbic import func_pmodbic as pmodbic
from .pmodmse import func_pmodmse as pmodmse
from .pmodbic import func_pmodbic as pmodbic

# Global variable to track the figure for plotting
_plotindex_fig = None
_plotindex_line = None

def plotindex(trec, goal, name, epoch):
	"""Update the live training performance plot.

    Parameters
    ----------
    trec : dict
        Training record (``trec['index']`` holds the per-epoch MSE values).
    goal : float
        Target performance value.  A horizontal dashed line is drawn if
        finite and positive.
    name : str
        Algorithm name shown in the figure title.
    epoch : int
        Current epoch index (0-based); only data up to this epoch is plotted.
	"""
	global _plotindex_fig, _plotindex_line

	# Get indices up to current epoch
	ind = slice(0, epoch + 1)
	epochs_arr = np.arange(epoch + 1)
	index_arr = trec['index'][ind]

	# Check if goal should be plotted
	print_goal = np.isfinite(goal)
	plot_goal = np.isfinite(goal) and (goal > 0)

	# Create or update figure
	if _plotindex_fig is None or not plt.fignum_exists(_plotindex_fig.number):
		plt.ion()  # Enable interactive mode
		_plotindex_fig, ax = plt.subplots()
		_plotindex_fig.canvas.manager.set_window_title(f'Training with {name}')
		_plotindex_line, = ax.semilogy(epochs_arr, index_arr, linewidth=2)
		if plot_goal:
			ax.axhline(y=goal, color='r', linestyle='--', linewidth=1, label='Goal')
		ax.set_ylabel('Performance Index')
		ax.set_xlabel('0 Epochs')
	else:
		ax = _plotindex_fig.axes[0]
		_plotindex_line.set_xdata(epochs_arr)
		_plotindex_line.set_ydata(index_arr)
		ax.relim()
		ax.autoscale_view()

	# Update title
	title_str = f'Performance is {trec["index"][epoch]:.6g}'
	if print_goal:
		title_str += f', Goal is {goal:.6g}'
	ax = _plotindex_fig.axes[0]
	ax.set_title(title_str)

	# Update x-label
	if epoch == 0:
		ax.set_xlabel('Zero Epochs')
	elif epoch == 1:
		ax.set_xlabel('One Epoch')
	else:
		ax.set_xlabel(f'{epoch} Epochs')

	# Set x-axis limits
	if epoch > 0:
		ax.set_xlim([0, epoch])

	# Update display
	_plotindex_fig.canvas.draw()
	_plotindex_fig.canvas.flush_events()
	plt.pause(0.01)

def reset_plotindex():
	"""Reset the plot figure for a new training session."""
	global _plotindex_fig, _plotindex_line
	_plotindex_fig = None
	_plotindex_line = None

def func_estimlm (pmod,y=[],u=[],show_plot=True,show_output=True) :
	"""Fit a prediction model with the Levenberg-Marquardt algorithm.

    Minimises the sum of squared one-step-ahead prediction errors using an
    iterative Levenberg-Marquardt update:

    .. math::

        \\Delta X = -(J^T J + \\mu I)^{-1} J^T E

    where :math:`J` is the numerical Jacobian of the prediction errors with
    respect to the model parameters :math:`X`, :math:`E` is the error vector,
    and :math:`\\mu` is the adaptive damping factor.

    Parameters
    ----------
    pmod : pmodel
        Prediction model with initial parameter values and estimation
        hyper-parameters in ``pmod.estimParams``.
    y : array-like or dict
        Desired output sequence.  If a dict, must have keys ``'y'``
        (output array) and ``'m'`` (sample-weight vector).
    u : array-like, optional
        Input sequence.  Default ``[]`` (univariate models).
    show_plot : bool, optional
        Display the live performance-index plot during training.
        Default ``True``.
    show_output : bool, optional
        Print epoch-by-epoch progress to stdout.  Default ``True``.

    Returns
    -------
    pmod : pmodel
        Fitted model with updated parameter values.
    trec : dict
        Training record with keys:

        - ``'index'`` — performance index at each epoch.
        - ``'mu'``    — damping factor :math:`\\mu` at each epoch.
    stat : dict
        Final statistics:

        - ``'sigma'`` — residual variance :math:`\\hat{\\sigma}^2`.
        - ``'stdx'``  — standard deviations of parameter estimates.

    Notes
    -----
    Default estimation hyper-parameters (set on ``pmod.estimParams``):

    ============  =======  ================================================
    Attribute     Default  Meaning
    ============  =======  ================================================
    epochs        10       Maximum iterations
    goal          0        Stop when performance ≤ goal
    mu            0.001    Initial damping factor
    mu_dec        0.1      Multiply mu by this on a successful step
    mu_inc        10       Multiply mu by this on a failed step
    mu_max        1e10     Stop if mu exceeds this value
    min_grad      1e-4     Stop if gradient norm falls below this
    max_time      inf      Wall-clock time limit (seconds)
    delta         1e-7     Finite-difference step for Jacobian
    ============  =======  ================================================

    See Also
    --------
    estimate  : Public interface — calls this function internally.
    func_jacobian : Numerical Jacobian calculation.
	"""


	ystru, y, m = sepym(y)

	# CALCULATION
	# == == == == == =

	# Constants
	uflag = len(u)>0


	this = 'ESTIMLM'

	epochs = pmod.estimParams.epochs
	goal = pmod.estimParams.goal
	min_grad = pmod.estimParams.min_grad
	mu = pmod.estimParams.mu
	mu_inc = pmod.estimParams.mu_inc
	mu_dec = pmod.estimParams.mu_dec
	mu_max = pmod.estimParams.mu_max
	show = pmod.estimParams.show
	max_time = pmod.estimParams.max_time
	delta = pmod.estimParams.delta


	stop = ''
	startTime = time.time()

	# Reset plot for new training session
	if show_plot:
		reset_plotindex()

	X = pmod.getmX()

	numParameters = len(X)

	ii = np.identity(numParameters)

	if uflag:

		index,e = func_calcindex(pmod, ystru, u)

	else:
		index,e = func_calcindex(pmod, ystru)

	trec = func_newrec(epochs,'index', 'mu' )

	repeat = 0
	jj = []

	# Train
	for epoch in range(epochs):
	#for epoch in [0,1]:

		# Jacobian

		if uflag:
			je, jj, normgX = jacobian(pmod,delta, ystru, u)
		else:
			je, jj, normgX = jacobian(pmod,delta, ystru)

		# Training Record
		trec['index'][epoch] = index
		trec['mu'][epoch] = mu

		# Stopping Criteria
		currentTime = time.time() - startTime

		if (index <= goal):
			stop = 'Performance goal met.'
		elif (epoch == (epochs-1)):
			stop = 'Maximum epoch reached, Performance goal was not met.'
		elif (currentTime > max_time):
			stop = 'Maximum time elapsed, performance goal was not met.'
		elif (normgX < min_grad):
			if show_output:
				print(normgX , min_grad )
			stop = 'Minimum gradient reached, performance goal was not met.'
		elif (mu > mu_max):
			stop = 'Maximum MU reached, performance goal was not met.'

		# Progress
		if (show > 0 and (epoch % show) == 0) | (len(stop) > 0):

			if epochs>0:
				message ='Epoch {}/{}'.format(epoch, epochs)
				message = message + ' ' + 'Time {}'.format(currentTime)

			if goal>-1:
				message = message + ' {} {}/{}'.format(pmod.indexFcn.upper(), index, goal)

			if min_grad>0:
				message = message + ' Gradient {}/{}'.format(normgX, min_grad)

			if mu_max>0:
				message = message + ' mu {}/{}'.format(mu, mu_max)

			if show_output:
				print(message)

			if show_plot:
				plotindex(trec, goal, this, epoch)

			if show_output and len(stop)>0:
				print('{}, {}\n\n'.format( this, stop))

			# Stop when criteria indicate its time
		if len(stop) > 0:
			break

		# Levenberg Marquardt

		repeat = 0
		while (mu <= mu_max):

			repeat = repeat + 1
			A = jj + ii * mu
			#dX = np.linalg.solve(A, je)
			dX = np.linalg.lstsq((jj + ii * mu), je,rcond=None)[0]

			dX = dX.T[0] *-1
			X2 = X + dX
			pmod2 = copy.deepcopy(pmod)
			pmod2.setmX(X2)

			if uflag:
				index2, e1 = func_calcindex(pmod2, ystru, u)
			else:
				index2, e1 = func_calcindex(pmod2, ystru)

			if (index2 < index):
				X = X2
				pmod = copy.deepcopy(pmod2);
				index = index2;
				mu = mu * mu_dec;
				break

			mu = mu * mu_inc;

		# End of Loop Mu
		if (mu < 1e-15):
			mu = 1e-15


	if uflag:
		e = y - pmod.predict(y, u)
	else:
		e = y - pmod.predict(y)

	x = e.shape
	prod = 1
	for i in range(len(x)):
		prod = prod * x[i]

	sigma = sum(sum(e ** 2)) / prod


	if np.linalg.det(jj) !=0:
		stdx = np.sqrt(sigma * np.diag(np.linalg.inv(jj)))
	else:
		print('Error no Inv Matrix')
		stdx = [0,0]

	stat = {
		'sigma': sigma,
		'stdx': stdx
	}

	# Finish
	trec = func_cliprec( trec, epoch)


	return pmod, trec, stat

##----------------
