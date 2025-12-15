import numpy as np
import copy
import time
from scipy.sparse import csr_matrix

from ..basefunctions.sepym import func_sepym as sepym
from ..basefunctions.calcindex import func_calcindex
from ..basefunctions.newrec import func_newrec
from ..basefunctions.cliprec import func_cliprec

from .jacobian import func_jacobian as jacobian
from .pmodaic import func_pmodaic as pmodaic
from .pmodbic import func_pmodbic as pmodbic
from .pmodmse import func_pmodmse as pmodmse
from .pmodbic import func_pmodbic as pmodbic

def func_estimlm (pmod,y=[],u=[]) :
	'''
		ESTIMLM Levenberg-Marquardt algorithm for prediction model fitting.
		
			Syntax
			
			  [pmod,trec,stat]=estimlm(pmod,y,u)
		
			Description
		
			  ESTIMLM is an algorithm for estimating the parameters of a prediction
			  model by fitting a data set according to Levenberg-Marquardt optimization.
		
			  ESTIMLM(PMOD,Y,U) takes these inputs,
			    PMOD - Prediction model.
			    Y    - Prediction model desired outputs.  Y may or may not be a
			           structure.  If Y is a structure, then Y.Y is the prediction 
			           model desired outputs, and Y.M is the vector
			           containing the weighting factors for each error.  
			    U    - Prediction model inputs.
			  and returns,
			    PMOD - New prediction model.
			    TREC - Training record (index).
			            TREC.index - Training performance index.
			            TREC.mu    - Adaptive mu value.
			    STAT - Statistics for final model.
		               STAT.sigma - Residual variance.
		               STAT.stdx - Vector of standard deviations of parameter estimates.
		
			  Training occurs according to the ESTIMLM's estimation parameters
			  shown here with their default values:
			    pmod.estimParam.epochs      10  Maximum number of epochs 
			    pmod.estimParam.goal         0  Performance index goal
			    pmod.estimParam.max_fail     5  Maximum validation failures
			    pmod.estimParam.mem_reduc    1  Factor to use for memory/speed trade off.
			    pmod.estimParam.mu       0.001  Initial mu value
			    pmod.estimParam.mu_dec     0.1  mu decrement value
			    pmod.estimParam.mu_inc      10  mu increment value
			    pmod.estimParam.mu_max    1e10  Maximum mu value
			    pmod.estimParam.min_grad  1e-4  Minimum performance gradient
			    pmod.estimParam.show        25  Epochs between displays (NaN for no displays)
			    pmod.estimParam.time       inf  Maximum time to train in seconds
			    pmod.estimParam.delta     1e-7  Increment to use in computing Jacobian
		
			Algorithm
		
			  The Jacobian jX of error with respect to the model parameters X
			  Is computed numerically.  Each parameter is adjusted according to 
			  the Levenberg-Marquardt algorithm,
		
			    jj = jX * jX
			    je = jX * E
			    dX = -(jj+I*mu) \ je
		
			  where E is all errors and I is the identity matrix.
		
			  The adaptive value MU is increased by MU_INC until the change above
			  results in a reduced performance value.  The change is then made to
			  the network and mu is decreased by MU_DEC.
		
		
			  Estimation stops when any of these conditions occurs:
			  1) The maximum number of EPOCHS (repetitions) is reached.
			  2) The maximum amount of TIME has been exceeded.
			  3) Performance has been minimized to the GOAL.
			  4) The performance gradient falls below MINGRAD.
			  5) MU exceeds MU_MAX.
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

		 FUNCTION INFO
		 =============

	'''


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

			print(message)

			# plotindex(trec, goal, this, epoch)

			if len(stop)>0:
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
