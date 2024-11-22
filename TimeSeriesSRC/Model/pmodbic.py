import numpy as np

from ..basefunctions.sdiff import func_sdiff as sdiff

def func_pmodbic (pmod,y,u=[]) :
	'''
		PMODBIC Compute the BIC for prediction model pmod and data u and y.
		
			Syntax
		
			  [BIC] = PMODBIC(PMOD,Y,U)
		
			Description
		
			  PMODBIC computes the BIC (Bayesian Information Criterion) for 
			  the prediction model PMOD.
		
			  PMODBIC(PMOD,Y,U) takes,
			    PMOD - Prediction model.
			    Y    - Prediction model desired outputs.
			    U    - Prediction model inputs.
			  and returns,
			    BIC  - Bayesian Information Criterion.
		
		
			Examples
		
			  Here are a few points from sample y and u sequences:
		
			    y = [-0.1867    0.5173   -3.4980    3.0093   -3.0414    1.5948];
			    u = [-0.4326   -1.6656    0.1253    0.2877   -1.1465    1.1909];
		
			  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
			  model with first order polynomials.
		
			    pmod = newbjtf(1,1,1,1);
		
			  Here the parameters of the model are estimated for up to 50 epochs 
			  to an error goal of 0.01. The BIC of the final model is then computed.
		
			    pmod.estimParam.epochs = 50;
			    pmod.estimParam.goal = 0.01;
			    pmod = estimate(pmod,y,u);
			    bic = pmodbic(pmod,y,u)
			    
			Algorithm
		
			  The BIC is a function of the mean squared prediction
			  error, the number of data points and the number of parameters.
		
			    bic = log(mse) + log(N)*numparams/N;
		
			See also PMODBIC

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

	uflag = (len(u)>0)

	# Difference the sequences

	period = np.append([1], pmod.period)
	diff = pmod.diff

	for i in range(len(diff)):
		d = diff[i]
		if (d!=0):
			if uflag:
				u = sdiff(u, d, period[i])

			y = sdiff(y, d, period[i])

	if uflag:
		yhat = pmod.predict(y, u)
	else:
		yhat = pmod.predict(y)

	e = y - yhat;

	N = e.shape[0] * e.shape[1]

	mse = sum(sum(e ** 2)) / N

	X = pmod.getmX()
	numparams = len(X);

	bic = np.log(mse) + np.log(N) * numparams / N

	return bic

