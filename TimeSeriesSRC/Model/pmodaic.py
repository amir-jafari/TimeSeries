import numpy as np

from ..basefunctions.sdiff import func_sdiff as sdiff

def func_pmodaic (pmod,y,u=[]) :
	'''
		PMODAIC Compute the AIC for prediction model pmod and data u and y.
		
			Syntax
		
			  [AIC] = PMODAIC(PMOD,Y,U)
		
			Description
		
			  PMODAIC computes the AIC (Akaike Information Criterion) for 
			  the prediction model PMOD.
		
			  PMODAIC(PMOD,Y,U) takes,
			    PMOD - Prediction model.
			    Y    - Prediction model desired outputs.
			    U    - Prediction model inputs.
			  and returns,
			    AIC  - Akaike Information Criterion.
		
		
			Examples
		
			  Here are a few points from sample y and u sequences:
		
			    y = [-0.1867    0.5173   -3.4980    3.0093   -3.0414    1.5948];
			    u = [-0.4326   -1.6656    0.1253    0.2877   -1.1465    1.1909];
		
			  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
			  model with first order polynomials.
		
			    pmod = newbjtf(1,1,1,1);
		
			  Here the parameters of the model are estimated for up to 50 epochs 
			  to an error goal of 0.01. The AIC of the final model is then computed.
		
			    pmod.estimParam.epochs = 50;
			    pmod.estimParam.goal = 0.01;
			    pmod = estimate(pmod,y,u);
			    aic = pmodaic(pmod,y,u)
			    
			Algorithm
		
			  The AIC is a function of the mean squared prediction
			  error, the number of data points and the number of parameters.
		
			    aic = log(mse) + 2*numparams/N
		
			See also PMODBIC

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

	uflag = (len(u) > 0)

	period = np.append([1], pmod.period)
	diff = pmod.diff
	for i in range(len(diff)):
		d = diff[i]
		if d != 0:
			if uflag:
				u = sdiff(u, d, period[i])
			y = sdiff(y, d, period[i])

	if uflag:
		yhat = pmod.predict( y, u)
	else:
		yhat = pmod.predict(y)


	e = y - yhat;

	N = e.size

	mse = np.sum(e ** 2) / N

	X = pmod.getmX()
	numparams = len(X)

	aic = np.log(mse) + 2 * numparams / N

	return aic
