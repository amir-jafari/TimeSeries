import numpy as np

from ..basefunctions.makerow import  func_makerow as makerow
from ..basefunctions.sepym import func_sepym as sepym

def func_pmodmse (pmod,y,u=[]):
	'''
		PMODMSE Find the error and the mean square error for a prediction model.
		
			Synopsis
		
			  [mse,e]=pmodmse(pmod,y,u)
		
			Description
		
			  This function calculates the performance index of the
			  prediction model for a given set of inputs and
			  desired outputs.
		
			  [MSE,E]=pmodmse(PMOD,Y,U) takes,
			    PMOD - Prediction model.
			    Y    - Desired outputs of the prediction model. Y may or may
			           not be a structure.  If Y is a structure, then Y.Y  
			           is the prediction model desired outputs, and Y.M is
			           the vector containing the weighting factors
			           for each error.  
			    U    - Inputs to the prediction model.
			  and returns,
			    MSE   - Prediction model mean square error.
			    E     - Prediction errors.
		
			Examples
		
			  Here we create a Box and Jenkins Transfer Function
			  prediction model with each polynomial first order.
		
			    pmod = newbjtf(1,1,1,1);
		
			  Here is a single input sequence u with 5 timesteps.
		
			    u = [0 0.1 0.3 0.6 0.4];
		
			  Here we define the desired predictions for 
			  each of the five time steps.
			  
			    y = [0.1 0.3 0.5 0.8 0.5];
		
			  Here we calculate the prediction model's mean square error
			  and prediction errors..
		
			    [mse,e] = pmodmse(pmod,y,u)
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

	uflag = len(u) > 0

	ystru, y, m = sepym(y);


	if uflag:
		yhat = pmod.predict(y, u)

	else:
		yhat = pmod.predict(y)

	e = y - yhat

	m = makerow(m)

	# Overflow is expected when LM tries unstable filter parameters; suppress it and
	# cap at a large finite value so the optimizer can still rank trial steps correctly.
	with np.errstate(over='ignore', invalid='ignore'):
		res = e * m * e

	mse = np.sum(res) / e.size
	if not np.isfinite(mse):
		mse = np.finfo(np.float64).max / 2

	return mse, e
