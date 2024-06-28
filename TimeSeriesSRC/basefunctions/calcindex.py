from ..Model.pmodaic import func_pmodaic as pmodaic
from ..Model.pmodbic import func_pmodbic as pmodbic
from ..Model.pmodmse import func_pmodmse as pmodmse
from ..Model.pmodsim import func_pmodsim as pmodsim

def func_calcindex (pmod,y,u):
	'''
		CALCINDEX Calculate performance index and prediction model errors
		
			Synopsis
		
			  [index,e]=calcindex(pmod,y,u)
		
			Description
		
			  This function calculates the performance index of the
			  prediction model for a given set of inputs and
			  desired outputs.
		
			  [INDEX,E]=calcindex(PMOD,Y,U) takes,
			    PMOD - Prediction model.
			    Y    - Desired outputs of the prediction model. Y may or may
			           not be a structure.  If Y is a structure, then Y.Y  
			           is the prediction model desired outputs, and Y.M is
			           the vector containing the weighting factors
			           for each error.  
			    U    - Inputs to the prediction model.
			  and returns,
			    INDEX - Prediction model performance index.
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
		
			  Here we calculate the prediction models performance index
			  and prediction errors..
		
			    [index,e] = calcindex(pmod,y,u);

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

	indexFcn = pmod.indexFcn


	if len(u) > 0:

		if indexFcn == 'pmodsim':

			index, e = pmodsim(pmod, y, u)

		elif indexFcn == 'pmodmse':

			index, e = pmodmse(pmod, y, u)

		elif indexFcn == 'pmodbic':

			index, e = pmodbic(pmod, y, u)

		elif indexFcn == 'pmodaic':

			index, e = pmodaic(pmod,y, u)

	else:

		if indexFcn == 'pmodsim':
			index, e = pmodsim(pmod, y)
		elif indexFcn == 'pmodmse':
			index, e = pmodmse(pmod, y)
		elif indexFcn == 'pmodbic':
			index, e = pmodbic(pmod, y)
		elif indexFcn == 'pmodaic':
			index, e = pmodaic(pmod, y)

	return index, e