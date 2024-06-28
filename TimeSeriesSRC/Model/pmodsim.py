import numpy as np

from ..basefunctions.makerow import func_makerow as makerow


def func_pmodsim (pmod,e,u=[]):
	'''
		PMODSIM Simulate a prediction model.
		
			Syntax
		
			  [Y] = PMODSIM(PMOD,E,U)
		
			Description
		
			  PMODSIM simulates the prediction model PMOD.
		
			  PMODSIM(PMOD,E,U) takes,
			    PMOD - Prediction model.
			    E    - White noise.
			    U    - Prediction model inputs.
			  and returns,
			    Y    - Prediction model output.
		
		
			Examples
		
			  Here are a few points from sample e and u sequences:
		
			    e = [ 0.9501    0.2311    0.6068    0.4860    0.8913    0.7621];
			    u = [-0.4326   -1.6656    0.1253    0.2877   -1.1465    1.1909];
		
			  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
			  model with first order polynomials.
		
			    pmod = newbjtf(1,1,1,1);
		
			  The parameters of the model are all set to random values by default. 
			  Now we can simulate this random prediction model: 
		
			    y = pmodsim(pmod,e,u);
			    ind = 1:length(y);
			    plot(ind,u,'o',ind,y,'x')
			    
			Algorithm
		
			  PMODSIM simulates the following system model.
		
		     y(t) = sumi{Gi ui(t)} + H e(t)
		
			See also PREDICT, ESTIMATE

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $


		preprocessing
	'''

	# preprocessing
	uflag = len(u)>0
	if uflag:
		u = makerow(u)

	upreproc = pmod.upreproc
	pr = upreproc.shape[0]
	if (uflag & pr!=0):
		if (pr!=1 & pr!= u.shape[0] ):
			xerror='rows of upreproc should either equals 1 or the number of inputs. '
			raise Exception (xerror)

		if pr == 1:
			pc = upreproc[pr,:]
			for i in range(pc):
				u = eval(upreproc[pr, i], u)

		else:
			for i in range(pr):
				pc = len(upreproc[i,:])
				for j in range(pc):
					if len(upreproc[i, j])>0:
						u[i,:] = eval(upreproc[i, j], u[i,:])

	if pmod.type== 'regr':
		# regression model

		ru, cu = u.shape
		udelay=np.zeroes(ru,cu)
		idel = 0
		for i in range(ru):
			if (pmod.delay)==0:
				idel = 0
			else:
				idel = pmod.delay[i]

			udelay[i,:] = np.add( np.zeros(1, idel) , u[i, : (cu - idel)])

			u1 = np.add(np.ones(1, len(e)), udelay)
			y = pmod.b[0] * u1 + e

	else: # bjtf, arma, armax, arx model

		# Expand the parameter vectors into g and h form
		ng, dg, nh, dh = pmod.getGH

		num_inputs = len(ng)
		diff = pmod.diff;
		period = pmod.period;

		# Add the differencing terms into H
		for i in range(diff[0]):
			dh = np.conv(dh, [1, -1])

		lp = len(period)

		for i in range(lp):
			per = period(i);
			ddh = np.add([1], np.zeros(1, per - 1) - 1)
			for j in range(diff[i + 1]):
				dh = np.conv(dh, ddh)

		# Simulate the prediction model
		y = filter(nh, dh, e)
		for i in range(num_inputs):
			y = y + filter(ng[i], dg[i], u[i,:])

	# post processing
	ypostproc = pmod.ypostproc;
	pc = ypostproc.shape[1] # only one output is possible
	if (pc & ypostproc.shape[0] !=1):
		xerror='ypostproc should have only one row. '
		raise Exception (xerror)

	if pc!=0:
		for i in range(pc):
			y = eval(ypostproc[i], y)

	return y
