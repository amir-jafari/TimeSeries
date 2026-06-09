import json
import numpy as np
import os

from ..basefunctions.gcombvec import func_gcombvec as gcombvec
from .model import pmodel
from .pmodbic import func_pmodbic as pmodbic
from .pmodaic import func_pmodaic as pmodaic
from .estimate import estimate as estimate
def func_selpmod (filename,y,u=[]):
	'''
		SELPMOD Select the best prediction model based on AIC and BIC criteria.
		
			Syntax
		
			  [estpmod] = selpmod(filename,y,u)
			  [estpmod] = selpmod(string,y,u)
			  [estpmod] = selpmod(cellarray,y,u)
		
			Description
		
			  SELPMOD select the best prediction model (model parameters defined
			  in 'filename') from data y, u based on aic and bic criteria.  
		
			  SELPMOD(FILENAME,Y,U) takes,
			    FILENAME - File name, cell array or string which specifies the estimation parameters.
			    Y        - Prediction model desired outputs.
			    U        - Prediction model inputs, default = [] (only for arma model).
			  and returns,
			    ESTPMOD  - Estimated prediction model, which has the following structure:
			      ESTPMOD.'TYPE'.AIC     - AIC array, along with the corresponding
			                              order parameters, such as na,nb,nc,nd.
			      ESTPMOD.'TYPE'.BIC     - BIC array, along with the corresponding
			                              order parameters, such as na,nb,nc,nd.
			      ESTPMOD.'TYPE'.AICSTAT.SIGMA - AIC model sum squared error.
			      ESTPMOD.'TYPE'.AICSTAT.STDX  - AIC model parameter standard deviation.
			      ESTPMOD.'TYPE'.BICSTAT.SIGMA - BIC model sum squared error.
			      ESTPMOD.'TYPE'.BICSTAT.STDX  - BIC model parameter standard deviation.
			      ESTPMOD.'TYPE'.AICMOD - The best prediction model selected from AIC criteria.
			      ESTPMOD.'TYPE'.BICMOD - The best prediction model selected from BIC criteria.
			      'TYPE' can be 'bjtf','arx','arma','armax','regr'.	      
		
			  FILENAME is the file that specify parameter ranges (orders of na,nb,nc,nd,nf,
			  ranges of delay,diff,per, etc.).  It should meet the following specifications:
			    (1) each line can only specify the order parameters for one model type.
			    (2) each line begins with model type, quoted with single quotation.
			    (3) each line ends with a full stop '.'.
			    (4) parameters can be specified as follows:
			          na=1-4   -> na = [1 2 3 4]
			          na=1~4   -> na = [1 2 3 4]
			          na=1,2   -> na = [1 2]
			          na=1-3,5 -> na = [1 2 3 5]
			    (4) delimiter can be '\t', ' ' , ',' , ';'.
			    (5) delimiters which are ajacent to each other are treated as one delimiter.
			    (6) lines after '%' are treated as comments.
			    (7) case is ignored.
			  
			  SELPMOD(STRING,...) takes string as the first arguement.  The string has the 
			  same specification as filename does.
		
			  SELPMOD(CELLARRAY,...) takes cell array as the first argument.  Each element of 
			  the cell is a string, which meets the same specification as filename does.
		
		
			Examples
		
			  Here is the real model parameters and y,u data generated from
			  the real model.
		
			    nb = 1; 
			    na = 2;
			    delay = 1;
			    b = {[2 3]};
			    a = {[-1 0.25]};
			    u=randn(1,2000);
			    e=randn(1,2000)*.5;
			    pmodr = newarx(na,nb,delay);
			    pmodr.a = a;
			    pmodr.b = b;
			    y = pmodsim(pmodr,e,u);
		
			  The models to be estimated can be specified as follows,

				SPEC{1} = arx ,na=1-3, nb=1~2, delay=0,1,2, diff=1-2,3
			   SPEC{2} = armax,na=1,2,nb=1,nc=0-1,nd=1,delay=0-1.

			   	Here the best prediction models are estimated based on AIC
			  	and BIC,

			    estpmod = selpmod(SPEC,y,u);
			    Mreal = getmX(pmodr)
			    Marxaic = getmX(estpmod.arx.aicmod)
			    Marxbic = getmX(estpmod.arx.bicmod)
			    Marmaxaic = getmX(estpmod.armax.aicmod)
			    Marmaxbic = getmX(estpmod.armax.bicmod)
		
			See also PREDICT, PMODSIM, ESTIMATE.

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''


	if type(filename) is dict:
		data = filename
	elif type(filename) is str:
		if filename[-4:] == 'json' and os.path.exists(filename):
			with open(filename) as json_file:
				data = json.load(json_file)
	else:
		xerror = 'First parameter should be a valid path to a JSON file or a valid path '
		raise Exception(xerror)

	result = {}

	if hasattr(u, 'ndim') and u.ndim == 1:
		u = np.reshape(u, (1, -1))

	for rec in data['models']:
		xtype =''
		na = []
		nb = []
		nc = []
		nd = []
		nf = []
		diff = []
		delay = []
		per = []

		line = 0
		for key, value in rec.items():
			if key != 'type':
				value = np.array(value)
			if key == 'type':
				xtype = value
			elif key == 'na':
				na = value
			elif key == 'nb':
				nb = value
			elif key == 'nc':
				nc = value
			elif key == 'nd':
				nd = value
			elif key == 'nf':
				nf = value
			elif key == 'diff' :
				diff = value
			elif key == 'delay':
				delay = value
			elif key == 'per':
				per = value

		line = line + 1
		if xtype not in ['bjtf','arx','arma','armax','regr']:
			xerror = 'Record No. ' + chr(line) + ' ' + xtype +' is not correct !!!'
			raise Exception(xerror)

		# set default na, nb, nc, nd, nf, diff, per, delay
		if len(na)==0:
			na =np.array([0, 1])
		if len(nb)==0:
			nb =np.array([0, 1])
		if len(nc)==0:
			nc =np.array([0, 1])
		if len(nd)==0:
			nd =np.array([0, 1])
		if len(nf)==0:
			nf =np.array([0, 1])
		if len(diff)==0:
			diff =np.array([0])
		if len(delay)==0:
			delay = np.array([0, 1])


		#na = na.reshape(1,-1)
		#nb = nb.reshape(1,-1)
		#nc = nc.reshape(1,-1)
		#nd = nd.reshape(1,-1)
		#nf = nf.reshape(1,-1)
		#diff = diff.reshape(1,-1)
		#delay = delay.reshape(1,-1)
		#per = per.reshape(1,-1)

		# automatic model selection switch type

		if xtype == 'arx':
			lna = 1
			lnb = u.shape[0]
			lDel = lnb
			A = np.array(na)
			A = A.reshape(1,-1)
			nb= nb.reshape(1,-1)
			delay = delay.reshape(1,-1)
			for i in range(lnb):
				A = gcombvec(A, nb)

			for i in range(lDel):
				A = gcombvec(A, delay)

			TIter = A.shape[1]

		elif xtype == 'arma':
			lnc = len(per) + 1
			lnd = lnc
			lDiff = lnc
			lPer = len(per)
			nc = np.array(nc).reshape(1,-1)
			nd = np.array(nd).reshape(1,-1)
			diff = np.array(diff).reshape(1,-1)
			per = np.array(per).reshape(1, -1)

			A = nc
			for i in range(lnc - 1):
				A = gcombvec(A, nc)

			for i in range(lnd):
				A = gcombvec(A, nd)


			for i in range(lDiff):
				A = gcombvec(A, diff)


			for i in range(lPer):
				A = gcombvec(A, per)


		elif xtype == 'armax':
			lna = 1;
			lnb = u.shape[0]
			lnc = 1
			lDel = lnb
			na=np.array(na).reshape(1,-1)
			nb=np.array(nb).reshape(1,-1)
			nc=np.array(nc).reshape(1,-1)
			delay=np.array(delay).reshape(1,-1)

			A = na
			for i in range(lnb):
				A = gcombvec(A, nb)

			A = gcombvec(A, nc)

			for i in range(lDel):
				A = gcombvec(A, delay)

		elif xtype == 'bjtf':
			lnb = u.shape[0]
			lnc = len(per) + 1
			lnd = lnc
			lnf = lnb
			lDel = lnb
			lDiff = lnc
			lPer = len(per)

			nb = np.array(nb).reshape(1,-1)
			nc = np.array(nc).reshape(1,-1)
			nd = np.array(nd).reshape(1,-1)
			nf = np.array(nf).reshape(1,-1)
			delay= np.array(delay).reshape(1,-1)
			diff = np.array(diff).reshape(1,-1)
			per = np.array(per).reshape(1, -1)

			A = nb
			for i in range(lnb-1):
				A = gcombvec(A, nb)

			for i in range(lnc):
				A = gcombvec(A, nc)

			for i in range(lnd):
				A = gcombvec(A, nd)

			for i in range(lnf):
				A = gcombvec(A, nf)

			for i in range(lDel):
				A = gcombvec(A, delay)

			for i in range(lDiff):
				A = gcombvec(A, diff)

			for i in range(lPer):
				A = gcombvec(A, per)

		elif xtype == 'regr':
			lnb = u.shape[0] + 1
			lDel = lnb - 1
			delay = np.array(delay).reshape(1,-1)
			A = delay
			for i in range(lDel - 1):
				A = gcombvec(A, delay)

		TIter = A.shape[1]

		# compute aic, aicmod, aicstat, bic, bicstat, bicmod
		aic = np.array([]).reshape(1,-1)
		bic = np.array([]).reshape(1,-1)
		aicstat = np.array([]).reshape(1,-1)
		bicstat = np.array([]).reshape(1,-1)
		minaic = float('Inf')
		minbic = float('Inf')
		iter = 0

		A = A.T

		for i in range(len(A)):
			temp = A[i].astype(int)

			# get model inputs switch type
			if xtype== 'arx':
				na = list(temp[0:1])
				nb = list(temp[1:lnb+1])
				#temp[: lnb] = np.zeros((lnb))
				delay = list(temp[lnb+1:])

				pmodt = pmodel(xtype, na=na, nb=nb, delay=delay);

			elif xtype == 'arma':
				nc = temp[:lnc]
				nd = temp[lnc:lnc+lnd]
				diff = temp[lnd+lnd:lnc+lnc+lDiff]
				per = temp[lnc+lnc+lDiff:]

				pmodt = pmodel(xtype, nc=nc, nd=nd, diff=diff, per=per)

			elif xtype == 'armax':
				na = temp[:lna]
				nb = temp[lna:lna+lnb]
				nc = temp[lna+lnb:lna+lnb+lnc]
				delay = temp[lna+lnb+lnc:]

				pmodt = pmodel(xtype, na=na,nb=nb, nc=nc, delay=delay)

			elif xtype == 'bjtf':
				nb = temp[:lnb]
				temp = temp[lnb:]
				nc = temp[:lnc]
				temp = temp[lnc: ]
				nd = temp[:lnd]
				temp = temp[lnd:]
				nf = temp[:lnf]
				temp = temp[lnf : ]
				delay = temp[:lnb]
				temp = temp[lnb : ]
				diff = temp[:lnc]
				temp = temp[lnc :]
				per = temp

				pmodt = pmodel(xtype, nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per = per)

			elif xtype == 'regr':
				delay = temp
				nb = [lnb - 1]
				pmodt = pmodel(xtype=xtype, nb=nb, delay=delay)

			pmodt.estimParams.epochs = 50
			pmodt.estimParams.goal = 0.01


			try:
				if len(u)==0:
					pmodt, trec, stat = estimate(pmodt, y, show_plot=False, show_output=False)
					aictmp = pmodaic(pmodt, y)
					bictmp = pmodbic(pmodt, y)
				else:
					pmodt, trec, stat = estimate(pmodt, y, u, show_plot=False, show_output=False)
					aictmp = pmodaic(pmodt, y, u)
					bictmp = pmodbic(pmodt, y, u)
			except Exception as ex:
				print(f'  [skipped — {ex}]')
				iter = iter + 1
				continue

			aic = [aic, aictmp]

			if minaic > aictmp:
				minaic = aictmp
				aicmod = pmodt
				aicstat = stat

			bic = [bic, bictmp]

			if minbic > bictmp:
				minbic = bictmp
				bicmod = pmodt
				bicstat = stat

			# display iteration
			iter = iter + 1
			if iter == 1:
				print('Selecting the best {0} prediction model'.format(xtype.upper()))

			_fmt = lambda a: int(np.asarray(a).ravel()[0]) if np.asarray(a).ravel().size == 1 else list(np.asarray(a).ravel().astype(int))
			if xtype == 'arx':
				_struct = 'na={}, nb={}, delay={}'.format(_fmt(na), _fmt(nb), _fmt(delay))
			elif xtype == 'arma':
				_struct = 'nc={}, nd={}'.format(_fmt(nc), _fmt(nd))
			elif xtype == 'armax':
				_struct = 'na={}, nb={}, nc={}, delay={}'.format(_fmt(na), _fmt(nb), _fmt(nc), _fmt(delay))
			elif xtype == 'bjtf':
				_struct = 'nb={}, nc={}, nd={}, nf={}, delay={}'.format(_fmt(nb), _fmt(nc), _fmt(nd), _fmt(nf), _fmt(delay))
			elif xtype == 'regr':
				_struct = 'nb={}, delay={}'.format(_fmt(nb), _fmt(delay))
			else:
				_struct = ''
			print('{0}: Combination {1} out of {2} total  [{3}].  aic = {4:.4f}, bic = {5:.4f}'.format(
				xtype, iter, TIter, _struct, aictmp, bictmp))


		res = {
			'model':xtype,
			'aic' : aic,
			'bic' : bic,
			'aicstat': aicstat,
			'bicstat': bicstat,
			'aicmod' : aicmod,
			'bicmod' : bicmod
		}

		result[xtype] = res

	return result

