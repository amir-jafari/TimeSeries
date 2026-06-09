from . import makerow
from . import xcorr
import numpy as np

def func_impest (u,y,k) :
	'''
		IMPEST Estimate the impulse response.
			
			Syntax
		
			  [g]=impest(u,y,k)
		
			Description
			
			  IMPEST estimates the impulse response between the
			  sequence u and the sequence y.
			
			  IMPEST(U,Y,K) takes these inputs,
			    U - Input sequence.
			    Y - Output sequence.
			    K - Number of lags of the impulse response to compute.
			  and returns,
			    G - Estimate of the impulse response function.
				
			Examples
		
			  This code generates a random sequence.
			
			    e = randn(1,2000)*0.2;
			    u = randn(1,2000);
			    y = filter(1,[1 .5],u) + filter(1,[1 -.8],e);
		
			  The following commands estimate the impulse response between 
			  u and y.  The impulse response will be computed for 10 lags.
		
			    g = impest(u,y,10)
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

		 Make sure that u and y are rows
	'''

	u = makerow.func_makerow(u);
	y = makerow.func_makerow(y);

	ru = xcorr.func_xcorr(u, u, k, 'biased');
	ruy = xcorr.func_xcorr(u, y, k, 'biased');

	# This was for the original version of xcorr
	# ruy = xcorr(y, u, k, 'unbiased');
	# This is for the updated xcorr

	l = k + 1
	r1 = np.array([])
	for n in range(l):
		xru = ru[0,l - 1 + n - k: l + n]
		if len(r1) ==0:
			r1= xru[::-1]
			r1= r1.transpose()
		else:
			r1 = np.vstack((r1, xru[::-1].transpose()))

		#r1 = [r1;ru(: -1:l - 1 + n - k)];

	rr = ruy[0,l-1:l + k]


	g= np.linalg.solve(r1,rr)


	return g
