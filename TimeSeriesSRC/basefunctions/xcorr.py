import numpy as np
from ..basefunctions import makerow

def func_xcorr (a,b,maxlag=20,flag='none') :
	'''
		XCORR Calculate autocorrelation function
		
			Syntax
		
			  C = XCORR(A,B,MAXLAG,'flag')
		
			Description
			
		    XCORR(A,B,MAXLAG,'flag') 
		    normalizes the correlation according to 'flag':
		        biased   - scales the raw cross-correlation by 1/N.
		        unbiased - scales the raw correlation by 1/(N-abs(k)), where k 
		                   is the index into the result.
		        coeff    - normalizes the sequence so that the correlations at 
		                   zero lag are identically 1.0.
		        none     - no scaling (this is the default).
				
			Examples
		
			  This code generates an autoregressive sequence.
			
			    e = randn(1,2000);
			    y = filter(1,[1 -.8],e);
		
			  The following command generates the autocorrelation functions.
			  The acf will be computed from lag -20 to lag 20.   
		
			    yacf = xcorr(y,y,20,'unbiased');
		

		 Yong Hu, Martin Hagan, 10-05-00
		 $Revision: 1.0 $ $Date: 05-Oct-2000 00:50:00 $

	'''


	error = ''

	if type(flag) != str:
		error='FLAG should be a string'
		raise Exception(error)

	a = makerow.func_makerow(a)
	b = makerow.func_makerow(b)
	n = len(a[0])
	n1 = len(b[0])

	if maxlag >= n:
		error= 'MAXLAG should be less than the lengths of A and B.'
		raise Exception(error)

	if n != n1:
		error = 'The lengths of A and B should be equal.'
		raise Exception(error)

	c = np.zeros([1, (2 * maxlag + 1)])

	# for i=0:maxlag

	for i in range(maxlag+1):

		ac = np.dot(a[0,0:(n-i)], np.transpose(b[0,i:n]))

		#ac = a(1:(n - i))*b(1 + i: n)';

		if i == 0:
			ac0 = ac

		# if i == 0,
		# ac0 = ac;

		if flag == 'unbiased':
			ac = ac / (n-i)
		elif flag == 'biased':
			ac = ac / n
		elif flag == 'coeff':
			ac = ac / ac0

		c[0, maxlag + i] = ac
		c[0, maxlag - i] = ac

	return c
