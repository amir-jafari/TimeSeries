import numpy as np
from scipy.special import gamma

def func_chisqrdf (q,n) :
	'''
		CHISQRDF Calculate Chi-square cumulative density function. 
		
			Syntax
		
			  p = chisqrdf(q,n)
		
			Description
			
			  CHISQRDF(Q,N) computes the chi square cumulative density function
			  with parameter N at the values in Q. 
		
			  CHISQRDF(Q,N) takes these inputs,
			    Q - Chi-square statistic.
			    N - Degree of freedom.
			  and returns,
			    P - Probability that the observation from the chi-square 
			        distribution will fall in the interval [0 Q].  
		
			Examples
		
			  From Chi-square distribution table, we know that if n = 10,
			  q= 18.3, then pr will be 0.95.  The following code verify 
			  this observation:
		
			    pr = chisqrdf(18.3,10)
		
			See also STATS\CHI2CDF

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

	t = np.round(np.arange(0, q, q/5000),4)
	x = gamma(n/2)

	y = (t**((n - 2) / 2)) * (np.exp((t * - 1)/2)) / ((2 ** (n / 2)) * gamma(n / 2))
	pr = np.trapz(y,t)

	#y = (t.^ ((n - 2) / 2)).* (exp(-t. / 2)) / ((2 ^ (n / 2)) * gamma(n / 2));

	return pr
