import numpy as np
from .xcorr import func_xcorr as xcorr
from .chisqrdf import func_chisqrdf as chisqrdf

def func_uniChi(pmod, y, k=20, alpha=0.05):
	'''
		UNICHI Univariate Portmanteau lack-of-fit (Chi-square) test.

			Syntax

			  [pass,q,n] = uniChi(pmod,y,k,alpha)

			Description

			  UNICHI performs a univariate portmanteau lack-of-fit test. It calculates
			  the chi-square statistics on the residuals of a fitted model.
			  Large values of the statistic indicate that the residuals are
			  not white, and therefore the model does not provide a good fit.

			  UNICHI(PMOD,Y,K,ALPHA) takes these inputs,
			    PMOD  - Prediction model.
			    Y     - Prediction model desired outputs (1-D array).
			    K     - Number of lags of the ACF to use in the statistic, default = 20.
			    ALPHA - Probability of type I error, default = 0.05.
			  and returns,
			    PASS - 1 if the chi-square test is passed, 0 otherwise.
			    Q    - The chi-square statistic.
			    N    - Degrees of freedom.
			    PVAL - p-value of the chi-square statistic.

			Examples

			  This code generates an ARMA sequence and an estimated
			  ARMA prediction model.

			    from scipy.signal import lfilter
			    import numpy as np
			    e = np.random.randn(2000)
			    y = lfilter([1, 0.5], [1, -0.8], e)
			    pmod = pmodel('arma', nc=[1], nd=[1], diff=[0], per=[])
			    pmod, trec, stat = estimate(pmod, y)

			  Then a portmanteau lack-of-fit test is performed
			  on the estimated ARMA model.

			    passed, q, n = uniChi(pmod, y)

			See also MULTICHI.

		Yong Hu, Martin Hagan, 9-15-00
		Python port: 2026

	'''

	y_1d = np.asarray(y).ravel()

	# Compute residuals
	e = y_1d - pmod.predict(y_1d)

	# ACF of residuals at lags 0..k
	e_2d = e.reshape(1, -1)
	acf_full = xcorr(e_2d, e_2d, k, 'unbiased')

	# Keep positive lags (0..k), normalize by lag-0 value, drop lag-0
	acf = acf_full[0, k:]       # lags 0..k
	acf = acf / acf[0]          # normalize → ρ_0 = 1
	acf = acf[1:]               # drop lag-0, now lags 1..k

	# Box-Pierce Q statistic
	q = len(y_1d) * np.sum(acf ** 2)

	# Degrees of freedom: lags minus number of free parameters
	X = pmod.getmX()
	num_para = len(X)
	n = k - num_para

	# Chi-square cumulative probability
	pr = chisqrdf(q, n)
	pval = 1.0 - pr
	passed = 1 if pval > alpha else 0
	print('pval:', pval)
	print('alpha:', alpha)
	print('pr:',pr)
	print('q:',q)

	return passed, q, n, pval
