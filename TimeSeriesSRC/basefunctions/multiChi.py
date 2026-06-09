import numpy as np
from scipy.signal import lfilter
from .xcorr import func_xcorr as xcorr
from .chisqrdf import func_chisqrdf as chisqrdf

def func_multiChi(pmod, y, u, k1=20, k2=20, alpha1=0.05, alpha2=0.05):
	'''
		MULTICHI Multivariate Chi-Square test.

			Syntax

			  [pass,q,s,nq,ns] = multiChi(pmod,y,u,k1,k2,alpha1,alpha2)

			Description

			  MULTICHI performs multivariate chi-square tests. It calculates
			  two chi-square statistics. The first statistic tests the
			  whiteness of the residuals. The second tests cross-correlations
			  between residuals and model inputs.

			  MULTICHI(PMOD,Y,U,K1,K2,ALPHA1,ALPHA2) takes these inputs,
			    PMOD   - Prediction model.
			    Y      - Prediction model desired outputs (1-D array).
			    U      - Prediction model inputs (1-D or 2-D array, shape (1,N)).
			    K1     - Number of lags of the ACF to use, default = 20.
			    K2     - Number of lags of the cross correlation to use, default = 20.
			    ALPHA1 - Probability of type I error for the residuals, default = 0.05.
			    ALPHA2 - Probability of type I error for the crosscorrelation, default = 0.05.
			  and returns,
			    PASS - List of two values.
			           PASS[0] = 1 if test for residual correlation is passed, 0 otherwise.
			           PASS[1] = 1 if test for cross correlation is passed, 0 otherwise.
			    Q    - The chi-square statistic for the residual autocorrelation.
			    S    - The chi-square statistic for the crosscorrelation.
			    NQ   - Degrees of freedom for Q.
			    NS   - Degrees of freedom for S.

			Examples

			  This code estimates a Box-Jenkins Transfer Function model from the
			  gas furnace data.

			    import scipy.io
			    data = scipy.io.loadmat('furnace.mat')
			    y = data['y'][0] - data['y'].mean()
			    u = data['u'] - data['u'].mean()
			    pmod = pmodel('bjtf', nb=[2], nc=[0], nd=[2], nf=[2], delay=[3], diff=[0], per=[])
			    pmod, trec, stat = estimate(pmod, y, u)

			  The estimated BJTF model then performs chi-square test.

			    pass_arr, q, s, nq, ns = multiChi(pmod, y, u)

			See also UNICHI.

		Yong Hu, Martin Hagan, 9-15-00
		Python port: 2026

	'''
	from ..Model.selpmod import func_selpmod as selpmod

	y_1d = np.asarray(y).ravel()
	u_1d = np.asarray(u).ravel()
	u_2d = u_1d.reshape(1, -1)

	# Prewhiten the input with an ARMA model
	# nd starts at 1 to avoid the degenerate ARMA(0,0) case (no free parameters)
	print('Prewhitening input, please wait...')
	arma_spec = {
		'models': [{
			'type': 'arma',
			'nc': [0, 1, 2, 3],
			'nd': [1, 2, 3],
			'diff': [0]
		}]
	}
	estpmodu = selpmod(arma_spec, u_1d)
	bicmod = estpmodu['arma']['bicmod']

	# Build MA (C) and AR (D) polynomials with leading 1
	c_coef = np.asarray(bicmod.c[0]).ravel() if bicmod.c and len(bicmod.c) > 0 else np.array([])
	d_coef = np.asarray(bicmod.d[0]).ravel() if bicmod.d and len(bicmod.d) > 0 else np.array([])
	Rq = np.concatenate([[1.0], c_coef])   # C polynomial (MA numerator)
	Sq = np.concatenate([[1.0], d_coef])   # D polynomial (AR denominator)

	# Prewhitened input: al = D(z)/C(z) * u
	al = lfilter(Sq, Rq, u_1d)

	# AUTO CORRELATION TEST — residuals from the fitted model
	e = y_1d - pmod.predict(y_1d, u_2d)

	e_2d = e.reshape(1, -1)
	acf_full = xcorr(e_2d, e_2d, k1, 'unbiased')

	# Keep positive lags (0..k1), normalize by lag-0, drop lag-0
	acf = acf_full[0, k1:]      # lags 0..k1
	acf = acf / acf[0]          # normalize → ρ_0 = 1
	acf = acf[1:]               # drop lag-0, now lags 1..k1

	q = len(y_1d) * np.sum(acf ** 2)

	# Degrees of freedom for Q: lags minus noise model free parameters
	nc = len(pmod.c[0]) if (hasattr(pmod, 'c') and len(pmod.c) > 0 and len(pmod.c[0]) > 0) else 0
	nd = len(pmod.d[0]) if (hasattr(pmod, 'd') and len(pmod.d) > 0 and len(pmod.d[0]) > 0) else 0
	nq = k1 - (nc + nd)

	prq = chisqrdf(q, nq)
	pvalq = 1.0 - prq
	pass_arr = [0, 0]
	pass_arr[0] = 1 if pvalq > alpha1 else 0

	# CROSS CORRELATION TEST — xcorr between prewhitened input and residuals
	al_2d = al.reshape(1, -1)
	ccf_full = xcorr(al_2d, e_2d, k2, 'unbiased')

	# Keep positive lags (0..k2)
	ccf = ccf_full[0, k2:]      # lags 0..k2 (k2+1 elements)

	# Normalize by sqrt(var(al) * var(e))
	alvar = np.var(al, ddof=1)
	epsvar = np.var(e, ddof=1)
	ccf = ccf / np.sqrt(alvar * epsvar)

	s = len(y_1d) * np.sum(ccf ** 2)

	# Degrees of freedom for S: transfer function free parameters
	nb = (len(pmod.b[0]) - 1) if (hasattr(pmod, 'b') and len(pmod.b) > 0 and len(pmod.b[0]) > 0) else 0
	nf = len(pmod.f[0]) if (hasattr(pmod, 'f') and len(pmod.f) > 0 and len(pmod.f[0]) > 0) else 0
	ns = k2 + 1 - (nb + nf + 1)

	prs = chisqrdf(s, ns)
	pvals = 1.0 - prs
	pass_arr[1] = 1 if pvals > alpha2 else 0

	return pass_arr, q, pvalq, s, pvals, nq, ns
