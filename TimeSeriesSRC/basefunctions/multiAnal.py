import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from .gpac import func_gpac
from .impest import func_impest
from .plotgpac import func_plotgpac
from .xcorr import func_xcorr
from .makerow import func_makerow

def func_multiAnal (u,y,nng=5,ndg=5,nnh=5,ndh=5,lg=12,lh=12) :
	'''
		MULTIANAL Multi-variable analysis.
			
			Syntax
		
			  [g,rv,g_gpac,h_gpac] = multiAnal(u,y,nng,ndg,nnh,ndh,lg,lh)
		
			Description
			
			  MULTIANAL provides an analysis between two sequences: u and y.
			  It estimates the impulse response beteen the two variables, the
			  residual autocorrelation function, the GPAC for the G transfer
			  function between u and y, and the GPAC for the H transfer function
			  between e and y (ARMA model). 
			
			  MULTIANAL(U,Y,NNG,NDG,NNH,NDH,LG,LH) takes these inputs,
			    U   - Input sequence.
			    Y   - Output sequence.
			    NNG - Maximum order for G transfer function numerator; default = 5.
			    NDG - Maximum order for G transfer function denominator; default = 5.
			    NNH - Maximum order for H transfer function numerator; default = 5.
			    NDH - Maximum order for H transfer function denominator; default = 5.
			    LG  - Number of lags of the impulse response to compute; default = 10.
			    LH  - Number of lags of the residual acf to compute; default = 10.
			  and returns,
			    G      - Estimated impulse response between u and y.
			    RV     - Residual autocorrelation function.
			    G_GPAC - GPAC for the G transfer function.
			    H_GPAC - GPAC for the H transfer function.
				
			Examples
		
			  This code generates a first order sequence y from an
			  input sequence u and a noise sequence e.
			
			    e = randn(1,2000)*0.2;
			    u = randn(1,2000);
			    y = filter(1,[1 .5],u) + filter(1,[1 -.8],e);
		
			  The following command generates the multivariate analysis 
			  for the u and y sequences.  It uses the default orders of 5.
		      
			    [g,rv,g_gpac,h_gpac] = multiAnal(u,y);
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $


	'''

	u = func_makerow(u)
	y = func_makerow(y)

	u_1d = u[0]
	y_1d = y[0]

	# Prewhiten u with a BIC-selected ARMA model; apply same filter to y.
	# This replaces the ill-conditioned direct Wiener-Hopf for correlated inputs.
	from ..Model.selpmod import func_selpmod as selpmod
	arma_spec = {
		'models': [{
			'type': 'arma',
			'nc': [0, 1, 2, 3],
			'nd': [1, 2, 3],
			'diff': [0]
		}]
	}
	print('Prewhitening input for impulse response estimation, please wait...')
	estpmodu = selpmod(arma_spec, u_1d)
	bicmod = estpmodu['arma']['bicmod']

	c_coef = np.asarray(bicmod.c[0]).ravel() if bicmod.c and len(bicmod.c) > 0 else np.array([])
	d_coef = np.asarray(bicmod.d[0]).ravel() if bicmod.d and len(bicmod.d) > 0 else np.array([])
	Rq = np.concatenate([[1.0], c_coef])   # C polynomial (MA)
	Sq = np.concatenate([[1.0], d_coef])   # D polynomial (AR)

	al = lfilter(Sq, Rq, u_1d)   # prewhitened input α
	be = lfilter(Sq, Rq, y_1d)   # filtered output β

	# K = nng + ndg + 1
	K = lg
	g = func_impest(al.reshape(1, -1), be.reshape(1, -1), K)
	index = range(0, K+1)

	# Get figure if it	exists, or create	new	figure
	me = 'Impulse Response and Residual ACF'

	fig, ax= plt.subplots(2)
	fig.suptitle(me)

	ax[0].stem(index,g)
	ax[0].set_title('Impulse Response')
	ax[0].set_xlabel('Lag')
	xlim = ax[0].get_xlim()
	ax[0].plot([xlim[0], xlim[1]], [0, 0], 'k')

	# Get figure if it exists, or create new figure

	me = 'G & H GPAC Arrays'

	# g_aug must have half-length >= nng+ndg-1 for gpac to index safely.
	# If lg < nng+ndg-1 we zero-pad g (equivalent to assuming g[k]=0 for k>K).
	K_gpac = max(K, nng + ndg - 1)
	g_padded = np.concatenate([g, np.zeros(K_gpac - K)]) if K_gpac > K else g
	g_aug = np.hstack((np.zeros(K_gpac), g_padded))
	g_gpac = func_gpac(g_aug, nng, ndg)


	y1 = np.convolve(g, u[0])

	#yy1 = filter(g, 1, u)  # is not used in the original code

	y1 = y1[0:len(y[0])];
	v = y[0] - y1

	L = nnh + ndh + 1
	rv = func_xcorr(v, v, L, 'biased')
	lag = range(-L,L+1)
	confint = rv[0,L] * 2 / np.sqrt(len(v))

	ax[1].stem(lag, rv[0])
	ax[1].set_title('Autocorrelation Function for V')
	ax[1].set_xlabel('Lag')

	xlim = ax[1].get_xlim()

	ax[1].plot([xlim[0],xlim[1]], [0, 0], 'k')
	ax[1].plot([xlim[0],xlim[1]], [confint, confint], ':r')
	ax[1].plot([xlim[0],xlim[1]], [-confint, -confint], ':r')

	plt.tight_layout()
	plt.show()
	plt.close()

	h_gpac = func_gpac(rv, nnh, ndh)





	fig, ax= plt.subplots(2)

	ax[0] = func_plotgpac(g_gpac, 'GPAC for G Transfer Function', ax[0])
	ax[1] = func_plotgpac(h_gpac, 'GPAC for H Transfer Function', ax[1])

	plt.tight_layout()
	plt.show()
	plt.close()

	return g,rv,g_gpac,h_gpac

