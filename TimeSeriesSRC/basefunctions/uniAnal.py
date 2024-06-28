import numpy as np
import matplotlib.pyplot as plt
from .sdiff import func_sdiff
from .xcorr import func_xcorr
from .plotgpac import func_plotgpac
from .parcor import func_parcor
from .gpac import func_gpac

def func_uniAnal (y,na=20,nump=10,nrg=5,ncg=0,diff=[0],per=[],perdsp=1) :
	'''
		UNIANAL Univariate analysis
			
			Syntax
		
			  [yacf,ypacf,ygpac] = uniAnal(y,na,np,nrg,ncg,diff,per,perdsp)
		
			Description
			
			  UNIANAL computes the autocorrelation and partial
			  autocorrelation functions and the generalized partial
			  autocorrelation array of the sequence y and
			  plots the results.
			
			  UNIANAL(Y,NA,NP,NRG,NCG,DIFF,PER,PERDSP) takes these inputs,
			    Y   - Vector containing the sequence.
			    NA  - The autocorrelation lags will range from -NA to +NA; default = 20.
			    NP  - Number of partial autocorrelation terms to compute; default = 10.
			    NRG - Number of rows of the GPAC to compute; default = 5.
			    NCG - Number of columns of the GPAC to compute; default = 5.
			    DIFF  = [diff1 diff2...diffNP], default = [0].
			      diffi - Order of the differencing for period i.
			    PER   = [per1 per2...perNP], default = [].
			      peri  - Period i.
			    PERDSP- Period at which to display acf and pacf data.; default = 1;
			  and returns,
			    YACF  - Autocorrelation function for Y.
			    YPACF - Partial autocorrelation function for Y.
			    YGPAC - Generalized partial autocorrelation function for Y.
				
			Examples
		
			  This code generates an autoregressive sequence.
			
			    e = randn(1,2000);
			    y = filter(1,[1 -.8],e);
		
			  The following command generates the autocorrelation and
			  partial autocorrelation functions.  The acf will be
			  computed from lag -20 to lag 20.  The pacf will be computed
			  from order 1 to order 10. 
		
			    [yacf,ypacf,ygpac] = uniAnal(y,20,10);
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''


	if ncg == 0:
		ncg = nrg

	numpts = len(y[0])

	if (len(per) + 1) != len(diff):
		error='per must have one less term than diff.'
		raise Exception(error)

	# Difference the sequence

	period = np.append([1],diff)

	for i in range(len(diff)):

		d = diff[i]
		if (d!=0):
			y = func_sdiff(y, d, period(i));

	tot = max([nump + 1, nrg + ncg])

	if na < tot:
		L = tot * perdsp
	else:
		L = na * perdsp

	if L > numpts:
		error='Not enough data to compute the acf for sufficient lags.'
		raise Exception(error)


	yacf = func_xcorr(y, y, L, 'unbiased');

	# Select data only at multiples of the display period

	xlen = len(yacf[0])
	if perdsp > 1:
		m =yacf[0,(L - perdsp):: -perdsp]
		n =yacf[0,L :xlen:perdsp]
		yacf =  np.append(m[::-1],n)
		yacf = yacf.reshape(1,-1)
		# yacf = [fliplr(yacf((L + 1 - perdsp):-perdsp: 1)) yacf((L + 1): perdsp:xlen)];


	# Calculate pacf and gpac

	ypacf, phi, sigma = func_parcor(yacf[0], nump);
	ygpac = func_gpac(yacf, nrg, ncg);


	# Convert acf to proper length for plotting

	xlen = len(yacf[0]);

	L = round((xlen - 1) / 2);


	if L != na:
		dif = L - na
		yacf = yacf[0,dif:xlen - dif]

	lag = np.arange(-na,na+1)

	confint = yacf[0,na] * 2 / np.sqrt(numpts)

	# Getfigure if it exists, or create new figure

	me = 'ACF and PACF'

	fig, (ax1, ax2) = plt.subplots(2)
	fig.suptitle(me)
	ax1.stem(lag, yacf[0])


	ax1.set_title('Autocorrelation Function')
	ax1.set_xlabel('Lag')
	xlim = ax1.get_xlim()
	ax1.plot([xlim[0], xlim[1]], [0 ,0], 'k');
	ax1.plot([xlim[0], xlim[1]], [ confint, confint], ':r');
	ax1.plot([xlim[0], xlim[1]], [-confint,-confint], ':r');

	ind = [i for i in range(1,nump+1)]
	ax2.stem(ind,ypacf[0])
	ax2.set_title('Partial Autocorrelation Function')
	ax2.set_xlabel('Lag')
	xlim = ax2.get_xlim()
	ax2.plot([xlim[0], xlim[1]], [0 ,0], 'k');

	# Get figure if it exists, or create new figure

	plt.tight_layout()
	plt.show()

	func_plotgpac(ygpac, 'GPAC Array')

	return yacf, ypacf, ygpac
