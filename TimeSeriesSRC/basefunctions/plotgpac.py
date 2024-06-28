import matplotlib.pyplot as plt
# importing the style package
from matplotlib import style
import numpy as np
import math

def func_plotgpac (gpac,gtitle="Gpac Array",  ax=None) :
	'''
		PLOTGPAC Plot the gpac array.
		
			Syntax
		
			  PLOTGPAC(GPAC,GTITLE)
		
			Description
			
			  PLOTGPAC(GPAC,GTITLE) takes these inputs,
			    GPAC   - GPAC array.
			    GTITLE - Title for the plot.
			  and displays the GPAC array represented as a grid of squares.
			
			  Each square's AREA represents the magnitude of an element.
			  Each square's COLOR represents the element's sign.
			  RED for negative values, GREEN for positive.
		
			Examples
		
			  This code generates an autoregressive sequence.
			
			    e = randn(1,2000);
			    y = filter(1,[1 -.8],e);
		
			  The following command generates the autocorrelation and
			  partial autocorrelation functions.  The acf will be
			  computed from lag -20 to lag 20.  The pacf will be computed
			  from order 1 to order 10. 
		
			    acf = xcorr(y,y,20,'unbiased');
		 	    [gpac_array] = gpac(acf,7,7);
			    plotgpac(gpac_array);
			    
		

		 Mark Beale, 1-31-92
		 Revised 12-15-93, MB
		 Revised 11-31-97, MB
		 Revised from network weights to GPAC display 7-1-00
		 Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $
		 $Revision: 1.1 $ $Date: 12-Dec-2006 16:44:39 $ use ind instead of [i,j]
		 to find min and max

	'''

	limit = 5
	ind = np.where(gpac > limit)
	if (len(ind)>0):
		gpac[ind] = limit

	ind = np.where(gpac < (limit * -1))
	if (len(ind)>0):
		gpac[ind] = limit * -1

	# max_m = max(max(abs(gpac)));
	max_m = limit
	min_m = max_m / 300

	# DEFINE BOX EDGES
	xn1 = np.array([-1.,-1.,1.]) * 0.5;
	xn2 = np.array([ 1., 1., -1.]) * 0.5;

	yn1 = np.array([1.,-1.,-1.]) * 0.5;
	yn2 = np.array([-1., 1. ,1.]) * 0.5;

	# DEFINE POSITIVE BOX
	xn = np.array([-1., -1., 1., 1., -1.]) * 0.5;
	yn = np.array([-1., 1., 1., -1., -1.]) * 0.5;

	# DEFINE POSITIVE BOX
	xp = np.append(xn,np.array([-1.,1.,1.,0., 0.]) * 0.5)
	yp = np.append(yn,np.array([ 0.,0.,1.,1.,-1.]) * 0.5)

	S, R = gpac.shape

	display=False
	with plt.style.context('seaborn'):
		if ax == None:
			fig, ax = plt.subplots()
			display = True
		xxlim = np.array([0,R]) + 0.5
		xylim = np.array([0,S]) - 0.5
		ax.xlim = xxlim
		ax.ylim = xylim

		new_x = range(math.floor(min(xxlim)), math.ceil(max(xxlim)) + 1)
		new_y = range(math.floor(min(xylim)), math.ceil(max(xylim)) + 1)

		ax.set_xticks(new_x)
		ax.set_yticks(new_y)
		ax.invert_yaxis()

		for i in range(S):
			i1 = i;
			for j in range(R):
				xval = (np.abs(gpac[i, j]) - min_m) / max_m
				if xval >= 0:
					m = round(np.sqrt(xval),2)
					m = min(m, max_m) * 0.95

					if gpac[i, j] >= 0:
						ax.fill(xn * m + j+1, yn * m + i1, color='green')
						ax.plot(xn1 * m + j+1, yn1 * m + i1, 'w' ,xn2 * m + j+1, yn2 * m + i1, 'k', linewidth=3)
					elif gpac[i, j] < 0:
						ax.fill(xn * m + j+1, yn * m + i1,color= 'red');
						ax.plot(xn1 * m + j+1, yn1 * m + i1, 'k', xn2 * m + j+1, yn2 * m + i1, 'w',linewidth=3);


		ax.plot(np.array([0, R, R, 0, 0]) + 0.5, np.array([0, 0, S, S, 0]) - 0.5, 'w');
		ax.set_xlabel('Denominator Order')
		ax.set_ylabel('Numerator Order')
		ax.set_title(gtitle)


		plt.tight_layout()

		if display:
			plt.tight_layout()
			plt.show()

		return ax