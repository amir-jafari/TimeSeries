import numpy as np

def func_gpac (acf,nrows,ncols) :
	'''
		GPAC Calculate the generalized partial autocorrelation function
			
			Syntax
		
			  [gpac_array] = gpac(acf,nrows,ncols)
		
			Description
			
			  GPAC computes the generalized partial
			  autocorrelation function for the acf.
			
			  GPAC(ACF,NROWS,NCOLS) takes these inputs,
			    ACF   - Autocorrelation sequence (zero lag in the center).
			    NROWS - Number of rows of the GPAC to compute.
			    NCOLS - Number of columns of the GPAC to compute.
			  and returns,
			    GPAC_ARRAY - Generalized partial autocorrelation function.
				
			Examples
		
			  This code generates an autoregressive sequence.
			
			    e = randn(1,2000);
			    y = filter(1,[1 -.8],e);
		
			  The following commands generate the generalized partial autocorrelation 
			  function.  The gpac will be computed for 7 rows and 7 columns.
		
			    acf = xcorr(y,y,20,'unbiased');
			    [gpac_array] = gpac(acf,7,7)
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

	gpac_array = np.zeros([nrows,ncols])

	if len(acf.shape) == 1:
		acf = acf.reshape(1,-1)
	elif acf.shape[0] != 1:
		acf = np.transpose(acf)

	l = int(acf.shape[1] / 2) + 1;

	for j in range(nrows):
		for m in range(ncols):
			r1 = np.array([])
			for n in range(m+1):
				xacf = acf[0, l+j-1+n-m:  l+j+n]
				r1 = np.append(r1, xacf[::-1])
				# r1 = [r1; acf(l + j - 2 + n : -1 :l + j - 1 + n - m)];

			xlen = len(r1)
			if xlen > m+1:
				xrows = int(xlen/(m+1))
				r1 = r1.reshape(xrows,-1)
			else:
				r1 = r1.reshape(1,-1)
			rr = np.transpose(acf[0,j + l:j + l + m+1].reshape(1,-1))
			r2 = r1.copy();
			r2[:, m] = rr[:,0];
			gpac_array[j, m] = float(np.linalg.det(r2)) / float(np.linalg.det(r1));

	return gpac_array

