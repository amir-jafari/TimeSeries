import numpy as np

def func_partoacf (phi,theta,lagmax,var_a) :
	'''
		partoacf
		
		  This routine takes the parameters of an
		  ARMA model and computes the autocorrelation
		  function of the process.  The ARMA parameters
		  should be given in the vectors phi (AR) and
		  theta (MA).  The routine assumes that the
		  variance of the white noise is var_a.
		  The number of acf elements to be computed will
		  be equal to lagmax.  The first elements of 
		  phi and theta are typically equal to 1, which
		  corresponds to the zeroth order term.
		  The form of phi is [1 phi1 phi2], and the
		  form of theta is [1 theta1 theta2], where
		   y(t) + phi1*y(t-1) + phi2*y(t-2) = a(t) + theta1*a(t-1) + theta2*a(t-2)
		

	'''

	p = len(phi);
	n, m = phi.shape;
	if m == p:
		phi = phi.transpose()

	q = len(theta);
	n, m = theta.shape;
	if m == q:
		theta = theta.transpose()

	if p > q:
		theta = np.array([theta,np.zeros((p - q), 1)])

	if p < q:
		phi = [phi,np.zeros((q - p), 1)]
		p = q

	a1 = phi
	for i in range(1, len(p)):
		a1 = np.append( a1, [phi[i:p], np.zeros((i-1),0)])
		#a1 = [a1[phi(i:p);zeros((i - 1), 1)]];

	a1[:, 1]=a1[:, 1] / 2;

	a2 = phi
	for i in range(2,len(p)):
		a2=np.append(a2, [np.zeros((i-1, 0)), phi[0: (p+ 1 - i)]])

		#a2 = [a2[zeros((i - 1), 1);
		#phi(1: (p + 1 - i))]];

	imp = np.linalg.solve(a2,theta)
	b1 = theta;

	for i in range(1, len(p)):
		b1 = np.append(b1, [theta[i:p], np.zeros((i-1),1) ])
		#b1 = [b1[theta(i:p);zeros((i - 1), 1)]];


	a2[:, 0]=a2[:, 0] / 2
	acf = np.dot(np.linalg.solve((a1 + a2),b1) , imp)
	mtot = len(acf)

	for i in range(mtot, lagmax - 1):

		tacf = np.dot((np.transpose(phi[1:p]) * -1), acf[i:-1:i-p+2])

		#tacf = -phi(2:p)'*acf(i:-1:i-p+2);
		#acf = [acf;tacf];

	acf = np.dot(var_a , acf)

	return acf,imp
