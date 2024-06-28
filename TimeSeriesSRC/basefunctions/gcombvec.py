import numpy as np

def func_gcombvec (a1, *argv) :
	'''
		GCOMBVEC Generalized vector combinations.
		
			Syntax
		
			  gcombvec(a1,a2)
		
			Description
		
			  GCOMBVEC(A1,A2) takes two inputs,
			    A1 - Matrix of N1 (column) vectors.
			    A2 - Matrix of N2 (column) vectors.
			  and returns a matrix of N1*N2 column vectors, where the columns
			  consist of all possibilities of A2 vectors, appended to
			  A1 vectors. It can handle the case in which rows of A1 is 
			  greater than columns of A1.
		
			Example
			
			  a1 = [7; 9];
			  a2 = [1 2 3; 4 5 6];
			  a3 = gcombvec(a1,a2)

		 Revised from \toolbox\nnet\combvec.m
		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

   #=============================================================
	def nncpy(m,n):

		mr, mc = m.shape
		b = np.zeros((mr, mc*n))
		ind = np.array([i for i in range(mc)])

		for i in range(n):
			ii = i * mc
			b[:, ind + ii] = m

		return b

	#==============================================================

	def nncpyi(m, n):

		mr, mc = m.shape
		b = np.zeros((mr * n, mc))
		ind = np.array([i for i in range(mr)])

		for i in range(n):
			ii = i * mr
			b[ind + ii,:] = m

		if b.shape[1] != mc*n and b.shape[1]>1:

			c= np.array([])

			for j in range(mc):
				for i in range(mr):
					if mr == 1:
						x = b[i * mr:, j]
					else:
						x = b[i*mr:i*mr+mr,j].transpose()
					if len(c) == 0:
						c = x
					else:
						if mr != 1:
							c = np.vstack((c,x))
						else:
							c = np.hstack((c,x))
			b=c.transpose()

		elif b.shape[1] ==1:

			b=b.transpose()

		return b

	narg = len(argv)

	if narg == 0:
		y = a1

	elif narg == 1:

		a2 = argv[0]
		len1 = a1.shape[1]
		len2 = a2.shape[1]
		x1 = nncpy(a1, len2)
		x2 = nncpyi(a2, len1)
		y = np.vstack((x1 ,x2 ))

	else:
		y = func_gcombvec(func_gcombvec(a1, argv[0]), *argv[1:])

	return y
