import numpy as np

def func_newrec (epochs, *argv) :
	'''
		NEWTR New training record with any number of optional fields.
		
			Syntax
		
			  tr = newtr(epochs,'fieldname1','fieldname2',...)
			  tr = newtr([firstEpoch epochs],'fieldname1','fieldname2',...)
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

	names = argv

	epoch = [i for i in range(epochs)]
	blank = [0 for i in range(epochs)]
	tr = {}
	for i in range(len(names)):
		tr[names[i]] = blank

	#tr['epoch'] = epoch

	return tr