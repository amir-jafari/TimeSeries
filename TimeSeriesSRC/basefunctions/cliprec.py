def func_cliprec (tr,epochs) :
	'''
		CLIPTR Clip training record to the final number of epochs.
		
			Syntax
		
			  tr = cliptr(tr,epochs)
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

	'''

	indexes = list(range(epochs + 1))
	for name in tr:
		tr[name]  = tr[name][indexes[0]:indexes[-1]]

	return tr
