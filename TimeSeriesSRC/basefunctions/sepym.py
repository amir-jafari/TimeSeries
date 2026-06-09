import numpy as np

def func_sepym (y):
	'''
		SEPYM Seperate y and m from y if y is a structure.
		
			Syntax
		
			  [ystru,y,m]=sepym(y);
		
			Description
		
			  This function extracts y and m from structured y if input y is a 
			  structure, and retains y as ystru.  If input y is not a structure,
			  m is returned as a row vector, which has the same order as y.
			  Constructs a structure ystru, which has field ystru.y = y and
			  ystru.m = m.
		
			  [ystru,y,m]=sepym(y) takes,
			    y     - Desired prediction model output.  If y is a structure, 
			            y contains y.y, and y.m.
			  and returns,
			    YSTRU - Structured y with field ystru.y and ystru.m.
			    y     - Desired prediction model output, not a structure.  
			    M     - Row vector containing the weighting factors for
			            each error.  If input y is not a structure, m is an 
			            ONE row vector which has the same order as y.
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $




	'''


	if type(y) is dict:
		ystru = y
		if 'm' in y:
			m = y['m']
		else:
			error='m is not a field of input y'
			raise Exception(error)

		if 'y' in y:
			y = y['y']
		else:
			error='y is not a field of input y'
			raise Exception(error)

		# Handle both 1D and 2D arrays
		y_len = len(y) if y.ndim == 1 else len(y[0])
		m_len = m.shape[1] if len(m.shape) > 1 else len(m)
		if y_len != m_len:
			error='y and m should have the same length'
			raise Exception(error)

	else:
		m = np.ones((1, y.shape[-1]))
		ystru = {
			'y':y,
			'm':m
		}

	return ystru, y, m
