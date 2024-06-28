def func_makerow (y):
	'''
		MAKEROW Convert vector into row vector

		   Parameters:
		   	 y = numpy array
		
		   Syntax
		
		     yr = makerow(y)
		
		   Description
		
		     MAKEROW converts any vector into a row vector.
		
		     MAKEROW(Y) takes this input
		       Y - Row or column vector
		     and returns,
		       YR - Row vector.
		
		   Examples
		
		     This code creates a column vector y and converts it to a row.
		
		       y = [1;2;3;4];
		       yr = makerow(y);
		

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

		------------------------------------------------
		Lilian SDR, 06,06.2020

	'''

	## what happens if the array size == 0 or grearter than 2

	xshape = y.shape
	if len(xshape) == 1:
		yr = y.reshape(1,-1)
	elif xshape[0] > xshape[1]:
		yr = y.transopose()
	else:
		yr = y

	return yr
