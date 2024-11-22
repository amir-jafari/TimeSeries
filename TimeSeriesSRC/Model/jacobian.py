import numpy as np
import time
from scipy.sparse import spdiags
import copy

from ..basefunctions.sepym import func_sepym
from ..basefunctions.makerow import func_makerow

def func_jacobian ( pmod, delta, y, u=np.array([])):
    '''
		JACOBIAN Calculates Jacobian matrix of prediction error.
		
			Syntax
		
			  [je,jj,normgX]=jacobian(pmod,delta,y,u);
		
			Description
		
			  This function calculates an approximate Hessian and
			  the gradient for the squared errors of a prediction 
			  model. Both of these terms are computed from the Jacobian
			  of the prediction errors.
		
			  The gradient of the squared error can be computed as
			  J'*E (Jacobian times errors) and the Hessian con be
			  computed as J'J (Jacobian squared).
		
			  [je,jj,normgX]=jacobian(pmod,delta,y,u) takes,
			    PMOD   - Prediction model.
			    DELTA  - Step size for derivative calculation.
			    Y    - Desired outputs of the prediction model. Y may or may
			           not be a structure.  If Y is a structure, then Y.Y  
			           is the prediction model desired outputs, and Y.M is
			           the vector containing the weighting factors
			           for each error.  
			    U      - Prediction model inputs.
			  and returns,
			    je     - Gradient.
			    jj     - Approximate Hessian.
			    normgX - Norm of gradient.
		
			Examples
		
			  Here are a few points from sample y and u sequences:
		
			    y = [-0.1867    0.5173   -3.4980    3.0093   -3.0414    1.5948];
			    u = [-0.4326   -1.6656    0.1253    0.2877   -1.1465    1.1909];
		
			  Here NEWBJTF is used to create a Box and Jenkins Transfer Function
			  model with first order polynomials.
		
			    pmod = newbjtf(1,1,1,1);
		
			  We can use JACOBIAN to calculate the gradient, Hessian,
			  and the norm of the gradient using a step size of 1e-5.
		
			    [je,jj,normgX]=jacobian(pmod,1e-5,y,u);
		
			See also ESTIMLM.

		 Yong Hu, Martin Hagan, 9-15-00
		 $Revision: 1.0 $ $Date: 21-Sep-2000 14:37:36 $

    '''

    #  def jacobian(self, delta,y,u=None):

    uflag = (len(u)>0)

    ystru, y, m = func_sepym(y)

    n_y = len(y);
    X = pmod.getmX();
    n_X = len(X);
    X1 = np.copy(X)

    if uflag:
        yhat = pmod.predict(y, u)
    else:
        yhat = pmod.predict(y)

    e = y - yhat
    j = np.array([])

    pmod2 = copy.deepcopy(pmod)

    for i in range(n_X):

        X1[i] = X[i] - delta
        pmod2.setmX(X1)

        if uflag:
            pred = pmod2.predict( y, u)
        else:
            pred = pmod2.predict( y)

        xres = (pred - yhat) / delta

        j = np.append(j, xres)
        X1[i] = X[i]

    j = j.reshape( n_X, -1)
    m = func_makerow(m)
    M = np.identity(m.shape[1])

    jj= np.dot(np.dot(j, M), j.T)
    je = np.dot(np.dot(j, M), e.T)
    normgX = np.linalg.norm(je)


    return je, jj, normgX

    ## ----------------------------------------------------------------------
