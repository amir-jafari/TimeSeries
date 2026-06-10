import numpy as np
import time
from scipy.sparse import spdiags
import copy

from ..basefunctions.sepym import func_sepym
from ..basefunctions.makerow import func_makerow

def func_jacobian ( pmod, delta, y, u=np.array([])):
    """Compute the numerical Jacobian, approximate Hessian, and gradient.

    Uses a forward-difference scheme to approximate the Jacobian
    :math:`J` of the one-step-ahead prediction errors with respect to the
    model parameters :math:`X`:

    .. math::

        J_{ij} \\approx \\frac{\\hat{y}(t_i;\\, X + \\delta e_j) - \\hat{y}(t_i;\\, X)}{\\delta}

    The approximate Gauss-Newton Hessian and gradient are then:

    .. math::

        JJ = J M J^T, \\qquad je = J M E

    where :math:`M` is a diagonal weight matrix and :math:`E` is the error
    vector.

    Parameters
    ----------
    pmod : pmodel
        Prediction model with current parameter values.
    delta : float
        Finite-difference step size (e.g. ``1e-7``).
    y : array-like or dict
        Desired output sequence.  If a dict, must have keys ``'y'`` and
        ``'m'`` (sample weights).
    u : array-like, optional
        Input sequence.  Default ``np.array([])``.

    Returns
    -------
    je : ndarray, shape (n_params, 1)
        Gradient :math:`J M E`.
    jj : ndarray, shape (n_params, n_params)
        Approximate Hessian :math:`J M J^T`.
    normgX : float
        Euclidean norm of the gradient vector.

    See Also
    --------
    func_estimlm : Levenberg-Marquardt optimiser that calls this function.
    """

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
