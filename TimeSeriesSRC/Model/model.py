import numpy as np
from dataclasses import dataclass
from scipy.signal import lfilter

from TimeSeriesSRC.Model.estimlm import func_estimlm
from ..basefunctions.makerow import  func_makerow as makerow
from ..basefunctions.sepym import func_sepym as sepym
from ..basefunctions.sdiff import func_sdiff as sdiff

@dataclass
class Parameters:
    epochs: int
    goal :float
    min_grad : float
    mu : float
    mu_dec : float
    mu_inc : float
    mu_max : float
    show : int
    max_time : float
    delta : float

class pmodel:
    def __init__(self, xtype, na=[0,1], nb=[], nc=[], nd=[], nf=[], delay=[], diff=[0], per=[], upre=[], ypre=[], ypost=[], eFcn='estimlm', indexFcn='pmodmse', initFcn='initrand'):
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nd = nd
        self.nf =nf
        self.delay = delay
        self.diff = diff
        self.per = per
        self.period = per
        self.upreproc = np.array(upre)
        self.ypreproc = np.array(ypre)
        self.ypostproc = np.array(ypost)
        self.initFcn = initFcn
        self.estimFcn = eFcn
        self.indexFcn = indexFcn
        self.type = xtype
        self.estimParams = Parameters(epochs = 100,
                                      goal = 0,
                                      min_grad=1e-4,
                                      mu=0.001,
                                      mu_dec=0.1,
                                      mu_inc=10,
                                      mu_max=1e10,
                                      show=10,
                                      max_time=float('inf'),
                                      delta=1.0000e-07
                                      )

        self.b = []
        self.c = []
        self.d = []
        self.f = []
        self.a = []


        self.new_model()
        # equivalent to predict model
        self.init()
    def __str__(self):
        s=  '   Prediction Model object:\n\n'
        s=s+'              type: {0}\n\n'.format(self.type)
        s=s+'  model parameters:\n\n'
        s=s+'                       a: {0}\n'.format(self.a)
        s=s+'                       b: {0}\n'.format(self.b)
        s=s+'                       c: {0}\n'.format(self.c)
        s=s+'                       d: {0}\n'.format(self.d)
        s=s+'                       f: {0}\n'.format(self.f)
        s=s+'                    diff: {0}\n'.format(self.diff)
        s=s+'                  period: {0}\n'.format(self.per)
        s=s+'                   delay: {0}\n'.format(self.delay)
        s=s+'\n'
        s=s+'         functions:\n'
        s=s+'\n'
        s=s+'                estimFcn: {0}\n'.format(self.estimFcn)
        s=s+'                indexFcn: {0}\n'.format(self.indexFcn)
        s=s+'                 initFcn: {0}\n'.format(self.initFcn)
        s=s+'\n'
        s=s+'pre-post processor:\n'
        s=s+'\n'
        s=s+'                upreproc: {0}\n'.format(self.upreproc)
        s=s+'                ypreproc: {0}\n'.format(self.ypostproc)
        s=s+'               ypostproc: {0}\n'.format(self.ypostproc)
        return s

    def set_data(self, y, u=np.array([[]])):
        self.y = y
        self.u = u

    ## -------------------------------------------------
    ## Create Model -- New functions
    ##--------------------------------------------------

    def new_model(self):

        if self.type == 'regr':
            self.newregr()
        elif self.type == 'bjtf':
            self.newbjtf()
        elif self.type == 'arx':
            self.newarx()
        elif self.type == 'arma':
            self.newarma()
        elif self.type == 'armax':
            self.newarmax()

    #----------------------------------------------------

    def newregr(self):

        if len(self.nb)>1:
            xerror = 'nb must contain only one element'
            raise Exception(xerror)

        if len(self.delay)==0:
            self.delay=np.zeros(self.nb[0])

        self.a = [[]]
        self.b = [np.zeros(self.nb[0] + 1)]
        self.c = []
        self.d = []
        self.f = []

        if len(self.delay) != int(self.nb[0]):
            xerror= 'delay and nb must have the same # of terms.'
            raise Exception(xerror)

        self.diff = np.array([[]])
        self.period = np.array([[]])

    #-----------------------------------------------------

    def newbjtf(self):

        for i in range(len(self.nb)):
            if self.nb[i] < 0:
                xerror= 'All nb(i) must be positive integers.'
                raise Exception(xerror)

        for i in range(len(self.nc)):
            if self.nc[i] < 0:
                xerror='All nc(i) must be positive integers.'
                raise Exception(xerror)

        for i in range(len(self.nd)):
            if self.nd[i] < 0:
                error='All nd(i) must be positive integers.'
                raise Exception(error)

        for i in range(len(self.nf)):
            if self.nf[i] < 0:
                error='All nf(i) must be positive integers.'
                raise Exception(error)

        self.a = [[]]

        nnb = len(self.nb)

        for i in range(nnb):
            self.b.append(np.zeros(int(self.nb[i]) + 1))

        nnc = len(self.nc)

        for i in range(nnc):
            self.c.append(np.zeros(int(self.nc[i])))

        nnd = len(self.nd)
        if nnd != nnc:
            xerror='nc and nd must have the same # of terms.'
            raise Exception(xerror)

        for i in range(nnd):
            self.d.append(np.zeros(int(self.nd[i])))

        nnf = len(self.nf)
        if nnf != nnb:
            xerror='nf and nb must have the same # of terms.'
            raise Exception(xerror)

        for i in range(nnf):
            self.f.append( np.zeros(int(self.nf[i])))

        if len(self.delay) != nnb:
            xerror='delay and nb must have the same # of terms.'
            raise Exception(xerror)

        if len(self.diff) != nnc:
            xerror='nc and diff must have the same # of terms.'
            raise Exception(xerror)

        if (len(np.array(self.per)) + 1) != nnc:
            xerror= 'per must have one less term than nc.'
            raise Exception(xerror)

        # class generation
        ##self.predictmodel()  refactor in  __init__

    ##--------------------------------------------------
    def newarx(self):

        # Fixed: na is a scalar, not an array
        if self.na < 0:
            xerror='na must be positive integers.'
            raise Exception (xerror)

        if type(self.nb)  is not list:
            lg =  self.nb.shape

            if len(lg) > 1:
                self.nb.reshape(1,-1)

            self.nb  = list(self.nb[0])

        # Fixed: Check each element of nb array
        for i in range(len(self.nb)):
            if self.nb[i] < 0:
                xerror='All nb(i) must be positive integers.'
                raise Exception(xerror)

        # Fixed: na is scalar, create array of size na
        self.a = [np.zeros(self.na)]


        nnb = len(self.nb)
        for i in range(nnb):
            self.b.append(np.zeros((int(self.nb[i])+ 1)))

        self.c = [[]]
        self.d = [[]]
        self.f = [[]]

        if len(self.delay) != nnb:
            xerror='delay and nb must have the same # of terms.'
            raise Exception(xerror)

        self.diff = np.array([0])
        self.period = np.array([])

    #-----------------------------------------------------------
    def newarmax(self):
        # Fixed: na is a scalar, not an array
        if self.na < 0:
            xerror='na must be positive integers.'
            raise Exception(xerror)

        # Fixed: Check each element of nb array
        for i in range(len(self.nb)):
            if self.nb[i] < 0:
                xerror='All nb(i) must be positive integers.'
                raise Exception(xerror)

        # Fixed: nc is a scalar, not an array
        if self.nc < 0:
            xerror='nc must be positive integers.'
            raise Exception(xerror)

        self.a.append(np.zeros(self.na))
        nnb = len(self.nb)
        for i in range(nnb):
            self.b.append(np.zeros(self.nb[i] + 1))

        self.c.append(np.zeros(self.nc))
        self.d = np.array([[]])
        self.f = np.array([[]])
        if len(self.delay) != nnb:
            xerror='delay and nb must have the same # of terms.'
            raise Exception(xerror)

        self.diff = np.array([0])
        self.period = np.array([])

    def newarma(self):

        if any(self.nc) < 0:
            xerror='All nc(i) must be positive integers.'
            raise Exception(xerror)

        if any(self.nd) < 0:
            xerror='All nd(i) must be positive integers.'
            raise Exception(xerror)

        self.a = np.array([[]])
        self.b = np.array([[]])

        nnc = len(self.nc)

        for i in range(nnc):
            self.c.append(np.zeros((self.nc[i])))

        nnd = len(self.nd)

        if nnd != nnc:
            xerror='nc and nd must have the same # of terms.'
            raise Exception(xerror)

        for i in range(nnd):
            self.d.append(np.zeros((self.nd[i])))

        self.f = np.array([[]])
        self.delay = np.array([[]])

        if len(self.diff) != nnc:
            xerror='nc and diff must have the same # of terms.'
            raise Exception(xerror)

        if (len(self.per) + 1) != nnc:
            xerror='per must have one less term than nc.'
            raise Exception(xerror)

        self.upreproc = []


##--------------------------------------------------
    ## Init Model

    def init(self):

        if self.initFcn == 'initzero':
            self.initzero()
        elif self.initFcn == 'initrand':
            self.initrand()
        elif self.initFcn == 'initrandn':
            self.initrandn()


    def initzero(self):

        for i in range(len(self.a)):
            self.a[i] = np.zeros(len(self.a[i]))

        for i in range(len(self.b)):
            self.b[i] = np.zeros(len(self.b[i]))

        for i in range(len(self.c)):
            self.c[i] = np.zeros(len(self.c[i]))

        for i in range(len(self.d)):
            self.d[i] = np.zeros(len(self.d[i]))

        for i in range(len(self.f)):
            self.f[i] = np.zeros(len(self.f[i]))


    def initrandn(self):
        variance = 0.125;

        for i in range(len(self.a)):
            self.a[i] = variance * (np.random.randn(self.a[i].shape))

        for i in range(len(self.b)):
            self.b[i] = variance * (np.random.randn(self.b[i].shape))

        for i in range(len(self.c)):
            self.c[i] = variance * (np.random.randn(self.c[i].shape))

        for i in range(len(self.d)):
            self.d[i] = variance * (np.random.randn(self.d[i].shape))

        for i in range(len(self.f)):
            self.f[i] = variance * (np.random.randn(self.f[i].shape))


    def initrand(self):

        llim = -0.125
        hlim = 0.125

        xrange = hlim - llim;
        bias = llim / xrange;

        for i in range(len(self.a)):
            self.a[i] = xrange * (np.random.rand(len(self.a[i]))+bias)

        for i in range(len(self.b)):
            self.b[i] = xrange * (np.random.rand(len(self.b[i]))+bias)

        for i in range(len(self.c)):
            self.c[i] = xrange * (np.random.rand(len(self.c[i]))+bias)

        for i in range(len(self.d)):
            self.d[i] = xrange * (np.random.rand(len(self.d[i]))+bias);


        for i in range(len(self.f)):
            self.f[i] = xrange * (np.random.rand(len(self.f[i]))+bias);

    ## --- End init functions
    ##--------------------------------------------------
    def getmX(self):

        if self.type == 'bjtf':
            X = self.getmXbjtf()
        elif self.type == 'regr':
            X = self.getmXregr()
        elif self.type == 'arma':
            X = self.getmXarma()
        elif self.type == 'armax':
            X = self.getmXarmax()
        elif self.type == 'arx':
            X = self.getmXarx()

        return X

    def getmXregr(self):
        X = self.b[0]

        return X

    def getmXbjtf(self):
        X = np.array([])
        nnb = len(self.b);

        for i in range(nnb):
            X = np.append( X,np.array(self.b[i]).transpose())

        nnc = len(self.c)
        for i in range(nnc):
            X = np.append(X, np.array(self.c[i]).transpose())

        nnd = len(self.d)
        for i in range(nnd):
            X = np.append(X, np.array(self.d[i]).transpose())

        nnf = len(self.f)
        for i in range(nnf):
            X = np.append(X, np.array(self.f[i]).transpose())

        return X

    def getmXarx(self):
        X = []

        X = [np.transpose(self.a[0])]
        nnb = len(self.b)
        for i in range(nnb):
            X = np.append(X, np.transpose(self.b[i]))

        return X

    def getmXarmax(self):
        X = []

        X = np.append(X, np.transpose(self.a[0]))
        nnb = len(self.b)
        start = 0;
        for i in range(nnb):
            X = np.append(X, np.transpose(self.b[i]))

        X = np.append(X,np.transpose(self.c[0]))

        return X

    def getmXarma(self):
        X = []

        nnc = len(self.c)
        for i in range(nnc):
            X = np.append(X, np.transpose(self.c[i]))
        nnd = len(self.d)
        for i in range(nnd):
            X = np.append(X, np.transpose(self.d[i]))

        return X

    ##-------------------------------------------------
    ##  SETMX functions -------------------------------
    #==================================================
    def setmX(self,X):
        setFcn = 'self.setmX' +self.type

        if self.type =='regr':
            self.setmXregr(X)
        elif self.type =='bjtf':
            self.setmXbjtf(X)
        elif self.type == 'arx':
            self.setmXarx(X)
        elif self.type == 'arma':
            self.setmXarma(X)
        elif self.type =='armax':
            self.setmXarmax(X)

    def setmXregr(self,X):
        self.b[0] = np.transpose(X[:len(self.b[0])])

        # clear all other unused parameters

        self.a = np.array([[]])
        self.c = np.array([[]])
        self.d = np.array([[]])
        self.f = np.array([[]])

    def setmXbjtf(self,X):
        nnb = len(self.b)
        start = 0

        for i in range(nnb):
            order = len(self.b[i])
            self.b[i] = list(X[start:start + order].transpose())
            start = start + order

        nnc = len(self.c)
        for i in range(nnc):
            order = len(self.c[i])
            self.c[i] = list(X[start:start + order].transpose())
            start = start + order

        nnd = len(self.d)
        for i in range(nnd):
            order = len(self.d[i])
            self.d[i] = X[start:start + order].transpose()
            start = start + order

        nnf = len(self.f)
        for i in range(nnf):
            order = len(self.f[i])
            self.f[i] = X[start:start + order].transpose()
            start = start + order

        # clear all other unused parameters
        self.a = np.array([[]])


    def setmXarx(self,X):

        start = 0
        order = len(self.a[0])
        self.a[0] = np.transpose(X[start:(start + order)])
        start = start + order;
        nnb = len(self.b)
        for i in range(nnb):
            order = len(self.b[i])
            self.b[i] = np.transpose(X[start:(start + order)])
            start = start + order

        self.c = []
        # clear all other unused parameters
        self.d = []
        self.f = []


    def setmXarmax(self,X):

        start = 0
        order = len(self.a[0])
        self.a[0] = np.transpose(X[start :(start + order)])
        start = start + order
        nnb = len(self.b)

        for i in range(nnb):
            order = len(self.b[i])
            self.b[i] = np.transpose(X[start:(start + order)])
            start = start + order

        order = len(self.c[0])
        self.c[0] = np.transpose(X[start:(start + order)])

        # clear all other unused parameters

        self.d = []
        self.f = []


    def setmXarma(self,X):
        start = 0;
        nnc = len(self.c)
        for i in range(nnc):
            order = len(self.c[i])
            self.c[i] = np.transpose(X[start:(start + order)])
            start = start + order

        nnd = len(self.d)
        for i in range(nnd):
            order = len(self.d[i])
            self.d[i] = np.transpose(X[start:(start + order)])
            start = start + order

        # clear all other unused parameters
        self.a = []
        self.b = []
        self.f = []

    ## predict functions ------------------------------
    ## Predict
    ## ==================================================

    def predict(self, y, u=[]):

        if  len(u)==0:
            uflag = False
        else:
            uflag = True

        predFcn = self.type

        if uflag:
            if predFcn == 'bjtf':
                yhat = self.predbjtf(y,u)
            elif predFcn == 'regr':
                yhat = self.predregr(y,u)
            elif predFcn == 'arx':
                yhat = self.predarx(y,u)
            elif predFcn == 'arma':
                yhat = self.predarma(y,u)
            elif predFcn == 'armax':
                yhat = self.predarmax(y,u)

        else:
            if predFcn == 'bjtf':
                yhat = self.predbjtf(y)
            elif predFcn == 'regr':
                yhat = self.predregr(y)
            elif predFcn == 'arx':
                yhat = self.predarx(y)
            elif predFcn == 'arma':
                yhat = self.predarma(y)
            elif predFcn == 'armax':
                yhat = self.predarmax(y)

        return yhat

    ##---------------------------------------------------------------------

    def predregr(self, y, u=np.array([[]])):
        '''
             PREDICT Compute one-step predictions for regression model.
        :param y:
        :param u:
        :return:
        '''
        num_inputs = len(self.b[0]) - 1
        m, n = u.shape
        if m != num_inputs:
            xerror = 'The number of rows of u does not match the number of elements in b'
            raise Exception(xerror)

        # Compute the prediction.

        udelay = np.zeros((m, n))

        for i in range(m):
            if len(self.delay) == 0:
                idel = 0
            else:
                idel = int(self.delay[i])

            udelay[i, :] = np.append(np.zeros((idel)), u[i, :( n - idel)])

        u1 = np.concatenate((np.ones((1,n)) , udelay))

        yhat = self.b @ u1

        return yhat

    def predbjtf(self, y, u=[] ):
        # Expand the parameter vectors into g and h form
        ng, dg, nh, dh = self.getGH()

        # Make the numerator and denominator of h the same size.
        num_inputs = len(self.b)
        lnh = len(nh)
        ldh = len(dh)
        if lnh > ldh:
            dh = np.append(dh ,np.zeros(lnh - ldh))
        elif ldh > lnh:
            nh = np.append(nh, np.zeros(ldh - lnh))

        # Make  sure that the number of inputs is correct.

        m= u.shape

        if m[0] !=num_inputs:
            xerror='The number of rows of u does not match the number of cells in b'
            raise Exception (xerror)

        # Compute the prediction.
        yhat = lfilter((nh - dh), nh, y);


        for i in range(num_inputs):
            yhat = yhat + lfilter(np.convolve(dh, ng[i], mode='full'), np.convolve(nh, dg[i],mode='full'), u[i,:])

        return yhat

    #--------------------------------------------------------------------------------------

    def predarx(self,y,u=np.array([[]])):
        # Expand the parameter vectors into g and h form
        ng, dg, nh, dh = self.getGH()

        # Make the numerator and denominator of h the same size.
        num_inputs = len(self.b)

        lnh = len(nh)
        ldh = len(dh)
        if lnh > ldh:
            dh = np.append(dh, np.zeros((1, lnh - ldh)))
        elif ldh > lnh:
            nh = np.append(nh, np.zeros((1, ldh - lnh)))


        # Make sure tha the number of inputs is correct.
        m, n = u.shape
        if m!=num_inputs:
            xerror='The number of rows of u does not match the number of cells in b'
            raise Exception(xerror)

        # Compute the prediction

        yhat = lfilter((nh - dh), nh, y);
        for i in range(num_inputs):
            yhat = yhat + lfilter(ng[i], nh, u[i,:]);

        return yhat


    def predarmax(self,y,u=[[]]):

        # Expand the parameter vectors into g and h form

        ng, dg, nh, dh = self.getGH()

        # Make the numerator and denominator of h the same size.
        num_inputs = len(self.b)
        lnh = len(nh)
        ldh = len(dh)

        if lnh > ldh:
            dh = np.append(dh, np.zeros((1, lnh - ldh)))
        elif ldh > lnh:
            nh = np.append(nh, np.zeros((1, ldh - lnh)))

        # Make sure that the number of inputs is correct.
        m, n = u.shape

        if m!=num_inputs:
            xerror='The number of rows of u does not match the number of cells in b'
            raise Exception(xerror)

        # Compute the prediction.
        yhat = lfilter((nh - dh), nh, y);
        for i in range(num_inputs):
            yhat = yhat + lfilter(ng[i], nh, u[i,:]);

        return yhat

    def predarma(self,y=[]):
        # Expand the parameter vectors into g and h form
        ng, dg, nh, dh = self.getGH()

        # Make the numerator and denominator of h the same size.
        num_inputs = len(self.b)
        lnh = len(nh)
        ldh = len(dh)
        if lnh > ldh:
            dh = np.append(dh, np.zeros(( lnh - ldh)))
        elif ldh > lnh:
            nh = np.append(nh, np.zeros((ldh - lnh)))

        # Compute the prediction.
        yhat = lfilter((nh - dh), nh, y);

        return yhat

    #------------------------------------------------------------------------------------------

    ## SEcond grupo predictdf functions ------------------------------
    ## Predictdf
    ## ==================================================


    def predictdf(self, y, u=[[]]):

        uflag = (len(u)> 0)

        predFcn ='self.preddf'+ self.type

        if uflag:
            yhat = eval(predFcn, y, u);
        else:
            yhat = eval(predFcn, y)

        return yhat

    def preddfbjtf(self, y, u):
        '''
            PREDICT Compute one-step predictions for the Box and Jenkins Transfer Function model.
            :param y:
            :param u:
            :return:
        '''

        # Expand the parameter vectors into g and h form
        ng, dg, nh, dh = self.getGHdf()

        # Make the numerator and denominator of h the same size.

        num_inputs = len(self.b)
        lnh = len(nh)
        ldh = len(dh)
        if lnh > ldh:
            dh = np.add(dh, np.zeros(1, lnh - ldh))
        elif ldh > lnh:
            nh = np.add(nh, np.zeros(1, ldh - lnh))

        # Make sure that the numbers of inputs is correct.
        m, n = u.shape
        if m != num_inputs:
            xerror = 'The number of rows of u does not match the number of cells in b'
            raise Exception(xerror)

        # Compute the prediction.
        yhat = lfilter((nh - dh), nh, y);
        for i in range(num_inputs):
            yhat = yhat + lfilter(np.convolve(dh, ng[i]), np.convolve(nh, dg[i]), u[i,:])

        return yhat


    def preddfarma(self,y):
        # Expand the parameter vectors into g and h form
        ng, dg, nh, dh = self.getGHdf()

        # Make the numerator and denominator of h the same size.

        num_inputs = len(self.b)
        lnh = len(nh)
        ldh = len(dh)
        if lnh > ldh:
            dh = np.add(dh, np.zeros(1, lnh - ldh))
        elif ldh > lnh:
            nh = np.add(nh, np.zeros(1, ldh - lnh))

        # Compute the prediction.
        yhat = lfilter((nh - dh), nh, y)

        return yhat

    # -----------------------------------------------------
    ## getGHdf
    # -----------------------------------------------------

    def getGHdf(self):
        getGHFcn = 'self.getGHdf' +self.type
        ng, dg, nh, dh = eval(getGHFcn)

        return ng, dg, nh, dh

    #--------------------------------------------------------


    def getGHdfbjtf(self):

        num_inputs = len(self.b)
        dg = np.zeros(num_inputs)
        ng = np.zeros(num_inputs)
        for i in range(num_inputs):
            ng[i] = np.add(np.zeros(1, self.delay[i]), self.b[i])
            if len(self.f[i])==0:
                dg[i] = [1]
            else:
                dg[i] = np.add([1], self.f[i])


        if len(self.c)==0:
            nh = [1]
        else:
            nh = np.add([1], self.c[1])

        if len(self.d)==0:
            dh = [1]
        else:
            dh = np.add([1], self.d[0])

        # Incorporate the differencing
        diffop = np.array([1, -1])
        if self.diff[0] == 0:
            ddh = np.array([1])
        else:
            ddh = diffop
            for i in range(1,self.diff[0]):
                ddh = np.convolve(ddh, diffop);

        dh = np.convolve(dh, ddh)

        lp = len(self.period)
        for i in range(lp):
            per = self.period[i]
            ctot = per * len(self.c[i + 1])
            nh1 = np.zeros(1, ctot)
            nh1[per:ctot] = self.c[i + 1]
            nh1 = np.add([1], nh1)
            nh = np.convolve(nh, nh1)
            dtot = per * len(self.d[i + 1])
            dh1 = np.zeros(1, dtot)
            #dh1(per: per:dtot) = pmod.d{i + 1};
            dh1[per:dtot] = self.d[i+1]
            dh1 = np.add([1] ,dh1)
            dh = np.convolve(dh, dh1)

            # Incorporate the differencing
            diffop = np.add([1], np.zeros(1, per))
            diffop[-1] = -1

            if self.diff[i + 1] == 0:
                ddh = [1]
            else:
                ddh = diffop
                for i in range(1,self.diff(i + 1)):
                    ddh = np.convolve(ddh, diffop)


            dh = np.convolve(dh, ddh)

        return ng, dg, nh, dh

    #-------------------------------------------------------
    def getGHdfarma(self):

        ng = []
        dg = []

        if len(self.c)==0:
            nh = np.array([1])
        else:
            nh = np.add([1] , self.c[0])

        if len(self.d)==0:
            dh = np.array([1])
        else:
            dh = np.add([1], self.d[1])

        # Incorporate the differencing
        diffop = np.array([1, -1])
        if self.diff[1] == 0:
            ddh = np.array([1])
        else:
            ddh = diffop

        for i in range(1,self.diff[0]):
            ddh = np.convolve(ddh, diffop)


        dh = np.convolve(dh, ddh)

        lp = len(self.period)
        for i in range(lp):
            per = self.period[i]
            ctot = per * len(self.c[i + 1])
            nh1 = np.zeros(1, ctot);
            nh1[per:, per:ctot] = self.c[i + 1]
            nh1 = np.add([1] ,nh1)
            nh = np.convolve(nh, nh1)
            dtot = per * len(self.d[i + 1])
            dh1 = np.zeros(1, dtot);
            dh1[per:, per:dtot] = self.d[i + 1]
            dh1 = np.add([1], dh1)
            dh = np.convolve(dh, dh1)
            # Incorporate the differencing
            diffop = np.add([1], np.zeros(1, per))
            diffop[-1] = -1;
            if self.diff[i + 1] == 0:
                ddh = np.array([1])
            else:
                ddh = diffop

                for i in range(1,self.diff[i + 1]):
                    ddh = np.convolve(ddh, diffop)

            dh = np.convolve(dh, ddh);

        return ng, dg, nh, dh

    #--------------------------------------------------------
    ## getGH
    #--------------------------------------------------------
    #========================================================
    def getGH(self):

        getGHFcn = self.type

        if getGHFcn == 'bjtf':
            ng, dg, nh, dh = self.getGHbjtf()
        elif getGHFcn == 'arma':
            ng, dg, nh, dh = self.getGHarma()
        elif getGHFcn == 'arx':
            ng, dg, nh, dh = self.getGHarx()
        elif getGHFcn == 'armax':
            ng, dg, nh, dh = self.getGHarmax()
        elif getGHFcn == 'arma':
            ng, dg, nh, dh = self.getGHdfarma()

        return ng,dg,nh,dh

    #--------------------------------------------------------

    def getGHbjtf(self):

        num_inputs = len(self.b)
        ng = []
        dg = []
        for i in range(num_inputs):
            # Fixed: Remove extra array wrapping of self.b[i]
            x1 = np.append(np.zeros(int(self.delay[i])), self.b[i])
            ng.append(x1)

            if len(self.f[i])==0:
                x2 = np.array([1])
                dg.append(x2)

            else:
                # Fixed: Remove extra array wrapping of self.f[i]
                x2 = np.append([1], self.f[i])
                dg.append(x2)

        if len(self.c)==0:
            nh = np.array([1])
        else:
            nh = np.append([1], self.c[0])

        if len(self.d)==0:
            dh = np.array([1])
        else:
            dh = np.append([1], self.d[0])

        lp = len(self.period)

        for i in range(lp):
            per = self.period[i]
            ctot = per * len(self.c[i+1])
            nh1 = np.zeros((1, ctot))
            n_i = 0
            xc = self.c[i+1]
            for j in range(per-1, ctot, per):
                nh1[0, j]= xc[n_i]
                n_i = n_i +1
            nh1 = np.append([1],nh1)
            nh = np.convolve(nh, nh1, mode = 'full')
            dtot = per * len(self.d[i+1])
            dh1 = np.zeros((1, dtot))
            for j in range(per-1, ctot, per):
                dh1[0,j:j+len(self.d[i+1])] = self.d[i+1]
            dh1 = np.append([1] ,dh1)
            dh = np.convolve(dh, dh1,mode='full')

        return ng, dg, nh, dh

    #---------------------------------------------------------------

    def getGHarx(self):

        num_inputs = len(self.b)
        dg = []
        ng = []
        for i in range(num_inputs):
            # Fixed: Remove extra parentheses in zeros
            ng.append(np.append(np.zeros(self.delay[i]), self.b[i]))
            if len(self.a)==0:
                dg.append(np.array([1]))
            else:
                # Fixed: Access self.a[0] since a is a list of arrays
                dg.append(np.append([1], self.a[0]))

        nh = [1]

        if  len(self.a)==0:
            dh = np.array([1])
        else:
            # Fixed: Access self.a[0] since a is a list of arrays
            dh = np.append([1], self.a[0])

        return ng, dg, nh, dh

    #---------------------------------------------------------------

    def getGHarmax(self):

        num_inputs = len(self.b)

        ng = []
        dg = []
        for i in range(num_inputs):
            # Fixed: Remove extra parentheses and array wrapping
            x1 = np.append(np.zeros(self.delay[i]), self.b[i])
            ng.append(x1)
            if len(self.a)==0:
                dg.append([1])
            else:
                x2 = np.append([1], self.a[0])
                dg.append(x2)
        if len(self.c)==0:
            nh = np.array([1])
        else:
            nh = np.append([1], self.c[0])

        if len(self.a)==0:
            dh = np.array([1])
        else:
            dh = np.append([1], self.a[0])

        return ng, dg, nh, dh



    #-------------------------------------------------------------------------------

    def getGHarma(self):

        ng = []
        dg = []

        if len(self.c)==0:
            # Fixed: np.array([1]) not np.array[1]
            nh = np.array([1])
        else:
            nh = np.append([1], self.c[0])

        if len(self.d)==0:
            dh = np.array([1])
        else:
            dh = np.append([1], self.d[0])

        lp = len(self.period)

        for i in range(lp):
            per = self.period[i]
            ctot = per * len(self.c[i + 1])
            nh1 = np.zeros((ctot))
            c_i = self.c[i+1]
            for j in range(len(c_i)):
                nh1[(j+1)*per-1] = c_i[j]
            nh1 = np.append([1] ,nh1)
            nh = np.convolve(nh, nh1)
            dtot = per * len(self.d[i + 1])
            dh1 = np.zeros((dtot))
            dh1[per:dtot] = self.d[i + 1]
            dh1 = np.append([1], dh1)
            dh = np.convolve(dh, dh1);

        return ng, dg, nh, dh
