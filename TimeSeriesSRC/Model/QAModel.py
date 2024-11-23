import os
import numpy as np
import scipy.io
from scipy.signal import lfilter

import TimeSeriesSRC as TS
from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.jacobian import func_jacobian as jacobian
from TimeSeriesSRC.Model.pmodsim import func_pmodsim as pmodsim
from TimeSeriesSRC.Model.estimate import estimate
from TimeSeriesSRC.Model.pmodmse import func_pmodmse as pmodmse
from TimeSeriesSRC.Model.pmodaic import func_pmodaic as pmodaic
from TimeSeriesSRC.Model.pmodbic import func_pmodbic as pmodbic
from TimeSeriesSRC.Model.selpmod import func_selpmod as selpmod

def read_data(file):
    path = os.getcwd()
    path_file = ".." + os.path.sep + "TestData" + os.path.sep + file
    data = np.genfromtxt(path_file, delimiter=',')
    return data

def read_matfile(file):
    path = os.getcwd()
    path_file = ".." + os.path.sep + "TestData" + os.path.sep + file
    data = scipy.io.loadmat(path_file)
    return data
def bjtf_testing():
    np.random.seed(seed=1)
    # Set parameters
    nb = [1, 1, 2]
    nc = [1, 2, 1]
    nd = [2, 1, 1]
    nf = [1, 2, 1]
    delay = [1, 2, 3]
    diff = [0, 0, 0]
    per = [3, 12]
    f = [[0.35],[-1,.3], [.5]]
    d = [[-1, 0.25],[.7],[.3]]
    c = [[0.65],[.5, -0.25],[-.6]]
    b = [[1, 1],[2 , 3] ,[1, - 1, .25]]
    u = np.random.randn(3, 50)
    e = np.random.randn(1, 50) * .5

    xtype =  'bjtf'
    pmoda = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per=per)
    pmoda.f = f
    pmoda.d = d
    pmoda.c = c
    pmoda.b = b

    y = pmodsim(pmoda, e, u)
    print(y.shape)
    file = 'C:\\RepoVS\\TimesSeriesSRC\\TimeSeriesSRC\\testdata\\ybjtf.mat'
    mat = scipy.io.loadmat(file)
    y=mat['ans']
    print(y.shape)

    pmodb = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per=per)
    pmodb.estimParams.show = 10
    pmoda.estimParams.show = 10
    pmodb.estimParams.epochs = 1
    pmoda.estimParams.epochs = 1

    pmod1, trec, stat = estimate(pmodb, y, u)

    aa = pmod1.getmX()
    bb = pmoda.getmX()

    print('aa',aa)
    print('bb',bb)

def pmodsim_parameters_testing():
    np.random.seed(seed=1)
    # Set parameters
    nb = [1, 1, 2]
    nc = [1, 2, 1]
    nd = [2, 1, 1]
    nf = [1, 2, 1]
    delay = [1, 2, 3]
    diff = [0, 0, 0]
    per = [3, 12]
    f = [[0.35], [-1, .3], [.5]]
    d = [[-1, 0.25], [.7], [.3]]
    c = [[0.65], [.5, -0.25], [-.6]]
    b = [[1, 1], [2, 3], [1, - 1, .25]]
    #u = np.random.randn(3, 50)
    #e = np.random.randn(1, 50) * .5

    umat = read_matfile('utest.mat')
    emat = read_matfile('etest.mat')

    u = umat['u']
    e = emat['e']
    xtype = 'bjtf'
    pmoda = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per=per)
    pmoda.f = f
    pmoda.d = d
    pmoda.c = c
    pmoda.b = b

    y = pmodsim(pmoda, e, u)

    pmodb = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per=per)
    pmodb.estimParams.show = 10
    pmoda.estimParams.show = 10

    pmod1, trec, stat = estimate(pmodb, y, u)

    aa = pmod1.getmX()
    bb = pmoda.getmX()

    print('aa', aa)
    print('bb', bb)


def jacobian_testing():
    spec = {
        'models': [{
            'type': 'bjtf',
            'nb': [[1, 2]],
            'nc': [[0]],
            'nd': [[2]],
            'nf': [[0, 1]],
            'delay': [3],
            'diff': [0]}]
    }

    # Read the data from a file

    res = read_data('furnace.csv')
    #y = res[0] - np.mean(res[0])
    #u = res[1] - np.mean(res[1])

    y = np.array([-0.1867,    0.5173,   -3.4980,    3.0093,   -3.0414,   1.5948])
    u = np.array([-0.4326,   -1.6656,    0.1253,   0.2877,   -1.1465,   1.1909])
    y = y.reshape(1, -1)
    u = u.reshape(1, -1)

    xtype =  'bjtf'
    nb= [1]
    nc= [1]
    nd= [1]
    nf= [1]

    delta= 0.001
    pmodt = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf )
    je, jj, normgX = jacobian(pmodt, delta, y, u)
    print(je, jj, normgX)

def pmodsim_testing():
    e = np.array([0.9501, 0.2311,  0.6068,  0.4860,  0.8913, 0.7621])
    u = np.array([-0.4326, -1.6656,    0.1253,    0.2877, -1.1465,  1.1909])
    xtype = 'bjtf'
    nb = [1]
    nc = [1]
    nd = [1]
    nf = [1]
    pmod = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf)
    y = pmodsim(pmod, e, u)
    print(y)

def estimate_testing():
    y = np.array([-0.1867,    0.5173, -3.4980,    3.0093, -3.0414,    1.5948])
    u = np.array([-0.4326, -1.6656,    0.1253,    0.2877 ,-1.1465,    1.1909])
    y=y.reshape(1, -1)
    u=u.reshape(1, -1)
    xtype = 'bjtf'
    nb = [1]
    nc = [1]
    nd = [1]
    nf = [1]
    pmod = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf)
    pmod.estimParams.epochs = 50
    pmod.estimParams.goal = 0.01

    pmod, trec, stat= estimate(pmod, y, u)
    y2= pmod.predict(y,u)
    print(y2)

def regr_testing():
    y = np.array([-0.1867,  0.5173, -3.4980,  3.0093, -3.0414, 1.5948])
    u = np.array([-0.4326, -1.6656, 0.1253, 0.2877, -1.1465, 1.1909])

    y = y.reshape(1, -1)
    u = u.reshape(1, -1)
    xtype = 'regr'
    nb=[1]
    pmod = pmodel(xtype=xtype, nb=nb)

    pmod.estimParams.epochs = 50
    pmod.estimParams.goal = 0.01
    pmod, trec, stat = estimate(pmod, y, u)
    y2 = pmod.predict(y, u)
    ind = range(1,len(y2)+1)

    print(y2)

def regrall_testing():
    umat = read_matfile('utest.mat')
    emat = read_matfile('etest.mat')
    u = umat['u']
    e = emat['e']
    xtype = 'regr'

    nb = [3]
    b = [1.2, 0.5, 2.0, 0.8]

    pmoda = pmodel(xtype=xtype, nb=nb)
    pmoda.b = b
    y = pmodsim(pmoda,e,u)

    pmodb = pmodel(xtype=xtype, nb=nb)
    pmodb.estimParams.show = 10
    pmoda.estimParams.show = 10

    pmod1, trec, stat = estimate(pmodb, y, u)
    aa = pmod1.b[0]
    bb = pmoda.b

    print('aa',aa)
    print('bb',bb)

def arx_testing():

    #Set parameters
    nb = [1, 1, 2]
    na = 2
    delay = [1, 2, 3]
    b = [[1 ,1] ,[2, 3] ,[1, -1, .25]]
    a = [[-1, 0.25]]

    umat = read_matfile('utest.mat')
    emat = read_matfile('etest.mat')

    u = umat['u']
    e = emat['e']

    xtype = 'arx'

    pmoda = pmodel(xtype=xtype,na=na, nb=nb, delay=delay)
    pmoda.a = a
    pmoda.b = b
    y = pmodsim(pmoda, e, u)

    pmodb = pmodel(xtype, na=na, nb=nb, delay=delay);
    pmodb.estimParams.show = 20
    pmoda.estimParams.show = 20

    [pmod1, trec, stat] = estimate(pmodb, y, u)

    aa = pmod1.getmX()
    bb = pmoda.getmX()

    print('aa',aa)
    print('bb',bb)


def arma_testing():
    #Set parameters

    nc = [1, 2, 1]
    nd = [2, 1, 1]
    diff = [0, 0, 0]
    per = [3, 12];
    d = [[-1 ,0.25],[.7],[.3]]
    c = [[0.65],[.5, -0.25],[-.6]]

    emat = read_matfile('etest.mat')
    e = emat['e']

    xtype = 'arma'


    pmoda = pmodel(xtype,nc=nc, nd=nd, diff=diff, per=per);
    pmoda.d = d
    pmoda.c = c

    pmoda.ypostproc = ['exp']

    y = pmodsim(pmoda, e)

    pmodb = pmodel(xtype, nc=nc, nd=nd, diff=diff, per=per)
    pmodb.estimParams.show = 1
    pmoda.estimParams.show = 10

    pmodb.ypreproc = ['log']
    pmodb, trec, stat = estimate(pmodb, y)

    aa = pmodb.getmX()
    bb = pmoda.getmX()

    print(aa)
    print(bb)

def armax_testing():
    nb = [1, 1, 2]
    nc = [1]
    na = [2]
    delay = [1, 2,3]
    b = [[1 ,1],[2 ,3] ,[1, -1, .25]]
    c = [[0.65]]
    a = [[-1, 0.25]]
    xtype = 'armax'

    umat = read_matfile('utest.mat')
    emat = read_matfile('etest.mat')

    u = umat['u']
    e = emat['e']

    pmoda = pmodel(xtype,na=na, nb=nb, nc=nc, delay=delay)
    pmoda.a = a
    pmoda.c = c
    pmoda.b = b

    y = pmodsim(pmoda, e, u)

    pmodb = pmodel(xtype, na=na, nb=nb, nc=nc, delay=delay)
    pmodb.estimParams.show = 1;
    pmoda.estimParams.show = 20;

    pmod1, trec, stat = estimate(pmodb, y, u)

    aa = pmod1.getmX()
    bb = pmoda.getmX()

    print(aa)
    print(bb)

def pmodmse_testing():
    np.random.seed(seed=1)
    xtype = 'bjtf'
    nb = [1]
    nc = [1]
    nd = [1]
    nf = [1]
    pmod = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf)
    u = np.array([0, 0.1, 0.3, 0.6, 0.4])
    y = np.array([0.1, 0.3, 0.5, 0.8, 0.5])
    y = y.reshape(1, -1)
    u = u.reshape(1, -1)
    mse, e = pmodmse(pmod, y, u)
    print('mse : {} e :{}'.format(mse,e))

def armaSimple_testing():
    y = np.array([-0.1867,    0.5173, -3.4980,    3.0093,   - 3.0414,    1.5948])
    y = y.reshape(1,-1)
    xtype = 'arma'
    nc = [1]
    nd = [1]
    pmod = pmodel(xtype=xtype, nc=nc, nd=nd)
    pmod.setmX(np.zeros((2)))
    pmod.estimParams.epochs = 50
    pmod.estimParams.goal = 0.01
    pmod.estimParams.show = 1
    pmod1, trec, stat = estimate(pmod,y)
    aa= pmod1.getmX()
    y2 = pmod1.predict(y)

    print(aa)
    print(y2)

def aicbic_testing():

    umat = read_matfile('utest.mat')
    emat = read_matfile('etest.mat')

    u = umat['u'][0]
    e = emat['e']

    u = u.reshape(1,-1)
    y= lfilter([1,1], [1, 0.5], u) + lfilter([1, 0.8], [1,-0.8],e)

    xtype = 'bjtf'
    nb = [1]
    nc = [1]
    nd = [1]
    nf = [1]
    delay =[0]

    bic= []
    aic= []

    #for i in range(6):
    for i in [0,1,2,3,4,5]:
        nd= [i]
        pmod = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf,delay=delay)
        pmod.estimParams.epochs = 50;
        pmod.estimParams.goal = 0.01;
        pmod, trec, stat = estimate(pmod, y, u)
        X = pmod.getmX()
        print('X',X)
        bic.append(pmodbic(pmod, y, u))
        aic.append(pmodaic(pmod, y, u))

    print('BIC',bic)
    print('AIC',aic)


def prepost_testing():
    nb = [1, 1, 2]
    na = [2]
    delay = [1, 2, 3];
    b = [[1 ,1],[2, 3],[1, -1, .25]]
    a = [[-1,0.25]]

    umat = read_matfile('uprepost.mat')
    emat = read_matfile('eprepost.mat')

    u = umat['u']
    e = emat['e']

    xtype = 'arx'
    pmoda = pmodel(xtype=xtype,na=na, nb=nb, delay=delay)
    pmoda.upreproc = ['exp', 'log','log', 'exp']
    pmoda.ypostproc = ['exp']
    pmoda.a = a;
    pmoda.b = b;

    y = pmodsim(pmoda, e, u)

    pmodb = pmodel(xtype=xtype,na=na, nb=nb, delay=delay)

    pmodb.upreproc = ['exp', 'log','log', 'exp']
    pmodb.ypreproc = ['log']

    pmod1, trec, stat = estimate(pmodb, y, u)

    aa = pmod1.getmX()
    bb = pmoda.getmX()

    print(aa)
    print(bb)

def selmod_testing(ttest):
    #ARX
    #Generat3 y, u b an ARX model.

    jsonparamsarx = {
        'models': [
            {
        'type' :'arx',
        'na' : [1,2,3],
        'nb' : [1,2],
        'delay' : [0,1,2],
        'diff' :[0,2,3],
        'b' : [[2, 3]],
        'a' :  [[-1, 0.25]]
            }

        ]
    }

    jsonparamsarma = {
        'models': [
            {
                'type': 'arma',
                'nc': [1, 2],
                'nd': [0, 2],
                'per': []
            }
        ]
    }

    jsonparamsarmax = {
        'models': [
            {
                'type': 'armax',
                'na' : [1,2],
                'nb': [1],
                'nc': [1, 2],
                'nd': [1],
                'delay' : [0,1]
            }
        ]
    }


    jsonparamsbjtf = {
        'models': [
            {
                'type': 'bjtf',
                'nb': [1],
                'nc': [0,1, 2],
                'nd': [1,2],
                'nf': [1,2],
                'delay' : [1],
                'diff':[0]
            }
        ]
    }


    jsonparamsregr = {
        'models': [
            {
                'type': 'regr',
                'nb': [3],
                'delay' : [0,1]
            }
        ]
    }

    #u = np.random(1, 2000)
    #e = np.random(1, 2000) * .5

    umat = read_matfile('uprepost.mat')
    emat = read_matfile('eprepost.mat')

    u = umat['u']
    e = emat['e']
    xtype = ttest

    if ttest == 'arx':
        nb = [1]
        na = [2]
        delay = [1]
        b = [[2, 3]]
        a = [[-1, 0.25]]
        pmodr = pmodel(xtype=xtype,na=na, nb=nb, delay=delay)
        pmodr.a = a
        pmodr.b = b
        y = pmodsim(pmodr, e, u)
        jsonparams = jsonparamsarx

    elif ttest == 'arma':
        nc = [2]
        nd = [2]
        c = [[0.5, 0.8]]
        d = [[-1, 0.25]]

        pmodr = pmodel(xtype=xtype,nc=nc, nd=nd);
        pmodr.d = d
        pmodr.c = c
        y = pmodsim(pmodr, e)

        jsonparams = jsonparamsarma

    elif ttest == 'armax':

        na = [2]
        nb = [1]
        nc = [1]
        delay = [1]
        a = [[-1, 0.25]]
        b = [[2 ,3]]
        c = [[0.65]]

        pmodr = pmodel(xtype=xtype, nb=nb, nc=nc, delay=delay)
        pmodr.a = a
        pmodr.b = b
        pmodr.c = c
        y = pmodsim(pmodr, e, u);
        jsonparams = jsonparamsarmax

    elif ttest == 'bjtf':
        nb = [1, 1]
        nc = [1]
        nd = [2]
        nf = [1, 1]
        delay = [1, 1]
        diff = [0]
        per = []
        b = [[1, 1],[2,3]]
        c = [[0.65]]
        d = [[-1, 0.25]]
        f = [[0.35],[-1]]
        pmoda = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per=per)
        pmoda.f = f
        pmoda.d = d
        pmoda.c = c
        pmoda.b = b
        y = pmodsim(pmoda, e, u)
        jsonparams = jsonparamsbjtf

    elif ttest=='regr':
        nb = [3]
        delay = [1, 0, 1]
        b = [[1.2, 0.5, 2.0, 0.8]]
        pmodr = pmodel(xtype=xtype, nb=nb, delay=delay)
        pmodr.b = b
        pmodr.delay = delay
        y = pmodsim(pmodr, e, u)
        jsonparams = jsonparamsregr

    #run selpmod
    if ttest in ('arma'):
        estpmod = selpmod(jsonparams, y)
    else:
        estpmod = selpmod(jsonparams, y,u)

def gcomb_testing():
    a1 = np.array([[7],[9]])
    a2 = np.array([[1, 2, 3],[4,5,6]])
    a3 = TS.TSAnalysis.gcombvec.func_gcombvec(a1,a2)
    print(a3)


if __name__ == '__main__':

    #jacobian_testing()
    #pmodsim_testing()
    #estimate_testing()
    #pmodmse_testing()
    #bjtf_testing()
    #pmodsim_parameters_testing()
    #regr_testing()
    #regrall_testing()
    #arx_testing()
    #armaSimple_testing()
    #arma_testing()
    #armax_testing()
    #aicbic_testing()
    #prepost_testing()
    #selmodarx_testing()
    selmod_testing('arx')
    #selmod_testing('arma')
    #selmod_testing('armax')
    #selmod_testing('bjtf')
    #selmod_testing('regr')