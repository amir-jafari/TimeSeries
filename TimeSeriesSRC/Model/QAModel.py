import os
import numpy as np

import TimeSeriesSRC as TS
from TimeSeriesSRC.Model.model import pmodel
from TimeSeriesSRC.Model.jacobian import func_jacobian as jacobian
from TimeSeriesSRC.Model.pmodsim import func_pmodsim as pmodsim

def read_data(file):
    path = os.getcwd()
    path_file = ".." + os.path.sep + "testdata" + os.path.sep + file
    data = np.genfromtxt(path_file, delimiter=',')
    return data


def test_bjtf():
    # Set parameters
    nb = [1, 1, 2]
    nc = [1, 2, 1]
    nd = [2, 1, 1]
    nf = [1, 2, 1]
    delay = [1, 2, 3];
    diff = [0, 0, 0];
    per = [3, 12];
    f = [[0.35],[-1,.3], [.5]]
    d = [[-1, 0.25],[.7],[.3]]
    c = [[0.65],[.5, -0.25],[-.6]]
    b = [[1, 1],[2 , 3] ,[1, - 1, .25]]
    u = np.random.randn(3, 200);
    e = np.random.randn(1, 200) * .5;

    xtype =  'bjtf'
    pmoda = pmodel(xtype=xtype, nb=nb, nc=nc, nd=nd, nf=nf, delay=delay, diff=diff, per=per)
    pmoda.f = f
    pmoda.d = d
    pmoda.c = c
    pmoda.b = b

    y = pmodsim(pmoda, e, u)

def test_jacobian():
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

if __name__ == '__main__':
    #test_jacobian()
    test_bjtf()