import os
import numpy as np

import TimeSeriesSRC as TS


def read_data(file):
    path = os.getcwd()
    path_file = ".." + os.path.sep + "testdata" + os.path.sep + file
    data = np.genfromtxt(path_file, delimiter=',')
    data = data.reshape(1,-1)
    return data

if __name__ == '__main__':
    xtimeseries = TS.TimeSeries.TSAnalysis.TimesSeries()

    '''
    # Testing Makerow
    print ('---- Testing Makerow -----')
    data = read_data('sample2000.csv')
    u=xtimeseries.makerow(data)

    # Testing XRow
    print ('---- Testing XCORR -----')

    data = read_data('sample100.csv')
    acf=xtimeseries.xcorr(data,data, 10, 'unbiased')
    print(acf)

    print ('--- Testing GPAC -------')
    gpac_array = xtimeseries.gpac(acf,3,2)
    print(gpac_array)

    print ('--- Testing PARCOR -------')

    pacf,phi ,sigma = xtimeseries.parcor(acf,4)
    print('pacf:', pacf, ' phi:', phi, ' sigma: ', sigma)

    print ('--- Testing SDIFF -------')

    q= np.array([ i**2 for i in range(1,21)])

    yd = xtimeseries.sdiff(q,2,3)
    print('yd- more than 1', yd)

    yd = xtimeseries.sdiff(q,2,1)
    print('yd- default 1', yd)

    print ('---- Testing PLOTGPAC -------')
    #xtimeseries.plotgpac(gpac_array,'This is a test')

    print ('---- Testing IMPEST ----- ')
    y = read_data('yimpest.csv')
    u = read_data('uimpest.csv')
    g = xtimeseries.impest(u,y,5)
    print('g:',g)

    print ('---- Testing CHISQRF  -----')
    pr = xtimeseries.chisqrdf(18.3,10)
    print('pr',pr)

    print('----- Testing UNIANAL-------')


    yacf, ypacf, ygpac= xtimeseries.uniAnal(data,10,5, 5, 5, np.array([0]), np.array([]), 3)

    print( 'yacf :', yacf)
    print( 'ypacf:', ypacf)
    print( 'ygpac:' , ygpac)

    print('----- Testing MULTIANAL------')
    g,rv,g_gpac,h_gpac = xtimeseries.multiAnal(u,y)
    print ('g', g)
    print( 'rv', rv)
    print('g_gpac', g_gpac)
    print('h_gpac', h_gpac)

    '''
    print('------ Testing GCOMBVEC ------')
    print('first test')
    a1 = np.array([7,9]).reshape(1,-1)
    a2 = np.array([[1,2,3],[4,5,6]])
    y = xtimeseries.gcombvec(a1,a2)
    print(y)

    print('second test')
    a1 = np.array([7,9]).reshape(1,-1)
    a2 = np.array([[1,2,3],[4,5,6]])
    y = xtimeseries.gcombvec(a1,a2)
    print(y)

    a1 = np.array([7,9]).reshape(1,-1)
    a2 = np.array([0]).reshape(1,-1)
    y = xtimeseries.gcombvec(a1,a2)
    print(y)

    a1 = np.array( [[1., 2.],[0., 0.], [2.,2.]])
    a2 = np.array( [[0, 1]])
    y = xtimeseries.gcombvec(a1, a2)
    print(y)
