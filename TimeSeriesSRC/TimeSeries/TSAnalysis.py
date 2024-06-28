from ..basefunctions import makerow
from ..basefunctions import xcorr
from ..basefunctions import partoacf
from ..basefunctions import parcor
from ..basefunctions import gpac
from ..basefunctions import sdiff
from ..basefunctions import chisqrdf
from ..basefunctions import plotgpac
from ..basefunctions import impest
from ..basefunctions import uniAnal
from ..basefunctions import multiAnal
from ..basefunctions import gcombvec

class TimesSeries:
    """ Class contains all the functions for Times Series Analysis

        Prediction Model Toolbox.
        Version 0.1.0  2022
        Translation from MatLab to Python

        Analysis functions.
            gpac      - Compute the GPAC for a given autocorrelation function.
            impest    - Estimate impulse response between two time series.
            parcor    - Compute the partial autocorrelation function.
            partoacf  -

        Pre and Post Processing.
            sdiff     - Difference a time series.

        Utility functions.
            chisqrdf - Calculate Chi square cumulative density function
            makerow  - Convert matrix so that it contains more columns than rows.
            XCORR    - Calculate autocorrelation function

    """

    def makerow(self,y):
        yr = makerow.func_makerow(y)
        return yr

    def xcorr(self, a,b,maxlags,flag):
        c = xcorr.func_xcorr(a,b,maxlags,flag)
        return c

    def partoacf(self,phi, theta, lagmax, var_a):
        acf, imp = partoacf.func_partoacf(phi,theta,lagmax,var_a)
        return acf, imp

    def parcor(self, acf,nump):
        pacf, phi, sigma = parcor.func_parcor(acf, nump)
        return pacf, phi, sigma

    def gpac(self, acf, nrows, ncols):
        gpac_array = gpac.func_gpac(acf, nrows, ncols)
        return gpac_array

    def sdiff(self, y, d, p):
        yd = sdiff.func_sdiff(y,d,p)
        return yd

    def chisqrdf(self, q,n):
        pr = chisqrdf.func_chisqrdf(q,n)
        return pr

    def plotgpac(self, gpac,title):
        plotgpac.func_plotgpac(gpac,title)

    def impest(self, u,y, k ):
        g = impest.func_impest(u,y,k)
        return g

    def uniAnal(self, y,na=20,nump=10,nrg=5,ncg=0,diff=[0],per=[],perdsp=1):
        yacf,ypacf,ygpac= uniAnal.func_uniAnal(y,na,nump,nrg,ncg,diff,per,perdsp)
        return yacf, ypacf, ygpac

    def multiAnal(self,u,y,nng=5,ndg=5,nnh=5,ndh=5,lg=20,lh=20):
        g,rv,g_gpac,h_gpac = multiAnal.func_multiAnal(u,y,nng,ndg,nnh,ndh,lg,lh)
        return g,rv,g_gpac,h_gpac

    def gcombvec(self,a1,*argv):
        y = gcombvec.func_gcombvec(a1, *argv);
        return y