import numpy as np
import math

from ..Model.estimlm import func_estimlm as estimlm

from ..basefunctions.makerow import  func_makerow as makerow
from ..basefunctions.sepym import func_sepym as sepym
from ..basefunctions.sdiff import func_sdiff as sdiff



def estimate(pmod, y, u=np.array([]), show_plot=True, show_output=True):

    math_functions = dir(math)

    pmod.set_data(y, u)

    uflag = (len(u) > 0)

    if uflag:
        u = makerow(u)

    ystru, y, m = sepym(y)
    y = makerow(y)

    # Preprocess the sequences
    upreproc = pmod.upreproc

    pr = len(upreproc)
    if (uflag and pr != 0):
        #if (pr != 1 and pr != u.shape[1]):
        #    xerror = 'rows of upreproc should either equals 1 or the number of inputs. '
        #    raise Exception(xerror)

        for i in range(pr):
            #u = eval(upreproc[pr, i], u)
            if upreproc[i] in math_functions:
                code = 'math.{}(x)'.format(upreproc[i])
            else:
                code = '{}(x)'.format(upreproc[i])

            for j in range(len(u)):
                uj = list(u[j])
                uj = list(map(lambda x: eval(code, globals(), {'x': x}), uj))
                uj = np.array(uj)
                u[j] = uj


    ypreproc = pmod.ypreproc

    pc = len(ypreproc)  # only one output is possible
    #if (pc>1):
    #    xerror = 'ypreproc should have only one row. '
    #    raise Exception(xerror)

    if pc != 0:
        for i in range(pc):

            if ypreproc[i] in math_functions:
                code = 'math.{}(x)'.format(ypreproc[i])
            else:
                code = '{}(x)'.format(ypreproc[i])

            for j in range(len(y)):
                yj = list(y[j])
                print(yj)
                yj = list(map(lambda x: eval(code, globals(), {'x': x}), yj))
                yj = np.array(yj)
                y[j] = yj

    # Difference the sequences before estimation so the optimizer minimises MSE
    # on the stationary (differenced) series.  predict() expects pre-differenced
    # data; callers that need predictions on the original scale must difference
    # their y (and u) before calling predict().
    period = [x for x in pmod.period]
    period.insert(0, 1)
    diff = pmod.diff
    for i in range(len(diff)):
        d = diff[i]
        if d != 0:
            if uflag:
                u = sdiff(u, d, period[i])
            y = sdiff(y, d, period[i])

    # check to see if y, u are zero mean
    if abs(np.mean(y)) > (2 * np.std(y)):
        print('The desired output may not be a zero mean sequence.');

    if uflag and any(abs(np.mean(u, 1)) > (2 * np.std(u[:,1]))):
        print('Input may not be zero mean sequences.');

    # Call the appropriate estimation function
    ystru['y'] = y
    ystru['m'] = ystru['m'][0, :len(y[0])].reshape(1,-1)


    if uflag:
        if pmod.estimFcn == 'estimlm':
            pmod, trec, stat = estimlm(pmod, ystru, u, show_plot=show_plot, show_output=show_output)
    else:
        if pmod.estimFcn == 'estimlm':
            pmod, trec, stat = estimlm(pmod, ystru, show_plot=show_plot, show_output=show_output)

    return pmod, trec, stat