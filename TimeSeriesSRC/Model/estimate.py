import numpy as np

from ..Model.estimlm import func_estimlm as estimlm

from ..basefunctions.makerow import  func_makerow as makerow
from ..basefunctions.sepym import func_sepym as sepym
from ..basefunctions.sdiff import func_sdiff as sdiff



def estimate(pmod, y, u=[]):

    pmod.set_data(y, u)

    uflag = len(u) > 0

    if uflag:
        u = makerow(u)

    ystru, y, m = sepym(y)
    y = makerow(y)

    # Preprocess the sequences
    upreproc = pmod.upreproc

    if len(upreproc[0])>0 and len(upreproc.shape)==2:
        pr = upreproc.shape[0]

        if (uflag and pr != 0):
            if (pr != 1 and pr != u.shape[1]):
                xerror = 'rows of upreproc should either equals 1 or the number of inputs. '
                raise Exception(xerror)

            if pr == 1:
                pc = upreproc[pr-1, :]
                for i in range(pc):
                    u = eval(upreproc[pr, i], u)

            else:
                for i in range(pr):
                    pc = len(upreproc[i, :])
                    for j in range(pc):
                        if len(upreproc[i, j]) > 0:
                            u[i, :] = eval(upreproc[i, j], u[i, :])

        ypreproc = pmod.ypreproc
        pc = ypreproc.shape[1]  # only one output is possible
        if (pc and ypreproc.shape[0] != 1):
            xerror = 'ypreproc should have only one row. '
            raise Exception(xerror)

        if pc != 0:
            for i in range(pc):
                y = eval(ypreproc[i], y)

    # Difference the sequences
    period = [1, pmod.period]
    diff = pmod.diff
    for i in range(len(diff)):
        d = diff[i]
        if (d != 0):
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
            pmod, trec, stat = estimlm(pmod,ystru,u)
    else:

        if pmod.estimFcn == 'estimlm':
            pmod, trec, stat = estimlm(pmod, ystru)

    return pmod, trec, stat