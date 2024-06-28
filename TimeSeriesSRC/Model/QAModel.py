import os
import numpy as np

import TimeSeriesSRC as TS


def read_data(file):
    path = os.getcwd()
    path_file = ".." + os.path.sep + "testdata" + os.path.sep + file
    data = np.genfromtxt(path_file, delimiter=',')
    return data

if __name__ == '__main__':
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
    y = res[0] - np.mean(res[0])
    u = res[1] - np.mean(res[1])

    y = y.reshape(1, -1)
    u = u.reshape(1, -1)

    estpmod = TS.selpmod.func_selpmod(spec, y, u)

    print(type(estpmod))