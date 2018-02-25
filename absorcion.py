"""Calculo de numero de platos en columna de absorcion
"""

######################
from __future__ import division
import os
from configparser import ConfigParser
import json
path = os.path.dirname(os.path.realpath(__file__))
print ("\n####Calculo de numero de platos en columna de absorcion####")
print ("\nImporting Libraries..."),
try:
    import numpy as np
    import matplotlib.pyplot as plt
    print ("Done")
except ImportError as err:
    import pip
    from subprocess import call
    print (err)
    call("pip install --upgrade numpy", shell=True)
    print ("\nDone installing dependencies")
######################

#/////////////////////////////////////////////#


def interpolated_intercept(x, y1, y2):
    """Find the intercept of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """Find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x, y

        L1 = line([point1[0], point1[1]], [point2[0], point2[1]])
        L2 = line([point3[0], point3[1]], [point4[0], point4[1]])

        R = intersection(L1, L2)

        return R

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xc, yc = intercept((x[idx], y1[idx]), (x[idx + 1], y1[idx + 1]),
                       (x[idx], y2[idx]), (x[idx + 1], y2[idx + 1]))
    return xc, yc

#/////////////////////////////////////////////#


def plates(x_, y_, p_op):
    """Returns a list of the intersections of the steps and the
    two curves
    """

    def operation(x): return (
        (((p_op[1][1] - p_op[1][0]) / (p_op[0][1] - p_op[0][0])) * x) + p_op[1][0])

    def add_pto(ptol, pto):
        ptol[0].append(pto[0])
        ptol[1].append(pto[1])

    equilibrium = [np.asarray(x_), np.asarray(y_)]
    sample_size = len(x_)
    x_max = equilibrium[0][-5]
    ptos = [[], []]
    ini_point = [0, operation(0)]
    add_pto(ptos, ini_point)

    while ini_point[0] < x_max:

        horizontal_line = np.asarray([ini_point[1]] * sample_size)
        x1, y1 = interpolated_intercept(
            equilibrium[0], horizontal_line, equilibrium[1])
        ini_point = [x1.tolist()[0][0], y1.tolist()[0][0]]
        add_pto(ptos, ini_point)

        if ini_point[0] < x_max:
            ini_point = [x1.tolist()[0][0], operation(x1.tolist()[0][0])]
            add_pto(ptos, ini_point)

    # A, B
    a = [[p_op[0][1], p_op[0][1]], [ptos[1][-1], p_op[1][1]]]
    b = [[ptos[0][-1], ptos[0][-1]], [ptos[1][-1], operation(ptos[0][-1])]]
    n_plates = ((len(ptos[0]) / 2) - 1 +
                ((a[1][1] - a[1][0]) / (b[1][1] - b[1][0])))

    return ptos, n_plates, a, b

#/////////////////////////////////////////////#


def plot(p_op, x_, y_, ptos, a, b, n_plates):

    # operation
    plt.plot(p_op[0], p_op[1],  color='blue', marker='o',
             mec='none', ms=4, lw=1, label='Operacion')
    # equilibrium
    plt.plot(x_, y_, color='orange', marker='o',
             mec='none', ms=4, lw=1, label='Equilibrio')
    # plates
    plt.plot(ptos[0], ptos[1], color='black', mec='none', ms=4,
             lw=1, label='plates: {0:.3f}'.format(round(n_plates, 3)))
    # a
    plt.plot(a[0], a[1], color='green', ls='--', lw=1)
    # b
    plt.plot(b[0], b[1], color='green', ls='--', lw=1)
    # extension
    plt.plot([a[0][0], b[0][0]], [a[1][1], b[1][1]],
             color='blue', marker='o', ms=4, ls='--', lw=1)

    plt.legend(frameon=False, fontsize=10, numpoints=1, loc='upper left')
    plt.show()


#/////////////////////////////////////////////#
def main():
    """
    Main instance
    """

    print ("\nLoading variables from file..."),
    config = ConfigParser()
    config.read('%s/variables.txt' % path)
    x_ = json.loads(config.get("equilibrio", "x"))
    y_ = json.loads(config.get("equilibrio", "y"))
    p_op = json.loads(config.get("operacion", "puntos"))
    print ("Done")

    ptos, n_plates, a, b = plates(x_, y_, p_op)

    while ptos[0][-1] < x_[-1] and ptos[0][-1] < x_[-2]:
        del x_[-1]
        del y_[-1]

    plot(p_op, x_, y_, ptos, a, b, n_plates)


######################
if __name__ == '__main__':
    main()
