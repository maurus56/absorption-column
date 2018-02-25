from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def interpolated_intercept(x, y1, y2):
    """Find the intercept of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
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


def steps(x, sample_size, operation, equilibrium):

    def add_pto(ptol, pto):
        ptol[0].append(pto[0])
        ptol[1].append(pto[1])

    x_max = x[-5]
    ptos = [[], []]
    ini_point = [0, operation(0)]
    add_pto(ptos, ini_point)

    while ini_point[0] < x_max:

        horizontal_line = np.asarray([ini_point[1]] * sample_size)
        x1, y1 = interpolated_intercept(x, horizontal_line, equilibrium)
        ini_point = [x1.tolist()[0][0], y1.tolist()[0][0]]
        add_pto(ptos, ini_point)

        if ini_point[0] < x_max:
            ini_point = [x1.tolist()[0][0], operation(x1.tolist()[0][0])]
            add_pto(ptos, ini_point)

    return ptos


def plot(p_op, x_, y_, ptos, a, b, n_steps):
    
    # curva operacion
    plt.plot(p_op[0], p_op[1],  color='blue', marker='o', mec='none', ms=4, lw=1, label='Operacion')
    # curva equilibrio
    plt.plot(x_, y_, color='orange', marker='o', mec='none', ms=4, lw=1, label='Equilibrio')
    # curva platos
    plt.plot(ptos[0], ptos[1], color='black', mec='none', ms=4, lw=1, label='Steps: {0:.3f}'.format(round(n_steps, 3)))
    # curva a
    plt.plot(a[0], a[1], color='green', ls='--', lw=1)
    # curva b
    plt.plot(b[0], b[1], color='green', ls='--', lw=1)
    # extension operacion
    plt.plot([a[0][0],b[0][0]], [a[1][1],b[1][1]], color='blue', marker='o', ms=4, ls='--', lw=1)

    plt.legend(frameon=False, fontsize=10, numpoints=1, loc='upper left')
    plt.show()


def main():
    # entrada
    x_ = [0.0000562, 0.0001403, 0.00028, 0.000422, 0.000564, 0.000842,
          0.001403, 0.001965, 0.00279, 0.0042, 0.00698, 0.01385, 0.0206, 0.0273]
    y_ = [0.00065789, 0.00157895, 0.00421053, 0.00763158, 0.01118421, 0.01855263, 0.03421053,
          0.05131579, 0.07763158, 0.12105263, 0.21184211, 0.44210526, 0.68026316, 0.91842105]
    # entrada
    p_op = [[0, 0.00355], [0.02, 0.2]]


    sample_size = len(x_)
    #x = np.asarray(x_)

    def operation(x): return (
        (((p_op[1][1] - p_op[1][0]) / (p_op[0][1] - p_op[0][0])) * x) + p_op[1][0])
    equilibrium = [np.asarray(x_) ,np.asarray(y_)]

    ptos = steps(equilibrium[0], sample_size, operation, equilibrium[1])

    # A y B
    a = [ [p_op[0][1], p_op[0][1]], [ptos[1][-1], p_op[1][1]] ]
    b = [ [ptos[0][-1], ptos[0][-1]], [ptos[1][-1], operation(ptos[0][-1])] ]
    n_steps = ( (len(ptos[0])/2) - 1 + ((a[1][1]-a[1][0])/(b[1][1]-b[1][0])) )

    while ptos[0][-1] < x_[-1] and ptos[0][-1] < x_[-2]:
        del x_[-1]
        del y_[-1]

    plot(p_op, x_, y_, ptos, a, b, n_steps)


if __name__ == '__main__':
    main()
