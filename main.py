import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return np.cos(np.exp(x/2)/25)


def lagrange(x, xknown, yknown):
    result = 0
    for i in range(len(yknown)):
        res = 1
        for k in range(len(xknown)):
            if k != i:
                res *= (x - xknown[k])/(xknown[i] - xknown[k])
        result += yknown[i] * res
    return result


def lebesgueConst(x, xknown, yknown):
    lebConst = 0
    result = 0
    for i in range(len(yknown)):
        res = 1
        for k in range(len(xknown)):
            if k != i:
                res *= (x - xknown[k])/(xknown[i] - xknown[k])
        result += yknown[i] * res
        lebConst = max(max(np.absolute(res)), lebConst)
    return result, lebConst


def interpolate(N, x0, xend, step):
    h = (xend - x0) / (N - 1)
    xknown = np.arange(x0, xend + h, h)
    xrange = np.arange(x0, xend, step)
    yknown = func(xknown)
    tmparray, lebConst = lebesgueConst(xrange, xknown, yknown)
    interpolated = np.array(tmparray)

    xknownChebyshev = np.zeros(N)
    for k in range(N):
        xknownChebyshev[k] = (1 / 2) * (x0 + xend) + (1 / 2) * (xend - x0) * (
            np.cos(((2 * (k + 1) - 1) / (2 * N)) * np.pi))
    yknownChebyshev = func(xknownChebyshev)
    tmparray, lebConstChebyshev = lebesgueConst(xrange, xknownChebyshev, yknownChebyshev)
    interpolatedChebyshev = np.array(tmparray)

    truevalues = func(xrange)

    return interpolated, interpolatedChebyshev, truevalues, lebConst, lebConstChebyshev


def show(N, x0, xend, step, interpolated, interpolatedChebyshev, truevalues):
    h = (xend - x0) / (N - 1)
    xknown = np.arange(x0, xend + h, h)
    xrange = np.arange(x0, xend, step)
    yknown = func(xknown)

    xknownChebyshev = np.zeros(N)
    for k in range(N):
        xknownChebyshev[k] = (1 / 2) * (x0 + xend) + (1 / 2) * (xend - x0) * (
            np.cos(((2 * (k + 1) - 1) / (2 * N)) * np.pi))
    yknownChebyshev = func(xknownChebyshev)

    plt.subplot(2, 2, 1)
    plt.title("Равномерный шаг")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.plot(xrange, truevalues, color='k', label='Истинное значение')
    plt.plot(xrange, interpolated, ls='--', color='k', label='Восстановленное значение')
    plt.plot(xknown, yknown, ls='', marker='.', color='k', label='Узлы интерполяции')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("")
    plt.xlabel("x")
    plt.ylabel("|Δy|")
    plt.grid()
    plt.plot(xrange, abs(truevalues - interpolated), color='k', label='Абсолютная погрешность')
    plt.plot(xknown, [0] * len(xknown), ls='', marker='.', color='k', label='Узлы интерполяции')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Узлы Чебышева")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.plot(xrange, truevalues, color='k', label='Истинное значение')
    plt.plot(xrange, interpolatedChebyshev, ls='--', color='k', label='Восстановленное значение')
    plt.plot(xknownChebyshev, yknownChebyshev, ls='', marker='.', color='k', label='Узлы интерполяции')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("")
    plt.xlabel("x")
    plt.ylabel("|Δy|")
    plt.grid()
    plt.plot(xrange, abs(truevalues - interpolatedChebyshev), color='k', label='Абсолютная погрешность')
    plt.plot(xknownChebyshev, [0] * len(xknownChebyshev), ls='', marker='.', color='k', label='Узлы интерполяции')
    plt.legend()

    plt.show()


def main():

    Nrange = range(2, 101)
    x0 = 0
    xend = 10
    step = 0.001

    i = 0
    lebConst = np.zeros(len(Nrange))
    lebConstChebyshev = np.zeros(len(Nrange))

    for N in Nrange:
        print(N)
        interpolated, interpolatedChebyshev, truevalues, lebConst[i], lebConstChebyshev[i] = interpolate(N, x0, xend, step)
        i += 1

    upperBound = np.zeros(len(Nrange))
    lowerBound = np.zeros(len(Nrange))
    upperBound[0] = 2
    lowerBound[0] = 1/(4 * np.sqrt(2))

    for i in range(1, len(Nrange)):
        upperBound[i] = 2 * upperBound[i - 1]
        lowerBound[i] = (upperBound[i - 1] / 2) / ((i+2)**(3/2))

    upperBoundChebyshev = np.zeros(len(Nrange))
    upperBoundChebyshev[0] = 1 + (2 * np.log(2)) / np.pi
    for i in range(1, len(Nrange)):
        upperBoundChebyshev[i] = 1 + (2 * np.log(i + 2)) / np.pi

    lebConst = np.log(lebConst)
    upperBound = np.log(upperBound)
    lowerBound = np.log(lowerBound)

    plt.subplot(1, 2, 1)
    plt.title("Равномерная сетка")
    plt.xlabel("N")
    plt.ylabel("log(Lebesgue constant)")
    plt.grid()
    plt.plot(Nrange, lebConst, color='k', label='Равномерная сетка')
    plt.plot(Nrange, upperBound, ls='--', color='k', label='Верхняя граница')
    plt.plot(Nrange, lowerBound, ls='--', color='k', label='Нижняя граница')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Узлы Чебышева")
    plt.xlabel("N")
    plt.ylabel("Lebesgue constant")
    plt.grid()
    plt.plot(Nrange, lebConstChebyshev, color='k', label='Узлы Чебышева')
    plt.plot(Nrange, upperBoundChebyshev, ls='--', color='k', label='Верхняя граница')
    plt.legend()

    plt.show()


    #show(N, x0, xend, step, interpolated, interpolatedChebyshev, truevalues)


main()
