import autograd.numpy as np


def beale(x, y):
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rosenbrockND(x, n=2):
    assert n % 2 == 0
    x_even = x[::2, ...]
    x_odd = x[1::2, ...]
    return (100 * (x_odd**2 - x_even) ** 2 + (x_odd - 1) ** 2).sum(axis=0)


def styblinskiTang(x, n=2):
    assert x.shape[0] == n
    return 0.5 * (5 * x - 16 * x**2 + x**4).sum(axis=0)


def spiral(x):
    assert x.shape[0] == 3
    return np.log(
        1 + x[2] ** 2 + ((x[0] - np.sin(x[2])) ** 2 + (x[1] - np.cos(x[2])) ** 2)
    )


def explode(x, theta=np.ones(2)):
    assert (x.shape[0],) == theta.shape
    return np.einsum("i,i...", theta, x)
