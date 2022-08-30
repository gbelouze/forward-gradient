import autograd.numpy as np


def beale(x, y):
    x = x + 3.0
    y = y + 0.5
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def banana(x, y):
    x = x + 1.0
    y = y + 1.0
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rosenbrock(x):
    a = x[1:, ...]
    b = x[:-1, ...]
    return banana(a, b).mean(axis=0)


def sphere(x):
    return np.mean(x**2, axis=0)


def hyperellipsoid(x):
    scale = np.arange(1, 1 + x.shape[0])
    if x.ndim > 1:
        scale = np.reshape(scale, [x.shape[0], *[1 for _ in x.shape[1:]]])
    return np.sum(scale * x**2, axis=0) / np.sum(scale)


def griewank(x):
    scale = np.sqrt(np.arange(1, 1 + x.shape[0]))
    if x.ndim > 1:
        scale = np.reshape(scale, [x.shape[0], *[1 for _ in x.shape[1:]]])
    return 1 + sphere(x) / 4000 - np.prod(np.cos(x / scale), axis=0)


def styblinskiTang(x):
    return 0.5 * (5 * x - 16 * x**2 + x**4).mean(axis=0)


def spiral(x):
    assert x.shape[0] == 3
    x[1] = x[1] + 1
    return np.log(
        1 + x[2] ** 2 + ((x[0] - np.sin(x[2])) ** 2 + (x[1] - np.cos(x[2])) ** 2)
    )


def explode(x, theta=np.ones(2)):
    assert (x.shape[0],) == theta.shape
    return np.einsum("i,i...", theta, x)
