import autograd.numpy as np  # type: ignore


def rademacher(size):
    return 2 * np.random.randint(2, size=size) - 1


def gaussian(size):
    return np.random.normal(size=size)
