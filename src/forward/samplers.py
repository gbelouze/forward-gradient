import autograd.numpy as np  # type: ignore

# from scipy.stats import ortho_group


def rademacher(size):
    # m = ortho_group.rvs(dim=size[0])
    # x = 2 * np.random.randint(2, size=size) - 1
    # return m @ x
    return 2 * np.random.randint(2, size=size) - 1.0


def gaussian(size):
    return np.random.normal(size=size)
