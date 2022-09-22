from typing import Callable, Iterable, Union

import autograd.numpy as np  # type: ignore

Size = Union[int, Iterable[int]]
Sampler = Callable[[Size], np.ndarray]


def rademacher(size: Size) -> np.ndarray:
    # m = ortho_group.rvs(dim=size[0])
    # x = 2 * np.random.randint(2, size=size) - 1
    # return m @ x
    return 2 * np.random.randint(2, size=size) - 1.0


def gaussian(size: Size) -> np.ndarray:
    return np.random.normal(size=size)


def circle(size: Size) -> np.ndarray:
    ret = np.random.normal(size=size)
    return np.sqrt(size) * ret / np.linalg.norm(ret)


all_samplers: dict[str, Sampler] = {
    "rademacher": rademacher,
    "gaussian": gaussian,
    "circle": circle,
}
