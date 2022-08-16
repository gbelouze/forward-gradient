from typing import Callable, Tuple

import autograd.numpy as np  # type: ignore
from autograd import make_jvp, make_vjp  # type: ignore

from .samplers import rademacher

Optimizable = Callable[[np.ndarray], float]
DOptimizable = Callable[[np.ndarray], Tuple[float, np.ndarray]]


def df(f: Optimizable) -> DOptimizable:
    """Returns the first order oracle of f: `x -> f(x), grad_f(x)`"""
    df_ = make_vjp(f)

    def ret(x):
        dfx, fx = df_(x)
        return fx, dfx(1.0)

    return ret


def df_fwd(f: Optimizable, sampler=rademacher) -> DOptimizable:
    """Returns the first order oracle of f: `x -> f(x), fgrad_f(x)`
    where `fgrad` is the forward gradient of `f`.
    See http://arxiv.org/abs/2202.08587 for more details."""
    df_fwd_ = make_jvp(f)

    def ret(x):
        v = sampler(x.shape)
        fx, dfx = df_fwd_(x)(v)
        return fx, dfx * v

    return ret
