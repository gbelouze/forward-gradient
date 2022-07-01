from autograd import make_jvp, make_vjp  # type: ignore

from .samplers import rademacher


def df(f, with_data_args=False):
    """Returns the first order oracle of f: `x -> f(x), grad_f(x)`"""
    df_ = make_vjp(f)

    def ret(x):
        dfx, fx = df_(x)
        return fx, dfx(1.0)

    def retData(x, data):
        dfx, fx = df_(x, data)
        return fx, dfx(1.0)

    return retData if with_data_args else ret


def df_fwd(f, sampler=rademacher, with_data_args=False):
    """Returns the first order oracle of f: `x -> f(x), fgrad_f(x)`
    where `fgrad` is the forward gradient of `f`.
    See http://arxiv.org/abs/2202.08587 for more details."""
    df_fwd_ = make_jvp(f)

    def ret(x):
        v = sampler(x.shape)
        fx, dfx = df_fwd_(x)(v)
        return fx, dfx * v

    def retData(x, data):
        v = sampler(x.shape)
        fx, dfx = df_fwd_(x, data)(v)
        return fx, dfx * v

    return retData if with_data_args else ret
