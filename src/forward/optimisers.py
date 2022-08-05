import autograd.numpy as np  # type: ignore

from .grad import df_fwd


def _fakeHook(epoch, x, fx, dfx, **kwargs):
    pass


class _Sgd:
    def __init__(self, f_df, lr=0.01, epochs=100, momentum=1, decay=0):
        self.f_df = f_df

        self.decay = decay
        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum

    def run(self, x0, data=None, lr=None, epochs=None, momentum=None, epoch_hook=None):
        epoch_hook = epoch_hook if epoch_hook is not None else _fakeHook
        lr = lr if lr is not None else self.lr
        epochs = epochs if epochs is not None else self.epochs
        momentum = momentum if momentum is not None else self.momentum

        f_df = self.f_df if data is None else (lambda x: self.f_df(x, data))

        x = x0.copy()
        m = np.zeros(x.shape)
        epoch_hook(0, x, *f_df(x))
        for epoch in range(1, epochs + 1):
            fx, dfx = f_df(x)
            m = (1 - momentum) * m + momentum * dfx
            lr_ = lr * np.exp(-epoch * self.decay)
            x = x - lr_ * m
            epoch_hook(epoch, x, fx, m)
        return x


class _Adam:

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def __init__(self, f_df, lr=0.01, epochs=100):
        self.f_df = f_df

        self.lr = lr
        self.epochs = epochs

    def run(self, x0, data=None, lr=None, epochs=None, epoch_hook=None):
        epoch_hook = epoch_hook if epoch_hook is not None else _fakeHook
        lr = lr if lr is not None else self.lr
        epochs = epochs if epochs is not None else self.epochs

        f_df = self.f_df if data is None else (lambda x: self.f_df(x, data))

        x = x0.copy()
        m = np.zeros(x.shape)
        v = np.zeros(x.shape)

        epoch_hook(0, x.copy(), *f_df(x))
        for epoch in range(1, epochs + 1):
            fx, dfx = f_df(x)
            m = self.beta1 * m + (1 - self.beta1) * dfx
            v = self.beta2 * v + (1 - self.beta2) * dfx**2
            mhat = m / (1 - self.beta1**epoch)
            vhat = v / (1 - self.beta2**epoch)
            x -= lr / (np.sqrt(vhat) + self.epsilon) * mhat

            epoch_hook(epoch, x.copy(), fx, m)
        return x.copy()


class _AdaBelief:
    """Adam like optimiser, stabilised for noisy gradients.
    See https://proceedings.neurips.cc/paper/2020/hash/
    d9d4f495e875a2e075a1a4a6e1b9770f-Abstract.html"""

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def __init__(self, f_df, lr=0.01, epochs=100):
        self.f_df = f_df

        self.lr = lr
        self.epochs = epochs

    def run(self, x0, data=None, lr=None, epochs=None, epoch_hook=None):
        epoch_hook = epoch_hook if epoch_hook is not None else _fakeHook
        lr = lr if lr is not None else self.lr
        epochs = epochs if epochs is not None else self.epochs

        f_df = self.f_df if data is None else (lambda x: self.f_df(x, data))

        x = x0.copy()
        m = np.zeros(x.shape)
        s = np.zeros(x.shape)

        epoch_hook(0, x.copy(), *f_df(x))
        for epoch in range(1, epochs + 1):
            fx, dfx = f_df(x)
            m = self.beta1 * m + (1 - self.beta1) * dfx
            s = self.beta2 * s + (1 - self.beta2) * (dfx - m) ** 2 + self.epsilon
            mhat = m / (1 - self.beta1**epoch)
            shat = s / (1 - self.beta2**epoch)
            x -= lr / (np.sqrt(shat) + self.epsilon) * mhat

            epoch_hook(epoch, x.copy(), fx, m)
        return x.copy()


class _AdaForward:

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def __init__(self, f, lr=0.01, epochs=100):

        self.f_df = df_fwd(f, self.sampler)
        self.mv_hat = None
        self.lr = lr
        self.epochs = epochs

    def sampler(self, _size):
        assert self.mv_hat is not None
        return np.random.normal(loc=self.mv_hat)

    def run(self, x0, data=None, lr=None, epochs=None, epoch_hook=None):
        epoch_hook = epoch_hook if epoch_hook is not None else _fakeHook
        lr = lr if lr is not None else self.lr
        epochs = epochs if epochs is not None else self.epochs

        f_df = self.f_df if data is None else (lambda x: self.f_df(x, data))

        x = x0.copy()
        m = np.zeros(x.shape)
        v = np.zeros(x.shape)
        self.mv_hat = np.zeros(x.shape)

        epoch_hook(0, x.copy(), *f_df(x))
        for epoch in range(1, epochs + 1):
            fx, dfx = f_df(x)
            m = self.beta1 * m + (1 - self.beta1) * dfx
            v = self.beta2 * v + (1 - self.beta2) * dfx**2
            mhat = m / (1 - self.beta1**epoch)
            vhat = v / (1 - self.beta2**epoch)
            self.mv_hat = mhat / (np.sqrt(vhat) + self.epsilon)
            x -= lr * self.mv_hat

            epoch_hook(epoch, x.copy(), fx, m)
        return x.copy()


def sgd(**kwargs):
    def ret(*args, **kwargs_):
        return _Sgd(*args, **kwargs, **kwargs_)

    return ret


def adam(**kwargs):
    def ret(*args, **kwargs_):
        return _Adam(*args, **kwargs, **kwargs_)

    return ret


def adabelief(**kwargs):
    def ret(*args, **kwargs_):
        return _AdaBelief(*args, **kwargs, **kwargs_)

    return ret


def adaforward(**kwargs):
    def ret(*args, **kwargs_):
        return _AdaForward(*args, **kwargs, **kwargs_)

    return ret
