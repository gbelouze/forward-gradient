import autograd.numpy as np  # type: ignore


def _fakeHook(epoch, x, fx, dfx, **kwargs):
    pass


class _Sgd:
    def __init__(self, f_df, lr=0.01, epochs=100, momentum=1):
        self.f_df = f_df

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
            x = x - lr * m
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


def sgd(**kwargs):
    def ret(*args, **kwargs_):
        return _Sgd(*args, **kwargs, **kwargs_)

    return ret


def adam(**kwargs):
    def ret(*args, **kwargs_):
        return _Adam(*args, **kwargs, **kwargs_)

    return ret
