import abc
from dataclasses import dataclass

import autograd.numpy as np  # type: ignore

from .grad import DOptimizable, Optimizable, df, df_fwd


class DidNotConverge(Exception):
    def __init__(self, loss, *args, **kwargs):
        self._loss = loss
        super().__init__()

    @property
    def loss(self):
        return self._loss


class DivergenceError(Exception):
    pass


class LockError(Exception):
    pass


@dataclass
class OptimConfig:
    momentum: float = 1.0
    forward: bool = False
    clip: bool = False  # clip the gradients to 1


class OptimUpdater(abc.ABC):
    def __init__(self):
        self.used = False

    @abc.abstractmethod
    def step(self, grad: np.ndarray):
        pass

    @abc.abstractmethod
    def dtheta(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def _use(self):
        """Acquire a lock on this updater."""
        if self.used:
            raise LockError
        self.used = True

    def _release(self):
        self.used = False
        self.reset()


class OptimContextManager:
    """Abstract base class for optimizers on a single function."""

    def __init__(self, updater: OptimUpdater, f_df: DOptimizable):
        self.updater = updater
        self.f_df = f_df
        self._loss = np.inf
        self.grad = None
        self.inside_context_management = False

    def loss(self, theta: np.ndarray) -> float:
        assert self.inside_context_management
        self._loss, self.grad = self.f_df(theta)
        return self._loss

    def dtheta(self):
        assert self.inside_context_management
        if self.grad is None:
            raise ValueError("OptimContextManager: dtheta should be called after loss.")
        self.updater.step(self.grad)
        return self.updater.dtheta()

    def __enter__(self):
        self.updater._use()
        self.inside_context_management = True
        return self

    def __exit__(self, type, value, traceback):
        self.updater._release()
        self.inside_context_management = False


class Optim:
    def __init__(self, config: OptimConfig, updater: OptimUpdater):
        self.config = config
        self.updater = updater

    def optimize(self, f: Optimizable) -> OptimContextManager:
        f_df = df_fwd(f) if self.config.forward else df(f)
        return OptimContextManager(self.updater, f_df)


class Scheduler(abc.ABC):
    """Abstract base class for descent schedulers"""

    def __init__(self, lr: float, epochs: int):
        self.lr0_ = lr
        self.lr_ = lr
        self.epochs = epochs

    def reset(self):
        self.lr_ = self.lr0_

    @property
    @abc.abstractmethod
    def lr(self):
        """Get the current learning rate"""
        pass

    @abc.abstractmethod
    def epoch(self):
        """Signifies the end of an epoch"""
        pass


class SgdUpdater(OptimUpdater):
    def __init__(self, config: OptimConfig):
        self.config = config
        self.m = None
        super().__init__()

    def step(self, grad: np.ndarray):
        if self.m is None:
            self.m = np.zeros(grad.shape)
        if self.config.clip:
            grad[np.abs(grad) > 1] = np.sign(grad[np.abs(grad) > 1])
        self.m = (1 - self.config.momentum) * self.m + self.config.momentum * grad

    def dtheta(self) -> np.ndarray:
        assert self.m is not None
        return self.m

    def reset(self):
        self.m = None


class AdamUpdater(OptimUpdater):

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def __init__(self, config: OptimConfig):
        self.config = config
        self.m = None
        self.v = None
        self.epoch = 0
        super().__init__()

    def step(self, grad: np.ndarray):
        if self.m is None:
            self.m = np.zeros(grad.shape)
        if self.v is None:
            self.v = np.zeros(grad.shape)
        if self.config.clip:
            grad[np.abs(grad) > 1] = np.sign(grad[np.abs(grad) > 1])

        self.epoch += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

    def dtheta(self) -> np.ndarray:
        assert self.m is not None
        assert self.v is not None
        mhat = self.m / (1 - self.beta1**self.epoch)
        vhat = self.v / (1 - self.beta2**self.epoch)
        return mhat / (np.sqrt(vhat) + self.epsilon)

    def reset(self):
        self.m = None
        self.v = None
        self.epoch = 0


class AdaBeliefUpdater(OptimUpdater):

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def __init__(self, config: OptimConfig):
        self.config = config
        self.m = None
        self.v = None
        self.epoch = 0
        super().__init__()

    def step(self, grad: np.ndarray):
        if self.m is None:
            self.m = np.zeros(grad.shape)
        if self.v is None:
            self.v = np.zeros(grad.shape)
        if self.config.clip:
            grad[np.abs(grad) > 1] = np.sign(grad[np.abs(grad) > 1])

        self.epoch += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad - self.m) ** 2

    def dtheta(self) -> np.ndarray:
        assert self.m is not None
        assert self.v is not None
        mhat = self.m / (1 - self.beta1**self.epoch)
        vhat = self.v / (1 - self.beta2**self.epoch)
        return mhat / (np.sqrt(vhat) + self.epsilon)

    def reset(self):
        self.m = None
        self.v = None
        self.epoch = 0


def SGD(config: OptimConfig) -> Optim:
    return Optim(config, SgdUpdater(config))


def Adam(config: OptimConfig) -> Optim:
    return Optim(config, AdamUpdater(config))


def AdaBelief(config: OptimConfig) -> Optim:
    return Optim(config, AdaBeliefUpdater(config))


def descent_fixed_cost(
    f: Optimizable,
    theta0: np.ndarray,
    target: float,
    optim: Optim,
    lr: float,
    max_epochs: int = 1_000,
):
    theta = theta0.copy()
    with optim.optimize(f) as optf:
        epoch = 0
        while (loss := optf.loss(theta)) > target:
            theta -= lr * optf.dtheta()
            epoch += 1
            if epoch >= max_epochs:
                break
            if loss > 1e5:
                raise DivergenceError
        return epoch, theta, optf.loss(theta)


def _fakeHook(epoch, x, fx, dfx, **kwargs):
    pass


class _Sgd:
    def __init__(
        self,
        f_df,
        lr=0.01,
        epochs=1,
        momentum=1,
        decay=0,
        decay_offset=0,
        threshold=False,
    ):
        self.f_df = f_df

        self.decay = decay
        self.decay_offset = decay_offset
        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.threshold = threshold

    def run(self, x0, data=None, epoch_hook=None):
        epoch_hook = epoch_hook if epoch_hook is not None else _fakeHook
        f_df = self.f_df if data is None else (lambda x: self.f_df(x, data))

        x = x0.copy()
        m = np.zeros(x.shape)
        epoch_hook(0, x, *f_df(x))
        for epoch in range(1, self.epochs + 1):
            fx, dfx = f_df(x)
            m = (1 - self.momentum) * m + self.momentum * dfx
            lr_ = self.lr / np.sqrt(max(1, (epoch - self.decay_offset) * self.decay))
            if self.threshold:
                m[np.abs(m) > 1] = np.sign(m[np.abs(m) > 1])
            x = x - lr_ * m
            epoch_hook(epoch, x, fx, m)
        return x


class _Adam:

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def __init__(
        self, f_df, lr=0.1, epochs=1, decay=0, decay_offset=0, threshold=False
    ):
        self.f_df = f_df

        self.lr = lr
        self.epochs = epochs
        self.decay = decay
        self.decay_offset = decay_offset
        self.threshold = threshold

    def run(self, x0, data=None, epoch_hook=None):
        epoch_hook = epoch_hook if epoch_hook is not None else _fakeHook

        f_df = self.f_df if data is None else (lambda x: self.f_df(x, data))

        x = x0.copy()
        m = np.zeros(x.shape)
        v = np.zeros(x.shape)

        epoch_hook(0, x.copy(), *f_df(x))
        for epoch in range(1, self.epochs + 1):
            fx, dfx = f_df(x)
            m = self.beta1 * m + (1 - self.beta1) * dfx
            v = self.beta2 * v + (1 - self.beta2) * dfx**2
            mhat = m / (1 - self.beta1**epoch)
            vhat = v / (1 - self.beta2**epoch)
            lr_ = self.lr / np.sqrt(max(1, (epoch - self.decay_offset) * self.decay))
            x -= lr_ / (np.sqrt(vhat) + self.epsilon) * mhat

            epoch_hook(epoch, x.copy(), fx, m)
        return x.copy()


class _AdaBelief:
    """Adam like optimizer, stabilised for noisy gradients.
    See https://proceedings.neurips.cc/paper/2020/hash/
    d9d4f495e875a2e075a1a4a6e1b9770f-Abstract.html"""

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    def __init__(self, f_df, lr=0.1, epochs=1):
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

    def __init__(self, f, lr=0.01, epochs=1):

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
