import abc
from dataclasses import dataclass
from typing import Optional

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

        assert self.m is not None  # make mypy happy

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

        assert self.m is not None
        assert self.v is not None  # make mypy happy

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

        assert self.m is not None
        assert self.v is not None  # make mypy happy

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


def project(domain: np.ndarray, theta: np.ndarray):
    under = theta < domain[0, :]
    over = theta > domain[1, :]
    theta[under] = domain[0, under]
    theta[over] = domain[1, over]
    return theta


def descent(
    f: Optimizable,
    theta0: np.ndarray,
    target: float,
    optim: Optim,
    lr: float,
    max_epochs: int = 1_000,
    domain: Optional[np.ndarray] = None,
):
    """Performs gradient descent until a loss lower than [target] is found,
    or [max_epochs] iterations have passed.
    If [target] is negative, performs [max_epochs] iterations no matter what"""
    theta = theta0.copy()
    best_theta, best_loss = theta.copy(), None
    with optim.optimize(f) as optf:
        epoch = 0
        loss0 = optf.loss(theta)
        best_loss = loss0
        while best_loss > target:
            loss = optf.loss(theta)
            if best_loss is None or loss < best_loss:
                best_theta, best_loss = theta.copy(), loss
            theta -= lr * optf.dtheta()
            if domain is not None:
                theta = project(domain, theta)
            epoch += 1
            if epoch >= max_epochs:
                break
            if target > 0 and loss - target > 1e5:
                raise DivergenceError
        return epoch, best_theta, best_loss, loss0
