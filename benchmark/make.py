"""Generate the benchmarking data"""

import time
from dataclasses import dataclass
from typing import Callable

import autograd.numpy as np
import pandas as pd

from forward import objectives as obj
from forward import optim


@dataclass
class Objectives:
    name: str
    dim: int
    f: Callable[[np.ndarray], float]


@dataclass
class Optimizer:
    name: str
    optim: optim.Optim
    stochastic: bool = False


objectives = [
    Objectives("beale", 2, lambda x: obj.beale(*x)),
    Objectives("banana", 2, lambda x: obj.banana(*x)),
    Objectives("rosenbrock10", 10, obj.rosenbrock),
    Objectives("rosenbrock100", 100, obj.rosenbrock),
    Objectives("rosenbrock1000", 1000, obj.rosenbrock),
    Objectives("sphere10", 10, obj.sphere),
    Objectives("sphere100", 100, obj.sphere),
    Objectives("sphere1000", 1000, obj.sphere),
    Objectives("hyperellipsoid10", 10, obj.hyperellipsoid),
    Objectives("hyperellipsoid100", 100, obj.hyperellipsoid),
    Objectives("hyperellipsoid1000", 1000, obj.hyperellipsoid),
    Objectives("griewank10", 10, obj.griewank),
    Objectives("griewank100", 100, obj.griewank),
    Objectives("griewank1000", 1000, obj.griewank),
]


optimizers = [
    Optimizer("Sgd", optim.SGD(optim.OptimConfig())),
    Optimizer("SgdClip", optim.SGD(optim.OptimConfig(clip=True))),
    Optimizer(
        "SgdForward", optim.SGD(optim.OptimConfig(forward=True)), stochastic=True
    ),
    Optimizer(
        "SgdClipForward",
        optim.SGD(optim.OptimConfig(forward=True, clip=True)),
        stochastic=True,
    ),
    Optimizer("Adam", optim.Adam(optim.OptimConfig())),
    Optimizer("AdaBelief", optim.AdaBelief(optim.OptimConfig())),
    Optimizer(
        "AdamForward", optim.Adam(optim.OptimConfig(forward=True)), stochastic=True
    ),
    Optimizer(
        "AdaBeliefForward",
        optim.AdaBelief(optim.OptimConfig(forward=True)),
        stochastic=True,
    ),
]


TARGET = 1e-1
N_INITIALISATIONS = 5
MAX_EPOCHS = 50_000
results_ = []

for objective in objectives:
    for _ in range(N_INITIALISATIONS):
        print(objective.name)
        theta0 = np.random.rand(objective.dim)
        theta0 += theta0 / np.linalg.norm(theta0)
        for optimizer in optimizers:

            for _ in range(5 if optimizer.stochastic else 1):
                try:
                    t1 = time.process_time()
                    epochs, theta, loss = optim.descent_fixed_cost(
                        objective.f,
                        theta0,
                        TARGET,
                        optimizer.optim,
                        lr=0.01,
                        max_epochs=MAX_EPOCHS,
                    )
                    t2 = time.process_time()
                    cpu_time = t2 - t1
                    results_.append(
                        {
                            "objective": objective.name,
                            "optimizer": optimizer.name,
                            "theta0": theta0,
                            "theta": theta,
                            "epochs": epochs,
                            "cpu_time": cpu_time,
                            "loss": loss,
                        }
                    )
                    if epochs < MAX_EPOCHS:
                        print("\t", optimizer.name, "❌⏳", f"[final loss = {loss:.2f}]")
                    else:
                        print("\t", optimizer.name, "✔️")

                except optim.DivergenceError:
                    results_.append(
                        {
                            "objective": objective.name,
                            "optimizer": optimizer.name,
                            "theta0": theta0,
                            "theta": None,
                            "epochs": None,
                            "cpu_time": None,
                            "loss": None,
                        }
                    )
                    print("\t", optimizer.name, "❌")

results = pd.DataFrame(
    results_,
    columns=["objective", "optimizer", "theta0", "theta", "epochs", "cpu_time", "loss"],
)
results.to_csv("experiments.csv")
