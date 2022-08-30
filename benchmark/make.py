"""Generate the benchmarking data"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import autograd.numpy as np
import pandas as pd
from benchfunctions import get_functions

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


def sample_in_domain(domain):
    d = domain.shape[0]
    theta = np.random.rand(d)
    return domain[:, 0] + theta * (domain[:, 1] - domain[:, 0])


DIM = 2
objectives = [f(DIM) for f in get_functions(DIM, multimodal=False)]

TARGET = 1e-1
N_INITIALISATIONS = 5
MAX_EPOCHS = 50_000
results_ = []

expID = 0
for objective in objectives:
    for _ in range(N_INITIALISATIONS):
        print(objective.name)
        theta_star, f_star = objective.get_global_minimum()
        theta0 = sample_in_domain(objective.input_domain)
        expID += 1
        for optimizer in optimizers:
            for _ in range(5 if optimizer.stochastic else 1):
                try:
                    t1 = time.process_time()
                    epochs, _theta, loss = optim.descent_fixed_cost(
                        f=objective.f,
                        theta0=theta0,
                        target=f_star + TARGET,
                        optim=optimizer.optim,
                        lr=0.01,
                        max_epochs=MAX_EPOCHS,
                    )
                    t2 = time.process_time()
                    cpu_time = t2 - t1
                    results_.append(
                        {
                            "objective": objective.name,
                            "optimizer": optimizer.name,
                            "expID": expID,
                            "epochs": epochs,
                            "cpu_time": cpu_time,
                            "loss": loss,
                        }
                    )
                    if epochs >= MAX_EPOCHS:
                        print("\t", optimizer.name, "❌⏳", f"[final loss = {loss:.2f}]")
                    else:
                        print("\t", optimizer.name, "✔️")

                except optim.DivergenceError:
                    results_.append(
                        {
                            "objective": objective.name,
                            "optimizer": optimizer.name,
                            "expID": expID,
                            "epochs": None,
                            "cpu_time": None,
                            "loss": None,
                        }
                    )
                    print("\t", optimizer.name, "❌")

results = pd.DataFrame(
    results_,
    columns=["objective", "optimizer", "expID", "epochs", "cpu_time", "loss"],
)

id = 0
while Path(f"data/experiments-dim{DIM}-{id}.csv").exists():
    id += 1
results.to_csv(f"data/experiments-dim{DIM}-{id}.csv")
