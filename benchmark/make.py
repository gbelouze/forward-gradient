"""Generate the benchmarking data"""

import time
from dataclasses import dataclass
from pathlib import Path

import autograd.numpy as np  # type: ignore
import click
import pandas as pd  # type: ignore
from benchfunctions import get_functions

from forward import optim

data_path = Path(__file__).parent.joinpath("data")


@dataclass
class Optimizer:
    name: str
    optim: optim.Optim
    stochastic: bool = False


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


def make(dim, n_initialisations, epsilon, max_epochs, kind):
    objectives = [f(dim) for f in get_functions(dim, multimodal=False)]
    results_ = []

    expID = 0
    for objective in objectives:
        for _ in range(n_initialisations):
            print(objective.name)
            theta_star, f_star = objective.get_global_minimum()
            theta0 = sample_in_domain(objective.input_domain)
            expID += 1
            for optimizer in optimizers:
                for _ in range(5 if optimizer.stochastic else 1):
                    try:
                        t1 = time.process_time()
                        epochs, _theta, loss, loss0 = optim.descent(
                            f=objective.f,
                            theta0=theta0,
                            target=f_star + epsilon,
                            optim=optimizer.optim,
                            lr=0.01,
                            max_epochs=max_epochs,
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
                                "loss0": loss0,
                            }
                        )
                        if epochs >= max_epochs:
                            print(
                                "\t", optimizer.name, "❌⏳", f"[final loss = {loss:.2f}]"
                            )
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
                                "loss0": None,
                            }
                        )
                        print("\t", optimizer.name, "❌")

    results = pd.DataFrame(
        results_,
        columns=[
            "objective",
            "optimizer",
            "expID",
            "epochs",
            "cpu_time",
            "loss",
            "loss0",
        ],
    )
    results.loc[results.epochs == max_epochs, "epochs"] = -1

    if not data_path.exists():
        data_path.mkdir()

    id = 0
    while data_path.joinpath(f"experiments-{kind}-dim{dim}-{id}.csv").exists():
        id += 1
    results.to_csv(data_path.joinpath(f"experiments-{kind}-dim{dim}-{id}.csv"))


@click.group()
def main():
    pass


@main.command()
@click.option("--dim", default=2, help="Dimension of the problems.")
@click.option(
    "--n-initialisations",
    default=5,
    help="Number of tested initialisations for each problem.",
)
@click.option("--epsilon", default=0.1, help="Target approximation of the optimal.")
@click.option(
    "--max-epochs",
    default=50_000,
    help="Maximum number of iterations, after which the optimizer is considered to"
    "have not converged.",
)
def performance(dim, n_initialisations, epsilon, max_epochs):
    make(dim, n_initialisations, epsilon, max_epochs, kind="performance")


@main.command()
@click.option("--dim", default=2, help="Dimension of the problems.")
@click.option(
    "--n-initialisations",
    default=5,
    help="Number of tested initialisations for each problem.",
)
@click.option(
    "--n-epochs", default=10_000, help="Number of epochs to run each test for."
)
def accuracy(dim, n_initialisations, max_epochs):
    make(dim, n_initialisations, max_epochs, epsilon=-1, kind="accuracy")


if __name__ == "__main__":
    main()
