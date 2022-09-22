"""Generate the benchmarking data"""

import time
from dataclasses import dataclass
from pathlib import Path

import autograd.numpy as np  # type: ignore
import click
import pandas as pd  # type: ignore
from benchfunctions import get_functions
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from forward import optim

data_path = Path(__file__).parent.parent.joinpath("data")


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
    assert kind in ["accuracy", "performance"]
    objectives = [f(dim) for f in get_functions(dim, multimodal=False)]
    results_ = []

    expID = 0

    # overall progress bar
    overall_progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        refresh_per_second=2,
    )

    # single benchfunction progress
    benchfunction_progress = Progress(
        TextColumn("{task.description}"),
        SpinnerColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        refresh_per_second=2,
    )

    # group of progress bars;
    # some are always visible, others will disappear when progress is complete
    progress_group = Group(overall_progress, benchfunction_progress)

    n_test_per_objective = n_initialisations * sum(
        [5 if optimizer.stochastic else 1 for optimizer in optimizers]
    )

    with Live(progress_group):
        overall_task_id = overall_progress.add_task(
            "[green]Generating data...", total=len(objectives) * n_test_per_objective
        )
        for objective in objectives:
            benchfunction_task_id = benchfunction_progress.add_task(
                f"[bold green]Testing {objective.name}", total=n_test_per_objective
            )
            domain = None
            if objective.strict_domain:
                domain = objective.input_domain
            for _ in range(n_initialisations):
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
                                domain=domain,
                            )
                            if loss - f_star < -1e-10:
                                print(loss, f_star)
                                print(theta_star, _theta)
                                assert False
                            t2 = time.process_time()
                            cpu_time = t2 - t1
                            results_.append(
                                {
                                    "objective": objective.name,
                                    "optimizer": optimizer.name,
                                    "expID": expID,
                                    "epochs": epochs,
                                    "cpu_time": cpu_time,
                                    "loss": loss - f_star,
                                    "loss0": loss0 - f_star,
                                }
                            )

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
                        finally:
                            benchfunction_progress.update(
                                benchfunction_task_id, advance=1
                            )
                            overall_progress.update(overall_task_id, advance=1)
            benchfunction_progress.update(benchfunction_task_id, visible=False)
            benchfunction_progress.stop_task(benchfunction_task_id)

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
        ]
        if kind == "performance"
        else [
            "objective",
            "optimizer",
            "expID",
            "cpu_time",
            "loss",
            "loss0",
        ],
    )
    if kind == "performance":
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
def accuracy(dim, n_initialisations, n_epochs):
    make(dim, n_initialisations, max_epochs=n_epochs, epsilon=-1, kind="accuracy")


if __name__ == "__main__":
    main()
