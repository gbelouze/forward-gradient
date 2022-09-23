"""Generate the benchmarking data"""

import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path

import autograd.numpy as np  # type: ignore
import click
import pandas as pd  # type: ignore
from benchfunctions import get_functions
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
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


# a global variable is necessary to share a Value between processes
overall_progress_completed = None


def value_initializer(value):
    global overall_progress_completed
    overall_progress_completed = value


def make_single_objective_(
    objective,
    expID,
    n_initialisations,
    epsilon,
    max_epochs,
    progress_dict,
    benchfunction_task_id,
):
    results = []

    domain = None
    if objective.strict_domain:
        domain = objective.input_domain

    task_completion = 0
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
                    results.append(
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
                    results.append(
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
                    task_completion += 1
                    progress_dict[benchfunction_task_id] = task_completion
                    with overall_progress_completed.get_lock():
                        overall_progress_completed.value += 1
    return results


def make_single_objective(*args, **kwargs):
    """Handle keyboard interrupt so that the child process immediately returns"""
    try:
        return make_single_objective_(*args, **kwargs)
    except KeyboardInterrupt:
        return "KeyboardInterrupt"


def stop_task(benchfunction_progress, benchfunction_task_id):
    benchfunction_progress.update(benchfunction_task_id, visible=False)
    benchfunction_progress.stop_task(benchfunction_task_id)


def make(dims, n_initialisations, epsilon, max_epochs, kind):
    global overall_progress_completed
    assert kind in ["accuracy", "performance"]

    # overall progress bar
    overall_progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        refresh_per_second=0.2,
    )

    # single benchfunction progress
    benchfunction_progress = Progress(
        TextColumn("{task.description}"),
        SpinnerColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        refresh_per_second=0.2,
    )

    # group of progress bars;
    # some are always visible, others will disappear when progress is complete
    progress_group = Group(Panel(benchfunction_progress), overall_progress)

    dim_to_objectives = {
        dim: [f(dim) for f in get_functions(dim, multimodal=False)] for dim in dims
    }
    results_ = []

    expID = 0

    n_test_per_objective = {
        dim: n_initialisations
        * sum([5 if optimizer.stochastic else 1 for optimizer in optimizers])
        for dim, objectives in dim_to_objectives.items()
    }

    with Live(progress_group):
        overall_task_id = overall_progress.add_task(
            "[green]Generating data...",
            total=sum(
                [
                    len(objectives) * n_test_per_objective[dim]
                    for dim, objectives in dim_to_objectives.items()
                ]
            ),
        )

        jobs = []
        with multiprocessing.Manager() as manager:
            # we share progress information between processes
            progress_dict = manager.dict()
            overall_progress_completed = multiprocessing.Value("i", 0)

            with multiprocessing.Pool(
                max(1, os.cpu_count() - 2),
                initializer=value_initializer,
                initargs=(overall_progress_completed,),
            ) as pool:

                for dim, objectives in dim_to_objectives.items():
                    for objective in objectives:
                        benchfunction_task_id = benchfunction_progress.add_task(
                            f"[bold green]Testing {objective.name} (dim={objective.d})",
                            total=n_test_per_objective[dim],
                            visible=False,
                        )
                        jobs.append(
                            pool.apply_async(
                                make_single_objective,
                                (
                                    objective,
                                    expID,
                                    n_initialisations,
                                    epsilon,
                                    max_epochs,
                                    progress_dict,
                                    benchfunction_task_id,
                                ),
                            )
                        )
                        expID += n_test_per_objective[dim]

                try:
                    while not all([job.ready() for job in jobs]):
                        overall_progress.update(
                            overall_task_id, completed=overall_progress_completed.value
                        )

                        to_remove = []
                        for benchfunction_task_id, completed in progress_dict.items():
                            benchfunction_progress.update(
                                benchfunction_task_id, completed=completed
                            )
                            task = benchfunction_progress.tasks[benchfunction_task_id]
                            if task.finished:
                                benchfunction_progress.update(
                                    benchfunction_task_id, visible=False
                                )
                                benchfunction_progress.stop_task(benchfunction_task_id)
                                to_remove.append(benchfunction_task_id)
                            else:
                                benchfunction_progress.update(
                                    benchfunction_task_id, visible=True
                                )
                        for benchfunction_task_id in to_remove:
                            del progress_dict[benchfunction_task_id]

                    for job in jobs:
                        results_ += job.get()
                except KeyboardInterrupt:
                    pool.terminate()
                    raise

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
@click.option("--dim", multiple=True, default=[2], help="Dimension of the problems.")
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
@click.option("--dim", multiple=True, default=[2], help="Dimension of the problems.")
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
