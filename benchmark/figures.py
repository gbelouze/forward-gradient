import re
from pathlib import Path

import click
import numpy as np
import pandas as pd  # type: ignore

image_path = Path(__file__).parent.parent.joinpath("images")


def get_experiment_files(kind: str) -> dict[int, list[Path]]:
    datapath = Path(__file__).parent.parent.joinpath("data")
    experiments_re = rf"experiments-{kind}-dim(\d+)-(\d+)\.csv"
    ret: dict[int, list[Path]] = {}
    for path in datapath.iterdir():
        if match := re.match(experiments_re, path.name):
            dim = int(match.group(1))
            if dim not in ret:
                ret[dim] = []
            ret[dim].append(path)
    return dict(sorted(ret.items()))


def read_experiment_files(dimToPaths: dict[int, list[Path]]) -> dict[int, pd.DataFrame]:
    return {
        dim: pd.concat(
            [pd.read_csv(path, index_col=0) for path in paths], ignore_index=True
        )
        for dim, paths in dimToPaths.items()
    }


def accuracy_profile(df: pd.DataFrame):
    M = 10
    df.loss += 10 ** (-M)
    if any(df.loss < 0):
        print(
            "Negative loss found for test functions",
            set(df.objective[df.loss < 0].values),
        )
        raise ValueError("Negative loss found in experiments")
    # df.loc[df.loss == 0, "loss"] = np.exp(-2 * M)
    df["accuracy"] = -np.log(df.loss) + np.log(df.loss0)
    df.loc[df.accuracy > M, "accuracy"] = M

    experiment_index = ["expID", "optimizer"]
    experiments = pd.DataFrame([], index=df.groupby(experiment_index).groups.keys())
    experiments.index.set_names(experiment_index, inplace=True)
    experiments["MeanAccuracy"] = (
        experiments.join(df.set_index(experiment_index))
        .groupby(experiment_index)
        .accuracy.mean()
    )

    n_experiments = len(df.groupby(["expID"]).groups)
    max_tau = M

    def ap(optimizer):
        """Accuracy profile"""
        values = sorted(
            experiments.xs(optimizer, level="optimizer").MeanAccuracy.dropna().values,
            reverse=True,
        )
        taus = sorted(list(set(values) | set([0, max_tau])))
        total = np.zeros(len(taus))
        for i, tau in enumerate(taus):
            for value in values:
                if value < tau:
                    break
                total[i] += 1
        total = total / n_experiments
        return taus, total

    return {optimizer: ap(optimizer) for optimizer in set(df.optimizer.to_numpy())}


def performance_profile(df: pd.DataFrame):
    df.loc[df.epochs == -1, "epochs"] = np.nan
    experiment_index = ["expID", "optimizer"]
    experiments = pd.DataFrame([], index=df.groupby(experiment_index).groups.keys())
    experiments.index.set_names(experiment_index, inplace=True)
    experiments["MeanEpoch"] = (
        experiments.join(df.set_index(experiment_index))
        .groupby(experiment_index)
        .epochs.mean()
    )
    experiments["Rps"] = experiments.groupby(["expID"]).MeanEpoch.transform(
        lambda x: x / x.min()
    )

    n_experiments = len(experiments.groupby(["expID"]).groups)
    max_tau = 2e3

    def pp(optimizer):
        """Performance profile"""
        values = sorted(
            experiments.xs(optimizer, level="optimizer").Rps.dropna().values
        )
        taus = sorted(list(set(values) | set([1, max_tau])))
        total = np.zeros(len(taus))
        for i, tau in enumerate(taus):
            for value in values:
                if value > tau:
                    break
                total[i] += 1
        total = total / n_experiments
        return taus, total

    return {
        optimizer: pp(optimizer)
        for optimizer in experiments.index.get_level_values("optimizer").unique()
    }


@click.command()
@click.argument(
    "task", type=click.Choice(["performance", "accuracy"], case_sensitive=False)
)
def main(task):
    import matplotlib.pyplot as plt  # type: ignore

    reverse_cmap = plt.get_cmap("Purples")
    forward_cmap = plt.get_cmap("Greens")

    def opt_to_col(name):
        cmap = forward_cmap if "Forward" in name else reverse_cmap
        d = {"SgdClip": 120, "Sgd": 60, "Adam": 180, "AdaBelief": 240}
        for key, coef in d.items():
            if key in name:
                return cmap(coef)
        raise ValueError(f"Unknown optimizer {name}")

    def sortAux(label):
        if label == "Sgd":
            label = "SgdD"
        if "Forward" in label:
            return "_" + label
        return label

    if task == "performance":
        dimToDf = read_experiment_files(get_experiment_files(kind="performance"))
        fig, axs = plt.subplots(
            1,
            len(dimToDf),
            squeeze=False,
            sharey=True,
            sharex=True,
            figsize=(len(dimToDf) * 5, 5),
        )
        for ax, (dim, df) in zip(axs[0], dimToDf.items()):
            profiles = performance_profile(df)
            for optimizer, (taus, proportion) in profiles.items():
                ax.semilogx(
                    taus, proportion, label=optimizer, color=opt_to_col(optimizer)
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(f"Dimension {dim}", fontdict={"fontsize": "medium"})

        handles, labels = axs[0, -1].get_legend_handles_labels()
        argsort = np.argsort([sortAux(label) for label in labels])
        fig.legend(
            [handles[i] for i in argsort],
            [labels[i] for i in argsort],
            loc="lower left",
            ncol=2,
            fontsize="small",
        )

        plt.tight_layout()
        if not image_path.exists():
            image_path.mkdir()
        fig.savefig(image_path.joinpath("performance-profile.png"), dpi=400)
        plt.show()

    elif task == "accuracy":
        dimToDf = read_experiment_files(get_experiment_files(kind="accuracy"))
        fig, axs = plt.subplots(
            1,
            len(dimToDf),
            squeeze=False,
            sharey=True,
            sharex=True,
            figsize=(len(dimToDf) * 5, 5),
        )
        for ax, (dim, df) in zip(axs[0], dimToDf.items()):
            profiles = accuracy_profile(df)
            for optimizer, (taus, proportion) in profiles.items():
                ax.plot(taus, proportion, label=optimizer, color=opt_to_col(optimizer))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, labels = axs[0, -1].get_legend_handles_labels()
        argsort = np.argsort([sortAux(label) for label in labels])
        fig.legend(
            [handles[i] for i in argsort],
            [labels[i] for i in argsort],
            ncol=2,
            fontsize="small",
        )

        plt.tight_layout()
        fig.savefig(image_path.joinpath("accuracy-profile.png"), dpi=400)
        plt.show()


if __name__ == "__main__":
    main()
