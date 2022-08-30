import re
from pathlib import Path

import numpy as np
import pandas as pd


def get_experiment_files() -> list[Path]:
    datapath = Path(__file__).parent.parent.joinpath("data")
    experiments_re = r"experiments-(\d+)\.csv"
    return [path for path in datapath.iterdir() if re.match(experiments_re, path.name)]


def read_experiment_files(paths: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_csv(path, index_col=0) for path in paths]
    return pd.concat(dfs, ignore_index=True)


def performance_profile(df: pd.DataFrame):
    experiment_index = ["objective", "theta0", "optimizer"]
    experiments = pd.DataFrame([], index=df.groupby(experiment_index).groups.keys())
    experiments.index.set_names(experiment_index, inplace=True)
    experiments["MeanEpoch"] = (
        experiments.join(df.set_index(experiment_index))
        .groupby(experiment_index)
        .epochs.mean()
    )
    experiments["Rps"] = experiments.groupby(
        ["objective", "theta0"]
    ).MeanEpoch.transform(lambda x: x / x.min())

    n_experiments = len(experiments.groupby(["objective", "theta0"]).groups)
    max_tau = experiments.Rps.dropna().max()

    def pp(optimizer):
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


def main():
    import matplotlib.pyplot as plt

    data = read_experiment_files(get_experiment_files())
    profiles = performance_profile(data)
    for optimizer, (taus, proportion) in profiles.items():
        plt.semilogx(taus, proportion, "+-", label=optimizer)
    plt.legend()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()


if __name__ == "__main__":
    main()
