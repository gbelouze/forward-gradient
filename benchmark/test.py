import benchfunctions as bf
import click
import numpy as np

import forward as fwd


class BadTestFunction(Exception):
    pass


def test(f):
    _, min_loss = f.get_global_minimum()
    if min_loss is None:
        return True, None
    min_found = min_loss + 1
    optim = fwd.optim.Adam(fwd.optim.OptimConfig())
    for tries in range(5):
        theta = np.random.random(f.d)
        with optim.optimize(f) as optf:
            for epoch in range(10_000):
                loss = optf.loss(theta)
                if min_loss - loss > 1e-10:
                    return False, BadTestFunction(
                        f"{f.name} : found loss {loss} < {min_loss} at {theta=}."
                    )
                if loss < min_found:
                    min_found = loss
                theta -= 0.1 * optf.dtheta()
                if f.strict_domain:
                    theta = fwd.optim.project(f.input_domain, theta)
        if min_found - min_loss > 0.1:
            continue
        return True, None
    return False, BadTestFunction(
        f"{f.name} : Unable to reach {min_loss} [best found: {min_found}]"
    )


def find_best_theta(f):
    best, min_loss = f.get_global_minimum()
    optims = [
        opt(fwd.optim.OptimConfig())
        for opt in [fwd.optim.Adam, fwd.optim.SGD, fwd.optim.AdaBelief]
    ]
    thetas = 1e-8 * np.ones((5, f.d))  # some test functions are not differentiable in 0
    thetas[1:, :] = np.random.random((4, f.d))
    for theta0 in thetas:
        for optim in optims:
            theta = theta0.copy()
            with optim.optimize(f) as optf:
                for epoch in range(10_000):
                    loss = optf.loss(theta)
                    if loss < min_loss:
                        print(min_loss)
                        best, min_loss = theta.copy(), loss
                    theta -= 0.01 * optf.dtheta()
                    if f.strict_domain:
                        theta = fwd.optim.project(f.input_domain, theta)
    print(f"Minimum loss={min_loss} found at theta={best}")


def _main(name, dim, find_best):
    if name == "all":
        exns = []
        for F in bf.get_functions(dim, randomized_term=False):
            print(F.name, end=" ")
            passed, exn = test(F(dim))
            if not passed:
                exns.append(exn)
                print("❌")
            else:
                print("✔️")
        if exns:
            raise Exception(exns)
        print("All good!")
        return

    if name not in bf.available_functions:
        raise ValueError(f"Unknown function {name}")
    F = bf.available_functions[name]
    if not F.is_dim_compatible(dim):
        raise ValueError(f"Function {name} has no definition in dimension {dim}")
    f = F(dim)
    if find_best:
        find_best_theta(f)
    else:
        passed, exn = test(f)
        if not passed:
            raise exn
        print("All good!")


@click.command()
@click.argument("name")
@click.option("--dim", default=2, help="Dimension of the benchfunction.")
@click.option(
    "--find-best",
    is_flag=True,
    default=False,
    help="Find the experimental minimum of the function.",
)
def main(name, dim, find_best):
    _main(name, dim, find_best)


if __name__ == "__main__":
    main()
