def beale(x, y):
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2
