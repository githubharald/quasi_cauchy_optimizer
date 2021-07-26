import numpy as np

from quasi_cauchy_optimizer import optimize, UpdateRule


# function to minimize: 5 * x**2 + y**2
def func(x):
    return 5 * x[0] ** 2 + x[1] ** 2


# gradient of function: (10x, 2y)
def grad(x):
    return np.asarray([10, 2]) * x


def test_diagonal_update():
    # start value
    x0 = np.asarray([1, 2])

    # run optimizer
    res = optimize(func, grad, x0, UpdateRule.DIAGONAL, grad_zero_tol=1e-5)

    # check distance to true minimizer
    dist = np.linalg.norm(res.x - np.array([0, 0]))
    assert dist < 1e-5


def test_scaled_identity_update():
    # start value
    x0 = np.asarray([1, 2])

    # run optimizer
    res = optimize(func, grad, x0, UpdateRule.SCALED_IDENTITY, grad_zero_tol=1e-5)

    # check distance to true minimizer
    dist = np.linalg.norm(res.x - np.array([0, 0]))
    assert dist < 1e-5


def test_identity_update():
    # start value
    x0 = np.asarray([1, 2])

    # run optimizer
    res = optimize(func, grad, x0, UpdateRule.IDENTITY, grad_zero_tol=1e-5)

    # check distance to true minimizer
    dist = np.linalg.norm(res.x - np.array([0, 0]))
    assert dist < 1e-5
