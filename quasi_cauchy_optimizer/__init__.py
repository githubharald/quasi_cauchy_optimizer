from dataclasses import dataclass
from enum import Enum
from typing import List, Callable

import numpy as np


def _line_search(func: Callable, g0: np.ndarray, x0: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Line-search in direction s, starting from x0, with the function func to minimize and g0 the gradient at x0."""
    f0 = func(x0)

    a = 1
    c = 1e-4
    rho = 0.5
    max_iter = 100

    for _ in range(max_iter):
        if func(x0 + a * s) > f0 + c * a * np.dot(g0, s):
            a *= rho
        else:
            return a * s

    return a * s


@dataclass
class Result:
    """Value x that minimizes the function, and path from initial guess to minimum."""
    x: np.ndarray
    path: List[np.ndarray]


class UpdateRule(Enum):
    """Update types that determine how Hessian approximation is computed."""
    DIAGONAL = 1
    SCALED_IDENTITY = 2
    IDENTITY = 3


def optimize(func: Callable,
             grad: Callable,
             x0: np.ndarray,
             update_rule: UpdateRule = UpdateRule.DIAGONAL,
             grad_zero_tol: float = 1e-5,
             eps: float = np.finfo(np.float).eps,
             min_curv: float = 0.01,
             max_curv: float = 1000,
             max_iter: int = 1000,
             verbose: bool = False) -> Result:
    """
    Minimize function value starting with an initial guess.

    Args:
        func: Function to be minimized.
        grad: Gradient of function to be minimized.
        x0: Start value (initial guess).
        update_rule: UpdateRule.DIAGONAL, UpdateRule.SCALED_IDENTITY, or UpdateRule.IDENTITY.
        grad_zero_tol: If gradient norm is below this value, the algorithm terminates.
        eps: Small value that it added to denominator to avoid division by 0.
        min_curv: Hessian values are clipped to [min_curv, max_curv].
        max_curv: Hessian values are clipped to [min_curv, max_curv].
        max_iter: Maximum number of iterations.
        verbose: Print internal state of algorithm.

    Returns:
        An object of type Result containing the value x that minimizes the function and path towards the minimum.
    """
    x = np.asarray(x0)
    path = [x]

    # apply initial step along steepest descent direction
    D = np.ones_like(x)
    g0 = grad(x)
    s = -g0
    s = _line_search(func, g0, x, s)
    x = x + s
    path.append(x)
    g1 = grad(x)

    # iterate
    for iter_ctr in range(max_iter):
        if np.linalg.norm(g1) < grad_zero_tol * max(np.linalg.norm(x), 1):
            break

        if update_rule == UpdateRule.DIAGONAL:
            y = g1 - g0
            b = s.T @ y
            a = (s ** 2).T @ D
            U = ((b - a) / (np.sum(s ** 4) + eps)) * (s ** 2)
            D = D + U

        elif update_rule == UpdateRule.SCALED_IDENTITY:
            y = g1 - g0
            D = (s.T @ y) / (s.T @ s + eps)

        elif update_rule == UpdateRule.IDENTITY:
            D = 1

        # a very crude way to ensure that diag(D) is pos. definite and s is a descent direction
        D = np.clip(D, min_curv, max_curv)

        s = -g1 / D
        s = _line_search(func, g1, x, s)
        x = x + s
        path.append(x)
        g0 = g1
        g1 = grad(x)

        if verbose:
            print(f'i={iter_ctr} | x={x} | s={s} | g1={g1} | D={D}')

    return Result(x, path)
