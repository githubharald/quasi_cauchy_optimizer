import numpy as np


def line_search(func, g0, x0, s):
    "apply line-search in direction s, starting from x0"
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


class Result:
    "class holding result of the optimize function"

    def __init__(self, x, path):
        self.x = x
        self.path = path


class UpdateRule:
    "update types for the estimate of the Hessian"
    DIAGONAL = 1
    SCALED_IDENTITY = 2
    IDENTITY = 3

    @classmethod
    def to_string(cls, update_rule):
        d = {UpdateRule.DIAGONAL: 'DIAGONAL', UpdateRule.SCALED_IDENTITY: 'SCALED_IDENTITY',
             UpdateRule.IDENTITY: 'IDENTITY'}
        return d[update_rule]


def optimize(func, grad, x0, update_rule=UpdateRule.DIAGONAL, grad_zero_tol=1e-5, eps=np.finfo(np.float).eps,
             min_curv=0.01, max_curv=1000, max_iter=1000, verbose=False):
    """
    iteratively minimize a function starting with an initial guess

    :param func: function to be minimized
    :param grad: gradient of function to be minimized
    :param x0: start value (initial guess)
    :param update_rule: UpdateRule.DIAGONAL, UpdateRule.SCALED_IDENTITY, or UpdateRule.IDENTITY
    :param grad_zero_tol: if gradient norm is below this value, the algorithm terminates
    :param eps: small value that it added to denominator to avoid division by 0
    :param min_curv: Hessian values are clipped to [min_curv, max_curv]
    :param max_curv: Hessian values are clipped to [min_curv, max_curv]
    :param max_iter: maximum number of iterations
    :param verbose: output internal state of algorithm
    :return: object of type Result with attributes x (solution) and path (path towards solution)
    """
    x = np.asarray(x0)
    path = [x]

    # apply initial step along steepest descent direction
    D = np.ones_like(x)
    g0 = grad(x)
    s = -g0
    s = line_search(func, g0, x, s)
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
        s = line_search(func, g1, x, s)
        x = x + s
        path.append(x)
        g0 = g1
        g1 = grad(x)

        if verbose:
            print(f'i={iter_ctr} | x={x} | s={s} | g1={g1} | D={D}')

    return Result(x, np.asarray(path))
