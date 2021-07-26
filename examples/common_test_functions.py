import sys

import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt

from quasi_cauchy_optimizer import optimize, UpdateRule


def plot_function(f):
    "plot 2d function"
    num_vals = 50
    x_vals = np.linspace(-5, 5, num_vals)
    y_vals = np.linspace(-5, 5, num_vals)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.empty([num_vals, num_vals], dtype=np.float64)
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            v = np.asarray([x, y])
            Z[j, i] = f(v)

    plt.pcolormesh(X, Y, Z, cmap='rainbow', shading='gouraud')


def test_function(name):
    func = None  # function
    x0 = None  # initial guess
    xt = None  # true value of minimum
    if name == 'beale':
        def func(x):
            x0, x1 = x
            return (1.5 - x0 + x0 * x1) ** 2 + (2.25 - x0 + x0 * x1 ** 2) ** 2 + (2.625 - x0 + x0 * x1 ** 3) ** 2

        x0 = (1, 1)
        xt = (3, 0.5)

    elif name == 'rosen':
        def func(x):
            x0, x1 = x
            a = 1.0
            b = 100.0
            return (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2

        x0 = (-1, 1.75)
        xt = (1, 1)

    elif name == 'poly2d':
        def func(x):
            x0, x1 = x
            return x0 ** 4 + x1 ** 2

        x0 = (3, 4)
        xt = (0, 0)

    elif name == 'polyNd' or name == 'quadraticNd':
        n = 50

        def func(x):
            exps = [2 * ((i % 4) + 1) for i in range(n)] if name == 'polyNd' else [2 for _ in range(n)]
            coeffs = [10 ** (i % 3 - 1) for i in range(n)] if name == 'polyNd' else [10 ** (i % 5 - 2) for i in
                                                                                     range(n)]
            return np.sum([c * x[i] ** e for i, (c, e) in enumerate(zip(coeffs, exps))])

        x0 = 2 * (np.random.rand(n) - 0.5)  # [1 for _ in range(n)]
        xt = [0 for _ in range(n)]

    elif name == 'quadratic2d':
        def func(x):
            x0, x1 = x
            return 100 * x0 ** 2 + x1 ** 2 + x0 * x1

        x0 = (5, 4)
        xt = (0, 0)

    return func, np.asarray(x0, np.float), np.asarray(xt, np.float)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == 'fast':
        func_names = ['beale', 'polyNd']
    else:
        func_names = ['beale', 'rosen', 'poly2d', 'quadratic2d', 'quadraticNd', 'polyNd']

    for func_name in func_names:
        print('Function:', func_name)

        func, x0, xt = test_function(func_name)

        def grad(v):
            return autograd.grad(func)(v)

        make_2d_plot = len(x0) == 2
        if make_2d_plot:
            plt.figure(func_name)

        for i, update_rule in enumerate([UpdateRule.DIAGONAL, UpdateRule.SCALED_IDENTITY, UpdateRule.IDENTITY]):
            res = optimize(func, grad, x0, update_rule=update_rule, max_iter=500, grad_zero_tol=1e-4)

            err = np.linalg.norm(xt - res.x)
            num_iter = len(res.path)
            print(f'{func_name}, {update_rule.name}, err={err:.3f}, iter={num_iter}')

            if not make_2d_plot:
                continue

            plt.subplot(1, 3, i + 1)

            plt.title(f'{update_rule.name}, err={err:.3f}, iter={num_iter}')

            plot_function(func)

            path = np.array(res.path)
            plt.plot(path[:, 0], path[:, 1], 'r-*', label='path')

            plt.plot(path[:1, 0], path[:1, 1], 'w^', label='init iterate')
            plt.plot(path[-1:, 0], path[-1:, 1], 'k^', label='final iterate')
            plt.plot([xt[0]], [xt[1]], 'g*', label='true min')

            plt.legend()

        # put newline after results
        print()

    # show all plots
    plt.show()


if __name__ == '__main__':
    main()
