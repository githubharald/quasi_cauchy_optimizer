import numpy as np

from quasi_cauchy_optimizer import optimize, UpdateRule


def get_data(dim):
    "create data-points with specified dimension for two classes"
    num_a = 100
    num_b = 100

    m_a = np.ones(dim) * 1  # mean
    s_a = np.diag(np.random.random([dim]) * 10 + 0.1)  # cov
    x_a = np.random.multivariate_normal(m_a, s_a, num_a)

    m_b = np.ones(dim) * 2  # mean
    s_b = np.diag(np.random.random([dim]) * 10 + 0.1)  # cov
    x_b = np.random.multivariate_normal(m_b, s_b, num_b)

    X = np.hstack([np.vstack([x_a, x_b]), np.ones([num_a + num_b, 1])])

    y_a = np.asarray([0] * num_a)
    y_b = np.asarray([1] * num_b)
    Y = np.append(y_a, y_b)

    return X, Y


def calc_prob(x, beta):
    "probability of x belonging to class 1, given beta. x contains the bias term."
    e = np.exp(np.dot(beta, x))
    return e / (1 + e)


def classify(x, beta):
    "get class of x. x contains the bias term."
    return 1 if calc_prob(x, beta) > 0.5 else 0


def conservative_log_exp_add1(v):
    "evaluate log(exp(v)+1) without overflow, by ignoring the +1 term for large exp(v) values"
    # this is a number with 16 digits (float64 can hold ~16 decimal digits)
    if v > 37:
        return v
    else:
        return np.log(1 + np.exp(v))


def train_and_evaluate(dim):
    "train and evaluate logistic regression for the specified number of dimensions (without counting bias term)"
    X, Y = get_data(dim)

    def neg_log_likelihood(beta):
        "negative log-likelihood"
        terms = [y_i * np.dot(x_i, beta) - conservative_log_exp_add1(np.dot(x_i, beta))
                 for x_i, y_i in zip(X, Y)]
        return -np.sum(terms)

    def grad_neg_log_likelihood(beta):
        "gradient of negative log-likelihood (w.r.t. parameter vector beta)"
        terms = [x_i * (y_i - calc_prob(x_i, beta)) for x_i, y_i in zip(X, Y)]
        return -np.sum(terms, axis=0)

    # try both the weak secant update and the scaled identity update
    beta0 = np.zeros([dim + 1])  # +1 for bias
    for update_rule in [UpdateRule.DIAGONAL, UpdateRule.SCALED_IDENTITY]:
        # compute optimal parameters
        res = optimize(neg_log_likelihood, grad_neg_log_likelihood,
                       beta0, update_rule=update_rule, grad_zero_tol=0.05)
        beta = res.x

        # output result
        update_rule_name = UpdateRule.to_string(update_rule)
        print(update_rule_name)
        acc = sum([classify(x, beta) == y for x, y in zip(X, Y)]) / len(X)
        print(f'iter={len(res.path)}, acc={acc}')


def main():
    # train and evaluate logistic regression model for different dimensions
    for dim in [1, 10, 20, 30, 40]:
        print(f'dim={dim}')
        train_and_evaluate(dim)
        print()  # put newline after results


if __name__ == '__main__':
    main()
