# Quasi Cauchy Optimizer

Implementation of the quasi cauchy optimizer as described in the paper of [Zhu et al](http://www.math.uwaterloo.ca/~hwolkowi/henry/reports/cauchy.pdf).
It approximates the Hessian by a diagonal matrix which satisfies the weak secant equation.
The method is very memory-efficient, because a diagonal matrix has the same memory-footprint as a vector.


## Get started

### 1. Run example scripts

* Install requirements: `pip install -r requirements.txt`
* Run optimizer on simple test-functions
    * Fast version tests only two functions: `python example_test_funcs.py fast`
    * Test all functions: `python example_test_funcs.py`
* Run optimizer on logistic regression problem: `python example_log_reg.py`

Expected output for `python example_test_funcs.py fast`:
````
Function: beale
beale, DIAGONAL, err=0.000, iter=172
beale, SCALED_IDENTITY, err=0.000, iter=33
beale, IDENTITY, err=0.001, iter=367

Function: polyNd
polyNd, DIAGONAL, err=0.371, iter=235
polyNd, SCALED_IDENTITY, err=0.605, iter=501
polyNd, IDENTITY, err=0.962, iter=501
````

![plot](doc/plot.png)

### 2. Integrate into your code

To use the optimizer in your own code, define the function to be minimized and the gradient of this function. 
Then, call the `optimize(...)` function specifying an initial guess for the solution.
The result structure holds both the final iterate (attribute `x`) and the complete path from the initial iterate to the final iterate (attribute `path`).
You can use libraries like autograd to compute the gradient for your function.
Here is a small example to compute the minimum of a quadratic function:

````python
from quasi_cauchy_optimizer import optimize, UpdateRule
import numpy as np

# function to minimize: 5 * x**2 + y**2
def func(x):
    return 5 * x[0]**2 + x[1]**2

# gradient of function: (10x, 2y)
def grad(x):
    return np.asarray([10, 2]) * x

# define start value
x0 = np.asarray([1, 2])

# run optimizer
res = optimize(func, grad, x0, UpdateRule.DIAGONAL, grad_zero_tol=1e-5)

# print result
print(res.x)
````

Function arguments: 
* func: function to be minimized
* grad: gradient of function to be minimized
* x0: start value (initial guess)
* update_rule
    * UpdateRule.DIAGONAL: Hessian approximated as diagonal matrix
    * UpdateRule.SCALED_IDENTITY: Hessian approximated as scaled identity matrix
    * UpdateRule.IDENTITY: Hessian approximated as identity matrix (that is vanilla gradient descent, included only to have a baseline for evaluation)
* grad_zero_tol: if gradient norm is below this value, the algorithm terminates
* eps: small value that it added to denominator to avoid division by 0
* min_curv: Hessian values are clipped to [min_curv, max_curv]
* max_curv: Hessian values are clipped to [min_curv, max_curv]
* max_iter: maximum number of iterations
* verbose: output internal state of algorithm


## Some notes
To ensure having a descent direction, the Hessian simply is clipped, where the minimum value (min_curv) should be set to some small value larger than 0.
A line-search is applied along the computed update-direction to get a reasonable step-size.

The diagonal approximation (UpdateRule.DIAGONAL) performs best for high-dimensional functions with scale varying across dimensions. 
Otherwise, the simple scaled identity approximation (UpdateRule.SCALED_IDENTITY) performs best. 
This also includes the typical 2D test-functions like Rosenbrock.

For results and details on how the Hessian approximation is computed see [this article](https://githubharald.github.io/quasi_cauchy.html).