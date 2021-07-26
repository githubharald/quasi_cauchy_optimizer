# Quasi Cauchy Optimizer

Implementation of the quasi Cauchy optimizer as described by [Zhu et al](http://www.math.uwaterloo.ca/~hwolkowi/henry/reports/cauchy.pdf).
It is a member of the quasi Newton family.
The Hessian is approximated by a diagonal matrix which satisfies the weak secant equation.
The method is memory-efficient, because a diagonal matrix has the same memory-footprint as a vector.

## Installation

* Go to the root level of the repository
* Execute `pip install .`
* Go to `tests/` and execute `pytest` to check if installation worked

## Usage

To use the optimizer in your own code, define the function to be minimized, and the gradient of this function 
(or compute it using e.g. the autograd package).
Then, call the `optimize(...)` function with an initial guess of the solution.
The result holds both the final iterate (attribute `x`) and the path from the initial to the final iterate (attribute `path`).
Here is a small example that computes the minimum of a quadratic function:

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
    * UpdateRule.DIAGONAL: Hessian is approximated as diagonal matrix
    * UpdateRule.SCALED_IDENTITY: Hessian is approximated as scaled identity matrix
    * UpdateRule.IDENTITY: Hessian is approximated as identity matrix (that is vanilla gradient descent, included only to have a baseline for evaluation)
* grad_zero_tol: if gradient norm is below this value, the algorithm terminates
* eps: small value that it added to denominator to avoid division by 0
* min_curv: Hessian values are clipped to [min_curv, max_curv]
* max_curv: Hessian values are clipped to [min_curv, max_curv]
* max_iter: maximum number of iterations
* verbose: output internal state of algorithm


## Examples

* Install requirements: `pip install -r requirements.txt`
* Go to `examples/`  
* Run optimizer on common test functions:
    * Fast version testing only two functions: `python common_test_functions.py fast`
    * Test all functions: `python common_test_functions.py`
* Run optimizer on logistic regression task: `python logistic_regression.py`

Expected output for `python common_test_functions.py fast`:
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


## Some notes
To ensure having a descent direction, the Hessian simply is clipped, where the minimum value (min_curv) should be set to some small value larger than 0.
A line-search is applied along the computed update-direction to get a reasonable step-size.

The diagonal approximation (UpdateRule.DIAGONAL) performs best for high-dimensional functions with scale varying across dimensions. 
Otherwise, the simple scaled identity approximation (UpdateRule.SCALED_IDENTITY) performs best. 
This also includes the typical 2D test-functions like Rosenbrock.

For results and details on how the Hessian approximation is computed see [this article](https://githubharald.github.io/quasi_cauchy.html).