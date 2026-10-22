"""Classical numerical methods implemented with NumPy.

``numerica`` is a small, typed library for root finding, interpolation,
quadrature, linear systems, and ordinary differential equations. The
algorithms are written out explicitly so they can be read and taught; SciPy
is not used.

Import functions from this package::

    from numerica import bisection, rk4, cubic_spline

or from the submodule that owns them::

    from numerica.roots import newton
"""

from numerica.differentiate import derivative
from numerica.eigen import inverse_power_method, power_method
from numerica.errors import ConvergenceError, DimensionError, NumericaError, SingularMatrixError
from numerica.fit import least_squares, polynomial_fit
from numerica.integrate import (
    gauss_legendre,
    legendre_nodes,
    romberg,
    simpson,
    simpson_38,
    trapezoidal,
)
from numerica.interpolate import (
    CubicSpline,
    LagrangeInterpolant,
    NewtonInterpolant,
    cubic_spline,
    divided_difference_table,
    lagrange,
    newton_divided_difference,
)
from numerica.linear import (
    LUFactorization,
    gauss_seidel,
    gaussian_elimination,
    jacobi,
    lu_factor,
    lu_solve,
)
from numerica.ode import euler, heun, rk4, rk45
from numerica.results import (
    EigenResult,
    FitResult,
    IntegralResult,
    LinearResult,
    ODEResult,
    RootResult,
)
from numerica.roots import bisection, fixed_point, newton, newton_raphson, secant

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "NumericaError",
    "ConvergenceError",
    "SingularMatrixError",
    "DimensionError",
    "RootResult",
    "IntegralResult",
    "LinearResult",
    "ODEResult",
    "FitResult",
    "EigenResult",
    "LUFactorization",
    "bisection",
    "newton",
    "newton_raphson",
    "secant",
    "fixed_point",
    "lagrange",
    "newton_divided_difference",
    "divided_difference_table",
    "cubic_spline",
    "LagrangeInterpolant",
    "NewtonInterpolant",
    "CubicSpline",
    "derivative",
    "trapezoidal",
    "simpson",
    "simpson_38",
    "gauss_legendre",
    "legendre_nodes",
    "romberg",
    "gaussian_elimination",
    "lu_factor",
    "lu_solve",
    "jacobi",
    "gauss_seidel",
    "euler",
    "heun",
    "rk4",
    "rk45",
    "polynomial_fit",
    "least_squares",
    "power_method",
    "inverse_power_method",
]
