"""Power and inverse-power iterations for a single eigenpair."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from numerica._common import (
    column_vector,
    ensure_converged,
    require_finite_scalar,
    require_positive,
    require_positive_int,
    square_matrix,
)
from numerica.errors import NumericaError
from numerica.linear import lu_factor
from numerica.results import EigenResult


def _orient(vector: np.ndarray) -> np.ndarray:
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        return -vector
    return vector


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return _orient(vector / norm)


def _prepare_vector(matrix: np.ndarray, x0: ArrayLike | None) -> np.ndarray:
    if x0 is None:
        guess = np.ones(matrix.shape[0])
    else:
        guess = column_vector(x0, name="x0", length=matrix.shape[0])
    return _normalize(guess, name="x0")


def _result(
    *,
    method: str,
    eigenvalue: float,
    eigenvector: np.ndarray,
    iterations: int,
    converged: bool,
    residual: float,
    tolerance: float,
    history: list[float],
    raise_on_failure: bool,
) -> EigenResult:
    ensure_converged(
        method=method,
        converged=converged,
        iterations=iterations,
        residual=residual,
        tolerance=tolerance,
        value=eigenvalue,
        raise_on_failure=raise_on_failure,
    )
    return EigenResult(
        eigenvalue=eigenvalue,
        eigenvector=eigenvector,
        iterations=iterations,
        converged=converged,
        residual=residual,
        tolerance=tolerance,
        method=method,
        history=tuple(history),
    )


def power_method(
    a: ArrayLike,
    *,
    x0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 500,
    raise_on_failure: bool = True,
) -> EigenResult:
    """Dominant eigenpair of ``A`` by the power method.

    The matrix needs a unique eigenvalue of strictly largest magnitude. The
    residual is ``||A v - λ v||_2`` with ``||v||_2 = 1``, and ``λ`` is the
    Rayleigh quotient ``vᵀ A v``. The eigenvector is oriented so that its
    largest entry is positive.

    Parameters
    ----------
    a :
        Square real matrix.
    x0 :
        Nonzero initial vector. Defaults to the vector of ones.
    tol :
        Positive residual tolerance.
    max_iter :
        Maximum number of matrix-vector products used as iterates.
    raise_on_failure :
        Raise :class:`~numerica.ConvergenceError` when ``tol`` is missed.

    Raises
    ------
    ConvergenceError
        If the iteration does not converge. A complex-conjugate dominant pair
        will not converge in real arithmetic.
    """
    matrix = square_matrix(a)
    tolerance = require_positive(tol, name="tol")
    limit = require_positive_int(max_iter, name="max_iter")
    vector = _prepare_vector(matrix, x0)
    history: list[float] = []
    eigenvalue = float("nan")
    residual = float("inf")
    converged = False
    iterations = 0

    for step in range(1, limit + 1):
        iterations = step
        image = matrix @ vector
        eigenvalue = float(vector @ image)
        residual = float(np.linalg.norm(image - eigenvalue * vector))
        history.append(eigenvalue)
        if residual <= tolerance:
            converged = True
            break
        norm = float(np.linalg.norm(image))
        if norm == 0.0:
            eigenvalue = 0.0
            residual = 0.0
            converged = True
            break
        vector = _orient(image / norm)

    return _result(
        method="power",
        eigenvalue=eigenvalue,
        eigenvector=vector.copy(),
        iterations=iterations,
        converged=converged,
        residual=residual,
        tolerance=tolerance,
        history=history,
        raise_on_failure=raise_on_failure,
    )


def inverse_power_method(
    a: ArrayLike,
    *,
    shift: float = 0.0,
    x0: ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 100,
    raise_on_failure: bool = True,
) -> EigenResult:
    """Eigenpair of ``A`` closest to ``shift``, by inverse iteration.

    Each step solves ``(A - shift I) y = v`` with the LU factorization from
    :func:`numerica.lu_factor` and normalizes ``y``. ``shift`` must not be
    exactly an eigenvalue, because then ``A - shift I`` is singular.

    The iteration never leaves an invariant subspace. An initial vector that
    is already an eigenvector is returned unchanged, even if another
    eigenvalue lies closer to ``shift``.

    Parameters
    ----------
    a :
        Square real matrix.
    shift :
        Target value. The eigenvalue nearest this number is amplified.
    x0, tol, max_iter, raise_on_failure :
        Same meaning as in :func:`power_method`.
    """
    matrix = square_matrix(a)
    target = require_finite_scalar(shift, name="shift")
    tolerance = require_positive(tol, name="tol")
    limit = require_positive_int(max_iter, name="max_iter")
    factors = lu_factor(matrix - target * np.eye(matrix.shape[0]))
    vector = _prepare_vector(matrix, x0)
    history: list[float] = []
    eigenvalue = float("nan")
    residual = float("inf")
    converged = False
    iterations = 0

    for step in range(1, limit + 1):
        iterations = step
        image = factors.solve(vector)
        norm = float(np.linalg.norm(image))
        if norm == 0.0:
            raise NumericaError("Inverse power iteration produced a zero vector")
        vector = _orient(image / norm)
        product = matrix @ vector
        eigenvalue = float(vector @ product)
        residual = float(np.linalg.norm(product - eigenvalue * vector))
        history.append(eigenvalue)
        if residual <= tolerance:
            converged = True
            break

    return _result(
        method="inverse_power",
        eigenvalue=eigenvalue,
        eigenvector=vector.copy(),
        iterations=iterations,
        converged=converged,
        residual=residual,
        tolerance=tolerance,
        history=history,
        raise_on_failure=raise_on_failure,
    )
