"""Scalar root finding: bisection, Newton, secant, and fixed-point iteration."""

from __future__ import annotations

import math

from numerica._common import (
    ScalarFunction,
    ensure_converged,
    require_finite_scalar,
    require_positive,
    require_positive_int,
)
from numerica.errors import NumericaError
from numerica.results import RootResult


def _call(f: ScalarFunction, x: float, *, name: str) -> float:
    return require_finite_scalar(f(x), name=name)


def _brackets(left: float, right: float) -> bool:
    if left == 0.0 or right == 0.0:
        return True
    return (left < 0.0) != (right < 0.0)


def _finish(
    *,
    method: str,
    root: float,
    iterations: int,
    residual: float,
    tolerance: float,
    history: list[float],
    converged: bool,
    raise_on_failure: bool,
) -> RootResult:
    result = RootResult(
        root=root,
        iterations=iterations,
        converged=converged,
        residual=residual,
        tolerance=tolerance,
        method=method,
        history=tuple(history),
    )
    ensure_converged(
        method=method,
        converged=converged,
        iterations=iterations,
        residual=residual,
        tolerance=tolerance,
        value=root,
        raise_on_failure=raise_on_failure,
    )
    return result


def bisection(
    f: ScalarFunction,
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
    raise_on_failure: bool = True,
) -> RootResult:
    """Find a root of ``f`` on ``[a, b]`` by bisection.

    ``f(a)`` and ``f(b)`` must have opposite signs, or either endpoint must
    already be a root. The iteration stops when the half-width of the bracket
    or the absolute residual is at most ``tol``.

    Parameters
    ----------
    f :
        Scalar function.
    a, b :
        Endpoints with ``a < b``.
    tol :
        Positive stopping tolerance.
    max_iter :
        Maximum number of midpoint evaluations.
    raise_on_failure :
        When ``True`` (the default), failure to meet ``tol`` raises
        :class:`~numerica.ConvergenceError`. When ``False``, the best midpoint
        is returned with ``converged=False``.

    Returns
    -------
    RootResult
        The root estimate. ``residual`` is ``abs(f(root))`` and ``history``
        contains every midpoint.

    Raises
    ------
    ValueError
        If the bracket is empty or ``f`` does not change sign.
    ConvergenceError
        If the tolerance is not met and ``raise_on_failure`` is ``True``.
    """
    left = require_finite_scalar(a, name="a")
    right = require_finite_scalar(b, name="b")
    tolerance = require_positive(tol, name="tol")
    limit = require_positive_int(max_iter, name="max_iter")
    if left >= right:
        raise ValueError("bisection requires a < b")

    f_left = _call(f, left, name="f(a)")
    f_right = _call(f, right, name="f(b)")
    if not _brackets(f_left, f_right):
        raise ValueError(
            "f(a) and f(b) must have opposite signs or a zero at an endpoint; "
            f"got f({left})={f_left}, f({right})={f_right}"
        )
    if f_left == 0.0:
        return _finish(
            method="bisection",
            root=left,
            iterations=0,
            residual=0.0,
            tolerance=tolerance,
            history=[left],
            converged=True,
            raise_on_failure=raise_on_failure,
        )
    if f_right == 0.0:
        return _finish(
            method="bisection",
            root=right,
            iterations=0,
            residual=0.0,
            tolerance=tolerance,
            history=[right],
            converged=True,
            raise_on_failure=raise_on_failure,
        )

    history: list[float] = []
    root = left
    residual = abs(f_left)
    converged = False
    iterations = limit
    for step in range(1, limit + 1):
        iterations = step
        root = left + (right - left) / 2.0
        residual_value = _call(f, root, name="f(x)")
        residual = abs(residual_value)
        history.append(root)
        half_width = (right - left) / 2.0
        if half_width <= tolerance or residual <= tolerance:
            converged = True
            break
        if _brackets(f_left, residual_value):
            right = root
        else:
            left = root
            f_left = residual_value

    return _finish(
        method="bisection",
        root=root,
        iterations=iterations,
        residual=residual,
        tolerance=tolerance,
        history=history,
        converged=converged,
        raise_on_failure=raise_on_failure,
    )


def _numerical_derivative(f: ScalarFunction, x: float, step: float) -> float:
    increment = step * max(1.0, abs(x))
    right = _call(f, x + increment, name="f(x)")
    left = _call(f, x - increment, name="f(x)")
    return (right - left) / (2.0 * increment)


def newton(
    f: ScalarFunction,
    x0: float,
    *,
    df: ScalarFunction | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    fd_step: float = 1e-8,
    raise_on_failure: bool = True,
) -> RootResult:
    """Find a root of ``f`` with the Newton-Raphson method.

    The iteration is ``x - f(x) / f'(x)``. If ``df`` is omitted, the
    derivative is approximated by a central difference whose spacing is
    ``fd_step * max(1, abs(x))``. The iteration stops when the step or the
    absolute residual is at most ``tol``.

    Parameters
    ----------
    f :
        Scalar function.
    x0 :
        Initial guess.
    df :
        Analytic derivative. Optional.
    tol :
        Positive stopping tolerance.
    max_iter :
        Maximum number of Newton updates.
    fd_step :
        Relative step used only when ``df`` is ``None``.
    raise_on_failure :
        Raise :class:`~numerica.ConvergenceError` if the tolerance is missed.

    Returns
    -------
    RootResult
        ``residual`` is ``abs(f(root))``. ``history`` starts at ``x0``.

    Raises
    ------
    NumericaError
        If a derivative is zero or non-finite, so the Newton step is undefined.
    ConvergenceError
        If the tolerance is not met and ``raise_on_failure`` is ``True``.
    """
    guess = require_finite_scalar(x0, name="x0")
    tolerance = require_positive(tol, name="tol")
    limit = require_positive_int(max_iter, name="max_iter")
    relative_step = require_positive(fd_step, name="fd_step") if df is None else None

    current = guess
    residual_value = _call(f, current, name="f(x)")
    history = [current]
    if abs(residual_value) <= tolerance:
        return _finish(
            method="newton",
            root=current,
            iterations=0,
            residual=abs(residual_value),
            tolerance=tolerance,
            history=history,
            converged=True,
            raise_on_failure=raise_on_failure,
        )

    converged = False
    iterations = limit
    for step in range(1, limit + 1):
        iterations = step
        slope = (
            _call(df, current, name="df(x)")
            if df is not None
            else _numerical_derivative(f, current, relative_step or 1e-8)
        )
        if slope == 0.0 or not math.isfinite(slope):
            raise NumericaError(
                f"Derivative is zero or non-finite at x={current:.6g}; "
                "the Newton step is undefined."
            )
        nxt = current - residual_value / slope
        nxt = require_finite_scalar(nxt, name="Newton iterate")
        residual_value = _call(f, nxt, name="f(x)")
        history.append(nxt)
        if abs(nxt - current) <= tolerance or abs(residual_value) <= tolerance:
            current = nxt
            converged = True
            break
        current = nxt

    return _finish(
        method="newton",
        root=current,
        iterations=iterations,
        residual=abs(residual_value),
        tolerance=tolerance,
        history=history,
        converged=converged,
        raise_on_failure=raise_on_failure,
    )


def secant(
    f: ScalarFunction,
    x0: float,
    x1: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 60,
    raise_on_failure: bool = True,
) -> RootResult:
    """Find a root of ``f`` with the secant method.

    Two distinct initial guesses replace the derivative in Newton's method.
    The iteration stops when the step or the absolute residual is at most
    ``tol``.

    Parameters
    ----------
    f :
        Scalar function.
    x0, x1 :
        Distinct initial guesses.
    tol :
        Positive stopping tolerance.
    max_iter :
        Maximum number of secant updates.
    raise_on_failure :
        Raise :class:`~numerica.ConvergenceError` if the tolerance is missed.

    Returns
    -------
    RootResult
        ``residual`` is ``abs(f(root))``. ``history`` starts with ``x0, x1``.

    Raises
    ------
    ValueError
        If the initial guesses are not distinct and finite.
    NumericaError
        If ``f(x1) - f(x0)`` vanishes, so the secant step is undefined.
    ConvergenceError
        If the tolerance is not met and ``raise_on_failure`` is ``True``.
    """
    previous = require_finite_scalar(x0, name="x0")
    current = require_finite_scalar(x1, name="x1")
    tolerance = require_positive(tol, name="tol")
    limit = require_positive_int(max_iter, name="max_iter")
    if previous == current:
        raise ValueError("secant requires two distinct initial guesses")

    f_previous = _call(f, previous, name="f(x0)")
    f_current = _call(f, current, name="f(x1)")
    history = [previous, current]
    if abs(f_current) <= tolerance:
        return _finish(
            method="secant",
            root=current,
            iterations=0,
            residual=abs(f_current),
            tolerance=tolerance,
            history=history,
            converged=True,
            raise_on_failure=raise_on_failure,
        )
    if abs(f_previous) <= tolerance:
        return _finish(
            method="secant",
            root=previous,
            iterations=0,
            residual=abs(f_previous),
            tolerance=tolerance,
            history=history,
            converged=True,
            raise_on_failure=raise_on_failure,
        )

    converged = False
    iterations = limit
    for step in range(1, limit + 1):
        iterations = step
        denominator = f_current - f_previous
        scale = max(1.0, abs(f_current), abs(f_previous))
        if not math.isfinite(denominator) or abs(denominator) <= 1e-15 * scale:
            raise NumericaError("Secant step is undefined because f(x1) ≈ f(x0).")
        nxt = current - f_current * (current - previous) / denominator
        nxt = require_finite_scalar(nxt, name="secant iterate")
        f_next = _call(f, nxt, name="f(x)")
        history.append(nxt)
        if abs(nxt - current) <= tolerance or abs(f_next) <= tolerance:
            current = nxt
            f_current = f_next
            converged = True
            break
        previous, f_previous = current, f_current
        current, f_current = nxt, f_next

    return _finish(
        method="secant",
        root=current,
        iterations=iterations,
        residual=abs(f_current),
        tolerance=tolerance,
        history=history,
        converged=converged,
        raise_on_failure=raise_on_failure,
    )


def fixed_point(
    g: ScalarFunction,
    x0: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
    raise_on_failure: bool = True,
) -> RootResult:
    """Find a fixed point of ``g`` by successive substitution.

    The iteration is ``x_{k+1} = g(x_k)`` and it stops when
    ``abs(x_{k+1} - x_k) <= tol``. Convergence requires a contraction near the
    fixed point; this routine does not test that condition in advance.

    Parameters
    ----------
    g :
        Iteration function.
    x0 :
        Initial guess.
    tol :
        Positive stopping tolerance on successive iterates.
    max_iter :
        Maximum number of updates.
    raise_on_failure :
        Raise :class:`~numerica.ConvergenceError` if the tolerance is missed.

    Returns
    -------
    RootResult
        ``root`` satisfies ``root ≈ g(root)``. ``residual`` is
        ``abs(g(root) - root)``.

    Raises
    ------
    ConvergenceError
        If the tolerance is not met and ``raise_on_failure`` is ``True``.
    """
    current = require_finite_scalar(x0, name="x0")
    tolerance = require_positive(tol, name="tol")
    limit = require_positive_int(max_iter, name="max_iter")
    history = [current]
    nxt = _call(g, current, name="g(x)")
    residual = abs(nxt - current)
    if residual <= tolerance:
        return _finish(
            method="fixed_point",
            root=current,
            iterations=0,
            residual=residual,
            tolerance=tolerance,
            history=history,
            converged=True,
            raise_on_failure=raise_on_failure,
        )

    converged = False
    iterations = limit
    for step in range(1, limit + 1):
        iterations = step
        nxt = _call(g, current, name="g(x)")
        history.append(nxt)
        residual = abs(nxt - current)
        if residual <= tolerance:
            current = nxt
            converged = True
            break
        current = nxt

    if converged:
        # One extra evaluation reports the true fixed-point residual.
        residual = abs(_call(g, current, name="g(x)") - current)
    return _finish(
        method="fixed_point",
        root=current,
        iterations=iterations,
        residual=residual,
        tolerance=tolerance,
        history=history,
        converged=converged,
        raise_on_failure=raise_on_failure,
    )


newton_raphson = newton
