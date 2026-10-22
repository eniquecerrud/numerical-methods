"""Explicit one-step methods for ordinary differential equations.

The equation is ``y' = f(t, y)`` with ``y(t0) = y0``. A scalar problem is a
Python float initial value; a system is a one-dimensional array. Scalar
solutions are returned with shape ``(n_times,)`` and systems with shape
``(n_times, dim)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from numerica._common import require_finite_scalar, require_positive
from numerica.errors import ConvergenceError, DimensionError
from numerica.results import ODEResult

ODEFunction = Callable[[float, float | np.ndarray], float | np.ndarray]

# Dormand–Prince 5(4), from J. R. Dormand and P. J. Prince, J. Comput. Appl.
# Math. 6 (1980). The last row equals the fifth-order weights, so the method
# is FSAL: the final stage is f at the accepted step.
_DP_C = (1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
_DP_A = (
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
_DP_B = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
_DP_E = (
    35.0 / 384.0 - 5179.0 / 57600.0,
    0.0,
    500.0 / 1113.0 - 7571.0 / 16695.0,
    125.0 / 192.0 - 393.0 / 640.0,
    -2187.0 / 6784.0 + 92097.0 / 339200.0,
    11.0 / 84.0 - 187.0 / 2100.0,
    -1.0 / 40.0,
)


def _prepare_ivp(
    t_span: ArrayLike,
    y0: ArrayLike,
) -> tuple[float, float, np.ndarray, bool]:
    span = np.asarray(t_span, dtype=float).reshape(-1)
    if span.size != 2 or not np.all(np.isfinite(span)):
        raise DimensionError("t_span must be a pair of finite numbers (t0, tf)")
    t0 = float(span[0])
    tf = float(span[1])
    initial = np.asarray(y0, dtype=float)
    if initial.ndim == 0:
        if not np.isfinite(initial):
            raise ValueError("y0 must be finite")
        return t0, tf, np.array([float(initial)]), True
    if initial.ndim != 1:
        raise DimensionError(f"y0 must be a scalar or a 1-D vector, got shape {initial.shape}")
    if initial.size == 0 or not np.all(np.isfinite(initial)):
        raise ValueError("y0 must be a non-empty finite vector")
    return t0, tf, initial.copy(), False


def _rhs(f: ODEFunction, t: float, y: np.ndarray, *, scalar: bool) -> np.ndarray:
    argument: float | np.ndarray = float(y[0]) if scalar else y.copy()
    try:
        raw = f(t, argument)
    except TypeError as exc:
        raise TypeError("f must be callable as f(t, y)") from exc
    value = np.asarray(raw, dtype=float)
    if scalar:
        if value.shape not in ((), (1,)):
            raise DimensionError(f"scalar f must return a float, got shape {value.shape}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("f returned a non-finite value")
        return np.array([number])
    if value.shape != (y.size,):
        raise DimensionError(f"f returned shape {value.shape}, expected {(y.size,)}")
    if not np.all(np.isfinite(value)):
        raise ValueError("f returned a non-finite value")
    return value


def _pack(
    t: np.ndarray,
    states: np.ndarray,
    *,
    scalar: bool,
    method: str,
    rejected: int,
) -> ODEResult:
    output = states[:, 0] if scalar else states
    return ODEResult(
        t=t,
        y=output,
        method=method,
        accepted_steps=int(t.size - 1),
        rejected_steps=rejected,
    )


def _time_nodes(t0: float, tf: float, h: float) -> np.ndarray:
    step_size = require_positive(h, name="h")
    if t0 == tf:
        return np.array([t0])
    step = math.copysign(step_size, tf - t0)
    n_full = int(math.floor((tf - t0) / step))
    nodes = t0 + step * np.arange(n_full + 1, dtype=float)
    if abs(float(nodes[-1]) - tf) <= 1e-9 * max(1.0, abs(tf), abs(step)):
        nodes[-1] = tf
    else:
        nodes = np.append(nodes, tf)
    return nodes


def _fixed_step(
    f: ODEFunction,
    t_span: ArrayLike,
    y0: ArrayLike,
    h: float,
    method: str,
    advance: Callable[[ODEFunction, float, np.ndarray, float, bool], np.ndarray],
) -> ODEResult:
    t0, tf, state, scalar = _prepare_ivp(t_span, y0)
    grid = _time_nodes(t0, tf, h)
    states = np.empty((grid.size, state.size))
    states[0] = state
    for index in range(grid.size - 1):
        hi = float(grid[index + 1] - grid[index])
        state = advance(f, float(grid[index]), state, hi, scalar)
        states[index + 1] = state
    return _pack(grid, states, scalar=scalar, method=method, rejected=0)


def _euler_advance(f: ODEFunction, t: float, y: np.ndarray, h: float, scalar: bool) -> np.ndarray:
    return y + h * _rhs(f, t, y, scalar=scalar)


def _heun_advance(f: ODEFunction, t: float, y: np.ndarray, h: float, scalar: bool) -> np.ndarray:
    k1 = _rhs(f, t, y, scalar=scalar)
    k2 = _rhs(f, t + h, y + h * k1, scalar=scalar)
    return y + 0.5 * h * (k1 + k2)


def _rk4_advance(f: ODEFunction, t: float, y: np.ndarray, h: float, scalar: bool) -> np.ndarray:
    k1 = _rhs(f, t, y, scalar=scalar)
    k2 = _rhs(f, t + 0.5 * h, y + 0.5 * h * k1, scalar=scalar)
    k3 = _rhs(f, t + 0.5 * h, y + 0.5 * h * k2, scalar=scalar)
    k4 = _rhs(f, t + h, y + h * k3, scalar=scalar)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler(f: ODEFunction, t_span: ArrayLike, y0: ArrayLike, *, h: float = 0.1) -> ODEResult:
    """Explicit Euler method with fixed step ``h``.

    The last step is shortened so the nodes finish exactly at ``t_span[1]``.
    ``h`` is a positive step magnitude; the direction follows ``t_span``.
    """
    return _fixed_step(f, t_span, y0, h, "euler", _euler_advance)


def heun(f: ODEFunction, t_span: ArrayLike, y0: ArrayLike, *, h: float = 0.1) -> ODEResult:
    """Heun's method (improved Euler, explicit trapezoidal rule).

    Each step averages the slopes at the beginning and at the Euler predictor.
    The global error is second order for smooth problems.
    """
    return _fixed_step(f, t_span, y0, h, "heun", _heun_advance)


def rk4(f: ODEFunction, t_span: ArrayLike, y0: ArrayLike, *, h: float = 0.1) -> ODEResult:
    """Classical fourth-order Runge-Kutta with fixed step ``h``."""
    return _fixed_step(f, t_span, y0, h, "rk4", _rk4_advance)


def rk45(
    f: ODEFunction,
    t_span: ArrayLike,
    y0: ArrayLike,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h_init: float | None = None,
    h_min: float = 1e-12,
    h_max: float | None = None,
    max_steps: int = 10_000,
) -> ODEResult:
    """Adaptive Dormand–Prince Runge-Kutta 5(4).

    A fifth-order step is accepted when the embedded fourth-order error,
    scaled by ``atol + rtol * max(|y|, |y_new|)``, has RMS norm at most 1.
    Rejected steps are retried with a smaller increment. The solution values
    are the fifth-order states.

    Parameters
    ----------
    f :
        Right-hand side ``f(t, y)``.
    t_span :
        ``(t0, tf)``. The integration may run backward.
    y0 :
        Scalar or vector initial value.
    rtol, atol :
        Relative and absolute tolerances. At least one must be positive, and
        neither may be negative.
    h_init :
        Initial step magnitude. The default is ``abs(tf - t0) / 50``.
    h_min, h_max :
        Bounds on the step magnitude. ``h_max`` defaults to the length of
        the interval.
    max_steps :
        Maximum number of attempted steps, accepted and rejected.

    Returns
    -------
    ODEResult
        Accepted nodes and values. ``rejected_steps`` counts discarded trials.

    Raises
    ------
    ConvergenceError
        If the tolerance cannot be met with ``h_min``, or ``max_steps`` is
        exhausted. ``value`` is the last accepted state.
    """
    t0, tf, state, scalar = _prepare_ivp(t_span, y0)
    relative = require_finite_scalar(rtol, name="rtol")
    absolute = require_finite_scalar(atol, name="atol")
    if relative < 0.0 or absolute < 0.0:
        raise ValueError("rtol and atol must be nonnegative")
    if relative == 0.0 and absolute == 0.0:
        raise ValueError("at least one of rtol and atol must be positive")
    minimum = require_positive(h_min, name="h_min")
    span = abs(tf - t0)
    maximum = span if h_max is None else require_positive(h_max, name="h_max")
    if maximum < minimum:
        raise ValueError("h_max must be >= h_min")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
        raise TypeError("max_steps must be an integer")
    if int(max_steps) < 1:
        raise ValueError("max_steps must be >= 1")
    limit = int(max_steps)

    if t0 == tf:
        return _pack(np.array([t0]), state.reshape(1, -1), scalar=scalar, method="rk45", rejected=0)

    direction = 1.0 if tf > t0 else -1.0
    if h_init is None:
        magnitude = min(maximum, max(minimum, span / 50.0))
    else:
        magnitude = require_positive(h_init, name="h_init")
        magnitude = min(maximum, max(minimum, magnitude))

    t = t0
    y = state.copy()
    k1 = _rhs(f, t, y, scalar=scalar)
    times = [t]
    samples = [y.copy()]
    accepted = 0
    rejected = 0
    h = direction * magnitude

    while (tf - t) * direction > 0.0:
        if accepted + rejected >= limit:
            raise ConvergenceError(
                f"rk45 did not reach t={tf} within {limit} steps.",
                iterations=accepted + rejected,
                residual=float("inf"),
                tolerance=1.0,
                value=samples[-1][0] if scalar else samples[-1],
            )
        if (t + h - tf) * direction > 0.0:
            h = tf - t

        stages = [k1]
        for coefficients, node in zip(_DP_A, _DP_C, strict=True):
            increment = np.zeros_like(y)
            for weight, stage in zip(coefficients, stages, strict=True):
                increment = increment + weight * stage
            stages.append(_rhs(f, t + node * h, y + h * increment, scalar=scalar))

        increment = np.zeros_like(y)
        error = np.zeros_like(y)
        for weight, error_weight, stage in zip(_DP_B, _DP_E, stages, strict=True):
            increment = increment + weight * stage
            error = error + error_weight * stage
        y_new = y + h * increment
        error = h * error
        scale = absolute + relative * np.maximum(np.abs(y), np.abs(y_new))
        scale = np.maximum(scale, np.finfo(float).tiny)
        err_norm = float(np.linalg.norm(error / scale) / math.sqrt(y.size))

        if abs(h) <= minimum * (1.0 + 1e-12) and err_norm > 1.0:
            raise ConvergenceError(
                f"rk45 could not meet the tolerance at t={t:.6g} with h_min={minimum:.3e}.",
                iterations=accepted + rejected + 1,
                residual=err_norm,
                tolerance=1.0,
                value=y[0] if scalar else y.copy(),
            )

        if err_norm <= 1.0:
            t_new = t + h
            if (tf - t_new) * direction <= 1e-14 * max(1.0, abs(tf)):
                t_new = tf
            if t_new == t and t != tf:
                raise ConvergenceError(
                    "rk45 step size underflowed before reaching the final time.",
                    iterations=accepted + rejected + 1,
                    residual=err_norm,
                    tolerance=1.0,
                    value=y[0] if scalar else y.copy(),
                )
            y = y_new
            t = t_new
            k1 = stages[-1]
            accepted += 1
            times.append(t)
            samples.append(y.copy())
            if t == tf:
                break
        else:
            rejected += 1

        growth = 10.0 if err_norm == 0.0 else 0.9 * err_norm ** (-0.2)
        factor = min(10.0, max(0.2, growth))
        h = direction * min(maximum, max(minimum, abs(h) * factor))

    grid = np.array(times, dtype=float)
    states = np.vstack(samples)
    return ODEResult(
        t=grid,
        y=states[:, 0] if scalar else states,
        method="rk45",
        accepted_steps=accepted,
        rejected_steps=rejected,
    )
