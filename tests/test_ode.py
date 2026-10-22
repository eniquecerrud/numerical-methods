"""One-step ODE methods against closed-form solutions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from numerica import ConvergenceError, DimensionError, euler, heun, rk4, rk45


def _exponential_error(method, h: float) -> float:
    solution = method(lambda t, y: y, (0.0, 1.0), 1.0, h=h)
    return abs(float(solution.y[-1]) - math.e)


def test_one_step_matches_the_butcher_update() -> None:
    step = 0.25
    euler_step = euler(lambda t, y: y, (0.0, step), 1.0, h=step)
    heun_step = heun(lambda t, y: y, (0.0, step), 1.0, h=step)
    rk_step = rk4(lambda t, y: y, (0.0, step), 1.0, h=step)
    assert euler_step.t == pytest.approx([0.0, step])
    assert euler_step.y[-1] == pytest.approx(1.25)
    assert heun_step.y[-1] == pytest.approx(1.28125)
    assert rk_step.y[-1] == pytest.approx(7889.0 / 6144.0)
    assert euler_step.accepted_steps == 1
    assert euler_step.rejected_steps == 0


def test_observed_order_of_the_fixed_step_methods() -> None:
    euler_ratio = _exponential_error(euler, 0.2) / _exponential_error(euler, 0.1)
    heun_ratio = _exponential_error(heun, 0.2) / _exponential_error(heun, 0.1)
    rk_ratio = _exponential_error(rk4, 0.2) / _exponential_error(rk4, 0.1)
    assert 1.7 < euler_ratio < 2.4
    assert 3.0 < heun_ratio < 5.0
    assert 12.0 < rk_ratio < 20.0


def test_fixed_step_grid_lands_on_the_endpoint() -> None:
    solution = euler(lambda t, y: -y, (0.0, 1.0), 1.0, h=0.3)
    assert solution.t == pytest.approx([0.0, 0.3, 0.6, 0.9, 1.0])
    # h = 0.3 is coarse for y' = -y, so the check is only that Euler stays
    # in the neighborhood of exp(-1) and that the last panel reaches t = 1.
    assert solution.y[-1] == pytest.approx(math.exp(-1.0), abs=0.1)


def test_harmonic_oscillator_system() -> None:
    def oscillator(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([y[1], -y[0]])

    solution = rk4(oscillator, (0.0, 1.0), np.array([1.0, 0.0]), h=0.01)
    assert solution.y.shape == (101, 2)
    assert solution.y[-1, 0] == pytest.approx(math.cos(1.0), abs=1e-8)
    assert solution.y[-1, 1] == pytest.approx(-math.sin(1.0), abs=1e-8)


def test_rk45_tracks_the_exponential_in_both_directions() -> None:
    forward = rk45(lambda t, y: y, (0.0, 1.0), 1.0, rtol=1e-7, atol=1e-9)
    backward = rk45(lambda t, y: y, (1.0, 0.0), math.e, rtol=1e-7, atol=1e-9)
    assert forward.y[-1] == pytest.approx(math.e, abs=1e-6)
    assert forward.t[-1] == pytest.approx(1.0)
    assert forward.accepted_steps >= 2
    assert backward.y[-1] == pytest.approx(1.0, abs=1e-5)
    assert backward.t[0] > backward.t[-1]


def test_rk45_rejects_an_oversize_initial_step() -> None:
    solution = rk45(
        lambda t, y: -40.0 * y,
        (0.0, 1.0),
        1.0,
        rtol=1e-5,
        atol=1e-8,
        h_init=0.5,
    )
    assert solution.rejected_steps > 0
    assert solution.y[-1] == pytest.approx(math.exp(-40.0), abs=1e-4)
    assert abs(solution.y[-1]) < 1e-3


def test_degenerate_and_invalid_problems() -> None:
    steady = heun(lambda t, y: y, (1.0, 1.0), 2.0, h=0.1)
    assert steady.t.shape == (1,)
    assert steady.y[0] == 2.0
    with pytest.raises(ValueError, match="h"):
        euler(lambda t, y: y, (0.0, 1.0), 1.0, h=0.0)
    with pytest.raises(DimensionError):
        rk4(lambda t, y: np.array([y[0], 0.0, 0.0]), (0.0, 1.0), np.array([1.0, 0.0]), h=0.2)
    with pytest.raises(ConvergenceError):
        rk45(lambda t, y: y, (0.0, 1.0), 1.0, max_steps=1, h_init=0.01)
