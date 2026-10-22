"""Finite differences against analytic derivatives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from numerica import derivative


def test_central_difference_matches_cosine() -> None:
    points = np.array([0.2, 0.5, 1.0, -0.7])
    assert derivative(math.sin, points) == pytest.approx(np.cos(points), abs=1e-8)
    assert derivative(math.sin, 0.3) == pytest.approx(math.cos(0.3), abs=1e-8)
    assert isinstance(derivative(math.sin, 0.3), float)


def test_second_derivative_matches_negative_sine() -> None:
    value = derivative(math.sin, 0.5, order=2)
    assert value == pytest.approx(-math.sin(0.5), abs=1e-6)


def test_central_difference_is_exact_for_a_quadratic() -> None:
    assert derivative(lambda x: x * x, 1000.0) == pytest.approx(2000.0, rel=1e-8)
    assert derivative(lambda x: x * x, 3.0, order=2) == pytest.approx(2.0, abs=1e-6)


def test_central_difference_is_more_accurate_than_forward() -> None:
    exact = math.cos(0.4)
    forward = abs(derivative(math.sin, 0.4, method="forward", step=1e-3) - exact)
    central = abs(derivative(math.sin, 0.4, method="central", step=1e-3) - exact)
    backward = abs(derivative(math.sin, 0.4, method="backward", step=1e-3) - exact)
    assert central < forward / 10.0
    assert central < backward / 10.0


def test_invalid_difference_options() -> None:
    with pytest.raises(ValueError, match="order"):
        derivative(math.sin, 0.0, order=3)
    with pytest.raises(ValueError, match="method"):
        derivative(math.sin, 0.0, method="complex")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step"):
        derivative(math.sin, 0.0, step=-1e-3)
