"""Analytic checks for scalar root finders."""

from __future__ import annotations

import math

import pytest

from numerica import (
    ConvergenceError,
    NumericaError,
    bisection,
    fixed_point,
    newton,
    newton_raphson,
    secant,
)


def test_bisection_finds_square_root() -> None:
    result = bisection(lambda x: x * x - 2.0, 0.0, 2.0, tol=1e-12)
    assert result.converged
    assert result.root == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert result.residual <= 1e-10
    assert result.history[-1] == result.root
    assert result.iterations >= 1
    assert str(result)


def test_bisection_accepts_an_endpoint_root() -> None:
    result = bisection(lambda x: x - 2.0, 2.0, 5.0)
    assert result.root == 2.0
    assert result.iterations == 0
    assert result.residual == 0.0


def test_bisection_rejects_a_bracket_without_a_sign_change() -> None:
    with pytest.raises(ValueError, match="opposite signs"):
        bisection(lambda x: x * x + 1.0, -1.0, 1.0)


def test_bisection_can_return_a_failed_iteration() -> None:
    result = bisection(
        lambda x: x - 0.3,
        0.0,
        1.0,
        tol=1e-15,
        max_iter=2,
        raise_on_failure=False,
    )
    assert result.converged is False
    assert result.iterations == 2


def test_newton_with_and_without_a_derivative() -> None:
    def f(x: float) -> float:
        return x * x - 2.0

    analytic = newton(f, 1.0, df=lambda x: 2.0 * x, tol=1e-14)
    numeric = newton_raphson(f, 1.0, tol=1e-12)
    assert analytic.root == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert numeric.root == pytest.approx(math.sqrt(2.0), abs=1e-10)
    assert analytic.iterations < 10


def test_newton_stops_on_a_zero_derivative() -> None:
    with pytest.raises(NumericaError, match="Derivative"):
        newton(lambda x: (x - 1.0) ** 2 + 1.0, 1.0, df=lambda x: 2.0 * (x - 1.0))


def test_secant_finds_square_root() -> None:
    result = secant(lambda x: x * x - 2.0, 1.0, 2.0, tol=1e-12)
    assert result.root == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert result.history[0] == 1.0
    assert result.history[1] == 2.0


def test_secant_rejects_a_vanishing_denominator() -> None:
    with pytest.raises(NumericaError, match="Secant"):
        secant(lambda x: x * x, 1.0, -1.0)


def test_secant_requires_distinct_guesses() -> None:
    with pytest.raises(ValueError, match="distinct"):
        secant(math.sin, 1.0, 1.0)


def test_fixed_point_babylonian_square_root() -> None:
    result = fixed_point(lambda x: 0.5 * (x + 2.0 / x), 1.0, tol=1e-14)
    assert result.root == pytest.approx(math.sqrt(2.0), abs=1e-12)
    assert result.residual <= 1e-14


def test_fixed_point_failure_exposes_the_partial_result() -> None:
    with pytest.raises(ConvergenceError) as caught:
        fixed_point(lambda x: x + 1.0, 0.0, tol=1e-12, max_iter=4)
    assert caught.value.iterations == 4
    assert caught.value.tolerance == 1e-12
    assert caught.value.value == 4.0


def test_methods_agree_on_the_dottie_number() -> None:
    target = 0.7390851332151607
    bracket = bisection(lambda x: math.cos(x) - x, 0.0, 1.0, tol=1e-12)
    iterated = fixed_point(math.cos, 0.5, tol=1e-12)
    assert bracket.root == pytest.approx(target, abs=1e-10)
    assert iterated.root == pytest.approx(target, abs=1e-10)
