"""Lagrange, Newton y un spline cúbico sobre los mismos nodos."""

from __future__ import annotations

import math

import numpy as np

from numerica import CubicSpline, lagrange, newton_divided_difference


def main() -> None:
    nodes = np.linspace(0.0, math.pi, 5)
    values = np.sin(nodes)
    query = math.pi / 6.0
    exact = math.sin(query)

    print(f"sin(pi/6) = {exact:.6f}")
    print(f"  Lagrange   {lagrange(nodes, values, query):.6f}")
    print(f"  Newton     {newton_divided_difference(nodes, values, query):.6f}")
    natural = CubicSpline(nodes, values)
    print(f"  spline     {natural(query):.6f}")

    print("\nUn polinomio cúbico se recupera con un spline anclado")
    sample = np.linspace(-1.0, 1.0, 4)

    def cubic(x: np.ndarray) -> np.ndarray:
        return x**3 - x

    clamped = CubicSpline(
        sample,
        cubic(sample),
        bc="clamped",
        dydx=(2.0, 2.0),
    )
    check = 0.35
    print(f"  p({check}) = {cubic(np.array(check)):.6f}")
    print(f"  S({check}) = {clamped(check):.6f}")


if __name__ == "__main__":
    main()
