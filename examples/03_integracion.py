"""Integrales de sin(x) en [0, pi], cuyo valor exacto es 2."""

from __future__ import annotations

import math

from numerica import gauss_legendre, romberg, simpson, simpson_38, trapezoidal


def main() -> None:
    exact = 2.0
    rules = (
        trapezoidal(math.sin, 0.0, math.pi, n=16),
        simpson(math.sin, 0.0, math.pi, n=16),
        simpson_38(math.sin, 0.0, math.pi, n=15),
        gauss_legendre(math.sin, 0.0, math.pi, n=6),
        romberg(math.sin, 0.0, math.pi, tol=1e-12),
    )
    print(f"Integral exacta: {exact:.12f}")
    for result in rules:
        error = abs(result.value - exact)
        print(
            f"  {result.method:<16} {result.value:.12f}   "
            f"error {error:.2e}   evaluaciones {result.evaluations}"
        )


if __name__ == "__main__":
    main()
