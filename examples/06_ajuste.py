"""Ajuste polinómico y método de la potencia."""

from __future__ import annotations

import math

import numpy as np

from numerica import inverse_power_method, polynomial_fit, power_method


def main() -> None:
    nodes = np.linspace(-1.0, 1.0, 7)
    noise = np.array([0.02, -0.01, 0.00, 0.03, -0.02, 0.01, -0.01])
    observations = 1.0 + 0.5 * nodes - nodes**2 + noise
    fit = polynomial_fit(nodes, observations, 2)
    print("Coeficientes (grado 0 a 2):", np.array2string(fit.coefficients, precision=4))
    print(f"R² = {fit.r_squared:.4f},  p(0.25) = {fit(0.25):.4f}")

    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    dominant = power_method(matrix)
    # [1, 1] ya es el autovector de λ = 3; hay que arrancar fuera de ese eje.
    closest = inverse_power_method(matrix, shift=0.5, x0=[1.0, 0.0])
    print(f"\nAutovalor dominante ≈ {dominant.eigenvalue:.6f}  (exacto 3)")
    print(f"Vector               {dominant.eigenvector}")
    print(f"Componente de referencia {1.0 / math.sqrt(2.0):.6f}")
    print(f"Más cercano a 0.5    {closest.eigenvalue:.6f}  (exacto 1)")


if __name__ == "__main__":
    main()
