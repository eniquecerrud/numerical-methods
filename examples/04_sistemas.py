"""Eliminación, LU y los métodos iterativos sobre el mismo sistema."""

from __future__ import annotations

import numpy as np

from numerica import gauss_seidel, gaussian_elimination, jacobi, lu_factor


def main() -> None:
    matrix = np.array(
        [
            [10.0, -1.0, 2.0],
            [-1.0, 11.0, -1.0],
            [2.0, -1.0, 10.0],
        ]
    )
    expected = np.array([1.0, 2.0, -1.0])
    rhs = matrix @ expected

    direct = gaussian_elimination(matrix, rhs)
    factors = lu_factor(matrix)
    iterative = jacobi(matrix, rhs, tol=1e-10)
    seidel = gauss_seidel(matrix, rhs, tol=1e-10)

    print("Solución exacta          ", expected)
    print("Eliminación de Gauss     ", direct)
    print("LU                       ", factors.solve(rhs))
    print(f"Jacobi                   {iterative.solution}  ({iterative.iterations} iteraciones)")
    print(f"Gauss-Seidel             {seidel.solution}  ({seidel.iterations} iteraciones)")
    residual = np.linalg.norm(factors.p @ matrix - factors.lower @ factors.u)
    print("||PA - LU||              ", residual)


if __name__ == "__main__":
    main()
