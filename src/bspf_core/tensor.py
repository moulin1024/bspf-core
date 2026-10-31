"""Tensor-product evaluation and fast diagonalization.

On a tensor-product block a field is :math:`u(x, y) = \\sum_{ij} C_{ij} X_i(x) Y_j(y)`.
With ``X`` and ``Y`` the basis values (or derivatives) at quadrature points,
all point values are ``X @ C @ Y.T`` and the Galerkin adjoint (testing a
point field ``F`` against every basis product) is ``X.T @ (F) @ Y``. No 2D
matrix is ever formed. :func:`evaluate` and :func:`adjoint` implement these,
and :func:`apply_axes` generalises them to any number of axes.

Fast diagonalization
--------------------
For operators of the two-term form

.. math::

    A = K_x \\otimes M_y + M_x \\otimes K_y + \\sigma\\, M_x \\otimes M_y

with symmetric ``K`` and symmetric positive-definite ``M``, the generalized
eigendecompositions :math:`K_x V_x = M_x V_x \\Lambda_x` with
:math:`V_x^T M_x V_x = I` (and the same in ``y``) give the exact inverse

.. math::

    A^{-1} F = V_x \\left[ (V_x^T F V_y) \\oslash (\\lambda_x \\mathbf 1^T + \\mathbf 1 \\lambda_y^T + \\sigma) \\right] V_y^T,

costing two small dense products per side. This is the Lynch-Rice-Thomas
method; :class:`FastDiagonalization` performs the host setup and a JAX solve.
It replaces ``pybspf.tensor.tensor_elliptic_solve`` and several
reimplementations in the elliptic solvers.
"""
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la


def evaluate(left, right, coefficients):
    """Point values of a 2D tensor-product field.

    Parameters
    ----------
    left : array_like, shape (px, nx)
        Basis values (or a derivative) along the first axis at its points.
    right : array_like, shape (py, ny)
        Same for the second axis.
    coefficients : array_like, shape (nx, ny)

    Returns
    -------
    array, shape (px, py)
        ``left @ coefficients @ right.T``.
    """
    return left @ coefficients @ right.T


def adjoint(left, right, values):
    """Galerkin adjoint of :func:`evaluate`.

    Parameters
    ----------
    left : array_like, shape (px, nx)
    right : array_like, shape (py, ny)
    values : array_like, shape (px, py)
        Point values already multiplied by the quadrature weights.

    Returns
    -------
    array, shape (nx, ny)
        ``left.T @ values @ right``, i.e. integrals against every basis product.
    """
    return left.T @ values @ right


def apply_axes(matrices, array):
    """Apply one matrix per leading axis of an array.

    Parameters
    ----------
    matrices : sequence of array_like or None
        ``matrices[d]`` has shape ``(m_d, n_d)`` and acts on axis ``d``;
        ``None`` leaves that axis unchanged.
    array : array_like, shape (n_0, n_1, ..., batch...)
        Trailing dimensions beyond ``len(matrices)`` are batches.

    Returns
    -------
    array, shape (m_0, m_1, ..., batch...)

    Examples
    --------
    >>> X, Y, C = np.ones((3, 2)), np.ones((4, 5)), np.ones((2, 5))
    >>> apply_axes([X, Y], C).shape
    (3, 4)
    """
    xp = jnp if any(isinstance(a, jnp.ndarray) for a in (*[m for m in matrices if m is not None], array)) else np
    out = array
    for d, M in enumerate(matrices):
        if M is not None:
            out = xp.moveaxis(xp.tensordot(M, out, axes=(1, d)), 0, d)
    return out


class FastDiagonalization:
    """Exact solver for ``K_x (x) M_y + M_x (x) K_y + sigma M_x (x) M_y``.

    Parameters
    ----------
    Kx, Mx : array_like, shape (nx, nx)
        Symmetric ``K`` and symmetric positive-definite ``M`` along the first axis.
    Ky, My : array_like, shape (ny, ny)
        The same along the second axis.
    sigma : float, optional
        Shift. Default 0.

    Attributes
    ----------
    Vx, Vy : ndarray
        Mass-orthonormal generalized eigenvectors.
    denominator : ndarray, shape (nx, ny)
        ``lambda_x[:, None] + lambda_y[None, :] + sigma``.

    Raises
    ------
    ValueError
        If the operator is singular (a zero denominator).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> def spd(n): a = rng.standard_normal((n, n)); return a @ a.T + n * np.eye(n)
    >>> Kx, Mx, Ky, My = spd(4), spd(4), spd(3), spd(3)
    >>> fd = FastDiagonalization(Kx, Mx, Ky, My, sigma=0.5)
    >>> A = np.kron(Kx, My) + np.kron(Mx, Ky) + 0.5 * np.kron(Mx, My)
    >>> F = rng.standard_normal((4, 3))
    >>> bool(np.allclose(fd.solve(F).ravel(), np.linalg.solve(A, F.ravel())))
    True
    """

    def __init__(self, Kx, Mx, Ky, My, sigma=0.0):
        lx, self.Vx = la.eigh(np.asarray(Kx), np.asarray(Mx))
        ly, self.Vy = la.eigh(np.asarray(Ky), np.asarray(My))
        self.denominator = lx[:, None] + ly[None, :] + sigma
        if np.any(self.denominator == 0):
            raise ValueError("singular operator: zero generalized eigenvalue sum")

    def solve(self, load):
        """Apply the exact inverse to a load (NumPy or JAX array).

        Parameters
        ----------
        load : array_like, shape (nx, ny)
            Right-hand side in coefficient space.

        Returns
        -------
        array, shape (nx, ny)
        """
        xp = jnp if isinstance(load, jnp.ndarray) else np
        Vx, Vy, d = (xp.asarray(a) for a in (self.Vx, self.Vy, self.denominator))
        return Vx @ ((Vx.T @ load @ Vy) / d) @ Vy.T
