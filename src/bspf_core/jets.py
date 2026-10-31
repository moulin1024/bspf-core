"""Endpoint-jet estimators: linear maps from samples to end derivatives.

A cardinal BSPF line constrains its spline to reproduce the derivatives of
orders ``0 .. q-1`` (the *jet*) of the data at both ends of the grid. The data
are samples, so those derivatives are estimated from a few samples near each
end. Every estimator here is linear and is returned as one matrix

.. math::

    J \\in \\mathbb{R}^{2q \\times N}, \\qquad
    \\begin{bmatrix} f^{(0)}(x_0) & \\cdots & f^{(q-1)}(x_0) &
    f^{(0)}(x_{N-1}) & \\cdots & f^{(q-1)}(x_{N-1}) \\end{bmatrix}^T \\approx J f,

so any estimator, including user-supplied stencils, plugs into
:func:`bspf_core.cardinal.cardinal_line` the same way. Rows ``0 .. q-1`` act on
the first ``points`` samples, rows ``q .. 2q-1`` on the last ``points``.

The right-end map is the left-end map mirrored: reversed sample order and a
sign ``(-1)^k`` for derivative order ``k``.
"""
from math import factorial

import numpy as np
import scipy.linalg as la


def _assemble(left, size):
    """Place a left-end block (q, points) and its mirror into a (2q, N) map."""
    q, points = left.shape
    if points > size:
        raise ValueError("estimator window exceeds the number of samples")
    J = np.zeros((2 * q, size))
    J[:q, :points] = left
    J[q:, -points:] = left[:, ::-1] * (-1.0) ** np.arange(q)[:, None]
    return J


def _uniform_spacing(x):
    x = np.asarray(x, dtype=float)
    h = np.diff(x)
    if x.ndim != 1 or len(x) < 2 or np.any(h <= 0) or not np.allclose(h, h[0], rtol=1e-10, atol=0):
        raise ValueError("x must be a strictly increasing uniform grid")
    return x, h[0]


def taylor_jets(x, order, points, fit="interpolation"):
    """Finite-difference (Taylor) jet estimator.

    Parameters
    ----------
    x : array_like, shape (N,)
        Uniform grid.
    order : int
        Number of derivative orders ``q`` (0 to ``q - 1``).
    points : int
        Samples used at each end, ``>= order``.
    fit : {"interpolation", "least_squares"}, optional
        ``"interpolation"`` differentiates the polynomial of degree
        ``points - 1`` through the window (``pybspf.plan_1d``,
        ``endpoint_method="finite_difference"``). ``"least_squares"`` fits a
        polynomial of degree ``order - 1`` (``pybspf._qr_trial``,
        ``endpoint_method="taylor"``). They coincide when ``points == order``.

    Returns
    -------
    ndarray, shape (2 * order, N)

    Examples
    --------
    >>> J = taylor_jets(np.linspace(0, 1, 11), order=2, points=3)
    >>> J[:, :3].round(6).tolist()[1]    # one-sided second-order first derivative
    [-15.0, 20.0, -5.0]
    """
    x, h = _uniform_spacing(x)
    if order < 0 or points < max(1, order):
        raise ValueError("need points >= max(1, order)")
    if fit not in ("interpolation", "least_squares"):
        raise ValueError("fit must be 'interpolation' or 'least_squares'")
    offsets = np.arange(points, dtype=float)
    if fit == "interpolation":
        V = offsets[None, :] ** np.arange(points)[:, None] / np.array([factorial(k) for k in range(points)])[:, None]
        left = np.linalg.solve(V, np.eye(points)[:, :order]).T
    else:
        T = np.column_stack([offsets**k / factorial(k) for k in range(order)])
        left = np.linalg.pinv(T, rcond=1e-14)
    return _assemble(left / h ** np.arange(order)[:, None], len(x))


def polynomial_jets(x, order, points, degree):
    """Least-squares polynomial jets with the end value kept exact.

    On each end window of ``points`` samples, fit a polynomial of ``degree``
    that passes through the end sample and fits the others by least squares,
    then differentiate it at the end. Rows for orders ``>= 1`` annihilate
    constants exactly.

    Parameters
    ----------
    x : array_like, shape (N,)
        Uniform grid.
    order : int
        Number of derivative orders ``q``.
    points : int
        Window size, ``> degree``; clipped to ``N``.
    degree : int
        Polynomial degree, ``>= order - 1``.

    Returns
    -------
    ndarray, shape (2 * order, N)

    Notes
    -----
    The fixed-window schemes ``"fixed{points}d{degree}"`` of the hand-off
    ``boundary_estimators.build_stencil``; the Navier-Stokes lines use
    ``points=16, degree=12``. The fit uses Legendre polynomials on the window
    mapped to [-1, 1] and a QR factorization, as there.

    Examples
    --------
    >>> x = np.linspace(0.0, 1.0, 41)
    >>> J = polynomial_jets(x, order=3, points=16, degree=12)
    >>> bool(np.allclose(J @ np.exp(x), [1, 1, 1, np.e, np.e, np.e], atol=1e-6))
    True
    """
    x, _ = _uniform_spacing(x)
    m = min(len(x), int(points))
    if m <= degree or degree < order - 1:
        raise ValueError("need points > degree >= order - 1")
    if order == 0:
        return np.zeros((0, len(x)))
    width = x[m - 1] - x[0]
    z = 2 * (x[:m] - x[0]) / width - 1
    A = np.polynomial.legendre.legvander(z, degree)
    A = A[1:, 1:] - A[0, 1:]                     # exact end value: fit f - f_0 by P - P(-1)
    Q, R = la.qr(A, mode="economic")
    inverse = la.solve_triangular(R, Q.T)
    left = np.zeros((order, m))
    left[0, 0] = 1.0
    for k in range(1, order):
        jet = np.polynomial.legendre.legval(-1.0, np.polynomial.legendre.legder(np.eye(degree + 1), m=k, axis=0))
        t = (jet * (2 / width) ** k)[1:] @ inverse
        left[k] = np.r_[-np.sum(t), t]
        left[k, 0] -= np.sum(left[k])            # constants exactly in the kernel
    return _assemble(left, len(x))


def chebyshev_jets(x, order, points, modes, alpha=1e-12, penalty_power=4):
    """Regularized local Chebyshev jet estimator.

    On each end window of ``points`` samples, fit a Chebyshev series with
    ``modes`` terms by least squares with a modal penalty, then differentiate
    the fit at the end point. Endpoint values (order 0) are copied exactly.

    Parameters
    ----------
    x : array_like, shape (N,)
        Uniform grid.
    order : int
        Number of derivative orders ``q``.
    points : int
        Window size, ``>= modes``.
    modes : int
        Number of Chebyshev terms, ``>= order``.
    alpha : float, optional
        Penalty weight ``>= 0``. Default ``1e-12``.
    penalty_power : int, optional
        The penalty on mode ``j`` is ``(j / (modes - 1))**penalty_power`` and
        zero for the first two modes. Default 4.

    Returns
    -------
    ndarray, shape (2 * order, N)

    Notes
    -----
    Same algorithm as ``pybspf.endpoints.chebyshev_boundary_blocks`` and its
    NumPy port ``pybspf._qr_trial.chebyshev_jets``: the penalised problem is
    solved by QR of the augmented matrix, without normal equations, and
    :math:`T_j^{(k)}(1) = \\prod_{l<k} (j^2 - l^2) / (2k - 1)!!`.
    """
    x, _ = _uniform_spacing(x)
    if order == 0:
        return np.zeros((0, len(x)))
    if not order <= modes <= points:
        raise ValueError("need order <= modes <= points")
    xi = np.linspace(-1.0, 1.0, points)
    V = np.polynomial.chebyshev.chebvander(xi, modes - 1)
    index = np.arange(modes, dtype=float)
    penalty = (index / max(1, modes - 1)) ** penalty_power
    penalty[: min(2, modes)] = 0
    Q, R = la.qr(np.vstack([V, np.sqrt(alpha) * np.diag(penalty)]), mode="economic")
    projector = la.solve_triangular(R, Q[:points].T)          # samples -> coefficients
    width = x[points - 1] - x[0]
    end_value = np.ones(modes)                                  # T_j^{(k)}(+1)
    left = np.empty((order, points))
    for k in range(order):
        if k:
            end_value = end_value * (index**2 - (k - 1) ** 2) / (2 * k - 1)
        left[k] = (2 / width) ** k * (((-1.0) ** (index - k) * end_value) @ projector)
    left[0] = 0
    left[0, 0] = 1                                              # exact end value
    return _assemble(left, len(x))
