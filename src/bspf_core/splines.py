"""Clamped B-spline knots and basis matrices.

Cardinal BSPF lines (:mod:`bspf_core.cardinal`) use one clamped B-spline
space on the whole interval to absorb the non-periodic behaviour at its ends.
This module builds the knot vectors and evaluates the basis; evaluation is
:class:`scipy.interpolate.BSpline` (Cox-de Boor), not a new implementation.

A clamped knot vector of degree ``p`` with ``n`` functions on ``[a, b]`` has
``p + 1`` copies of each end point and ``n - p - 1`` interior knots, so the
basis interpolates the end values and its derivatives at the ends depend only
on the first (or last) ``p + 1`` functions.
"""
import numpy as np
from scipy.interpolate import BSpline


def clamped_knots(a, b, *, degree, n_basis, clustering=0.0):
    """Clamped knot vector on ``[a, b]``, optionally clustered at both ends.

    Parameters
    ----------
    a, b : float
        Interval end points, ``a < b``.
    degree : int
        Spline degree ``p >= 1``.
    n_basis : int
        Number of basis functions, ``>= degree + 1``.
    clustering : float, optional
        ``0`` gives uniform interior knots. A positive value ``s`` maps the
        uniform break points ``u`` in [-1, 1] through ``tanh(s u) / tanh(s)``,
        concentrating knots near both ends. Default 0.

    Returns
    -------
    ndarray, shape (n_basis + degree + 1,)
        Non-decreasing knots, exactly ``a`` and ``b`` at the ends.

    Notes
    -----
    Same construction as ``pybspf.basis.open_knots`` (uniform case also as in
    the hand-off ``open_uniform_knots`` and ``pybspf._qr_trial``).

    Examples
    --------
    >>> clamped_knots(0.0, 1.0, degree=2, n_basis=4).tolist()
    [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    """
    if not a < b or degree < 1 or n_basis < degree + 1:
        raise ValueError("need a < b, degree >= 1 and n_basis >= degree + 1")
    if not np.isfinite(clustering) or clustering < 0:
        raise ValueError("clustering must be finite and non-negative")
    u = np.linspace(-1.0, 1.0, n_basis - degree + 1)
    if clustering:
        u = np.tanh(clustering * u) / np.tanh(clustering)
    breaks = a + (b - a) * (u + 1.0) / 2.0
    breaks[0], breaks[-1] = a, b            # exact clamped ends despite rounding
    return np.r_[np.repeat(a, degree), breaks, np.repeat(b, degree)]


def basis_matrix(knots, degree, x, derivative=0):
    """B-spline basis values (or derivatives) at points.

    Parameters
    ----------
    knots : array_like
        Clamped knot vector.
    degree : int
        Spline degree.
    x : array_like, shape (points,)
        Points inside ``[knots[0], knots[-1]]``. At the right end the
        left-hand limit is used.
    derivative : int, optional
        Derivative order (physical units). Default 0.

    Returns
    -------
    ndarray, shape (points, n_basis)

    Raises
    ------
    ValueError
        If a point lies outside the knot interval (no implicit extrapolation).
    """
    t = np.asarray(knots, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if np.any(x < t[0]) or np.any(x > t[-1]):
        raise ValueError("points must lie inside the knot interval")
    n_basis = len(t) - degree - 1
    return BSpline(t, np.eye(n_basis), degree)(x, nu=derivative)


def primitive_matrix(knots, degree, x):
    """Values of the basis primitives :math:`\\int_{t_0}^{x} B_j`.

    Parameters
    ----------
    knots : array_like
    degree : int
    x : array_like, shape (points,)

    Returns
    -------
    ndarray, shape (points, n_basis)
        Exact integrals from the left end of the knot vector.
    """
    t = np.asarray(knots, dtype=float)
    n_basis = len(t) - degree - 1
    return BSpline(t, np.eye(n_basis), degree).antiderivative()(np.atleast_1d(x))


def _mp_basis(t, degree, x, derivative=0, primitive=False):
    """All basis values (a derivative, or primitives from ``t[0]``) at one point, exactly typed.

    ``t`` is a list of knots and ``x`` a point of one number type (``gmpy2.mpfr``
    in practice, but any field type works, e.g. :class:`fractions.Fraction`);
    the arithmetic stays in that type. Cox-de Boor recursion on the knot span
    of ``x``; at the right end the last non-empty span is used, which gives
    the left-hand limit exactly. Used by
    :class:`bspf_core.cardinal.RemainderFactors`.
    """
    zero = x - x
    one = zero + 1
    if primitive:
        # int_{t_0}^x B_{j,p} = (t_{j+p+1} - t_j)/(p+1) * sum_{i>j} B_{i,p+1}(x) on [t_0] + t + [t_end]
        values = _mp_basis([t[0]] + list(t) + [t[-1]], degree + 1, x)
        return [(t[j + degree + 1] - t[j]) / (degree + 1) * sum(values[j + 1:], zero)
                for j in range(len(t) - degree - 1)]
    spans = [i for i in range(len(t) - 1) if t[i] < t[i + 1]]
    span = next((i for i in spans if t[i] <= x < t[i + 1]), spans[-1])
    levels = [[one if i == span else zero for i in range(len(t) - 1)]]
    for q in range(1, degree + 1):
        low = levels[-1]
        level = []
        for i in range(len(t) - q - 1):
            value = zero
            if t[i + q] != t[i]:
                value += (x - t[i]) / (t[i + q] - t[i]) * low[i]
            if t[i + q + 1] != t[i + 1]:
                value += (t[i + q + 1] - x) / (t[i + q + 1] - t[i + 1]) * low[i + 1]
            level.append(value)
        levels.append(level)

    def differentiate(q, r):
        if r == 0:
            return levels[q]
        low = differentiate(q - 1, r - 1)
        out = []
        for i in range(len(t) - q - 1):
            value = zero
            if t[i + q] != t[i]:
                value += q / (t[i + q] - t[i]) * low[i]
            if t[i + q + 1] != t[i + 1]:
                value -= q / (t[i + q + 1] - t[i + 1]) * low[i + 1]
            out.append(value)
        return out

    if derivative > degree:
        return [zero] * (len(t) - degree - 1)
    return differentiate(degree, derivative)
