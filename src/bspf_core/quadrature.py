"""Gauss-Legendre quadrature on intervals and composite rules.

Every Galerkin assembly in BSPF needs integrals of products of basis functions
and coefficients. Coefficients and some bases are only piecewise smooth (for
example piecewise-linear sound-speed profiles, or B-splines with interior
knots), so rules are split at those breakpoints; a Gauss rule is then exact,
or spectrally accurate, on each smooth piece.

The node and weight computation itself is NumPy's
:func:`numpy.polynomial.legendre.leggauss`; this module only maps and joins
rules. It replaces about a dozen private copies in the existing solvers.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def gauss_legendre(order, interval=(0.0, 1.0)):
    """Gauss-Legendre nodes and weights on an interval.

    Parameters
    ----------
    order : int
        Number of nodes. The rule integrates polynomials of degree
        ``2 * order - 1`` exactly.
    interval : tuple of float, optional
        Integration interval ``(a, b)`` with ``a < b``. Default ``(0, 1)``.

    Returns
    -------
    nodes : ndarray, shape (order,)
        Increasing nodes inside ``(a, b)``.
    weights : ndarray, shape (order,)
        Positive weights summing to ``b - a``.

    Raises
    ------
    ValueError
        If ``order < 1`` or the interval is empty or not finite.

    Examples
    --------
    >>> x, w = gauss_legendre(3)
    >>> float(round(w @ x**5, 15))   # exact: 1/6
    0.166666666666667
    """
    a, b = map(float, interval)
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    if not (np.isfinite(a) and np.isfinite(b) and a < b):
        raise ValueError("interval must be finite with a < b")
    t, w = leggauss(int(order))
    return a + (b - a) * (t + 1) / 2, (b - a) / 2 * w


def composite_gauss(breakpoints, order):
    """Composite Gauss-Legendre rule over consecutive sub-intervals.

    The interval ``[breakpoints[0], breakpoints[-1]]`` is split at every
    breakpoint and an ``order``-point rule is used on each piece. Repeated
    breakpoints are ignored.

    Parameters
    ----------
    breakpoints : array_like
        Increasing breakpoints, at least two distinct values. Interior values
        mark kinks or knots of the integrand.
    order : int
        Nodes per sub-interval.

    Returns
    -------
    nodes : ndarray, shape (order * pieces,)
        Increasing nodes.
    weights : ndarray, shape (order * pieces,)
        Weights summing to the interval length.

    Raises
    ------
    ValueError
        If the breakpoints are not finite and non-decreasing, or fewer than
        two are distinct.

    Examples
    --------
    A piecewise-linear function with a kink at 0.3 is integrated exactly:

    >>> x, w = composite_gauss([0.0, 0.3, 1.0], 2)
    >>> float(round(w @ abs(x - 0.3), 15))  # (0.09 + 0.49) / 2
    0.29
    """
    edges = np.asarray(breakpoints, dtype=float)
    if edges.ndim != 1 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) < 0):
        raise ValueError("breakpoints must be finite and non-decreasing")
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError("at least two distinct breakpoints are required")
    t, w = leggauss(int(order))
    h = np.diff(edges)
    nodes = (edges[:-1, None] + h[:, None] * (t + 1) / 2).ravel()
    weights = (h[:, None] / 2 * w).ravel()
    return nodes, weights
