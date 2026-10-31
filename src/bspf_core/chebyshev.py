"""Chebyshev series on the reference interval [0, 1].

All bases in this package are stored as Chebyshev coefficient matrices. Column
``j`` of a coefficient matrix ``C`` (shape ``(terms, functions)``) represents

.. math::

    \\phi_j(x) = \\sum_k C_{kj}\\, T_k(2x - 1), \\qquad x \\in [0, 1],

where :math:`T_k` is the Chebyshev polynomial of the first kind. This
representation is exact for polynomial blocks, converges super-exponentially
for Fourier and exponential functions, and gives derivatives by an exact
coefficient recurrence.

Evaluation and differentiation are NumPy's
:func:`numpy.polynomial.chebyshev.chebval` and
:func:`numpy.polynomial.chebyshev.chebder`; this module fixes the [0, 1]
convention (one factor of 2 per derivative) in one place.
"""
import numpy as np
from numpy.polynomial import chebyshev as _cheb


def evaluate(coefficients, x):
    """Evaluate Chebyshev series at points of [0, 1].

    Parameters
    ----------
    coefficients : array_like, shape (terms, functions)
        Chebyshev coefficients in the variable ``t = 2x - 1``, one column per
        function.
    x : array_like, shape (points,)
        Evaluation points, normally in ``[0, 1]``.

    Returns
    -------
    ndarray, shape (points, functions)
        ``values[i, j] = phi_j(x[i])``.

    Examples
    --------
    >>> c = np.array([[0.5], [0.5]])        # 0.5 + 0.5 * (2x - 1) = x
    >>> evaluate(c, [0.0, 0.25, 1.0]).ravel().tolist()
    [0.0, 0.25, 1.0]
    """
    return _cheb.chebval(2 * np.asarray(x, dtype=float) - 1, np.asarray(coefficients)).T


def derivative(coefficients, order=1):
    """Chebyshev coefficients of the ``order``-th derivative in ``x``.

    Parameters
    ----------
    coefficients : array_like, shape (terms, functions)
        Series in ``t = 2x - 1``.
    order : int, optional
        Derivative order, ``>= 0``. Default 1.

    Returns
    -------
    ndarray, shape (terms - order, functions)
        Coefficients of :math:`d^m \\phi / dx^m`. Since ``dt/dx = 2`` each
        derivative carries a factor 2 relative to differentiation in ``t``.
        For ``order = 0`` a copy of the input is returned.
    """
    c = np.asarray(coefficients)
    if order == 0:
        return c.copy()
    return _cheb.chebder(c, m=order, axis=0) * 2.0**order


def chebyshev_points(count):
    """Chebyshev points of the first kind mapped to [0, 1].

    Parameters
    ----------
    count : int
        Number of points.

    Returns
    -------
    ndarray, shape (count,)
        ``(1 - cos(pi (i + 1/2) / count)) / 2`` for ``i = 0 .. count-1``, in
        increasing order.
    """
    return (1 + _cheb.chebpts1(int(count))) / 2


def interpolate(functions, terms):
    """Chebyshev coefficients of functions by interpolation.

    Each function is sampled at ``terms`` Chebyshev points and the interpolating
    series of degree ``terms - 1`` is returned. For smooth functions the
    coefficients decay quickly; inspect the trailing coefficients to confirm the
    representation is resolved.

    Parameters
    ----------
    functions : callable or sequence of callable
        Each maps an array of ``x`` in [0, 1] to values of the same shape.
    terms : int
        Number of Chebyshev coefficients per function.

    Returns
    -------
    ndarray, shape (terms, len(functions))
        One column per function.

    Examples
    --------
    >>> c = interpolate(lambda x: np.exp(x), 20)
    >>> bool(abs(evaluate(c, [0.3])[0, 0] - np.exp(0.3)) < 1e-14)
    True
    """
    funcs = [functions] if callable(functions) else list(functions)
    columns = [_cheb.chebinterpolate(lambda t, f=f: f((np.asarray(t) + 1) / 2), int(terms) - 1) for f in funcs]
    return np.column_stack(columns)
