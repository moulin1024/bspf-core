"""Coordinate systems for the span of a function family.

A raw family (see :mod:`bspf_core.families`) spans the right space but is a
poor basis. This module produces three standard bases of the same span:

``orthonormal``
    L2(0, 1)-orthonormal functions, by repeated weighted QR.
``trace``
    Two affine lifts ``1 - x`` and ``x`` followed by an L2-orthonormal basis of
    the functions that vanish at both ends. Only the lifts are non-zero at the
    ends, so continuity between elements and Dirichlet conditions concern the
    first and last coefficient only, and static condensation can eliminate the
    interior functions element by element.
``nodal``
    Cardinal (Lagrange-type) functions with ``phi_j(x_i) = delta_ij`` at given
    nodes, used by collocation-style and nodal Galerkin solvers.

All three return Chebyshev coefficient matrices with one column per basis
function. The QR, null-space and solve steps are NumPy/SciPy calls; the value
of the module is the fixed, tested sequence and its conventions, which the
existing axes reproduce exactly (see the parity tests).
"""
import numpy as np
import scipy.linalg as la
from numpy.polynomial.legendre import leggauss

from . import chebyshev


def _gram_rule(size):
    """Gauss-Legendre rule on [-1, 1] used for orthonormalization.

    ``max(80, 4 * size)`` points, weights halved so they sum to one (the
    length of [0, 1]). This matches every existing implementation.
    """
    t, w = leggauss(max(80, 4 * int(size)))
    return t, w / 2


def orthonormalize(raw, passes=3):
    """L2(0, 1)-orthonormal coordinates for the span of ``raw``.

    Columns are first normalised in the coefficient 2-norm, then the weighted
    values at Gauss points are QR-factorised and the triangular factor is
    divided out. Repeating the step (re-orthogonalisation) removes the loss of
    orthogonality caused by a badly conditioned raw family.

    Parameters
    ----------
    raw : ndarray, shape (terms, size)
        Chebyshev coefficients of a linearly independent family.
    passes : int, optional
        Number of QR passes, ``>= 1``. Default 3, as in all existing axes.

    Returns
    -------
    coefficients : ndarray, shape (terms, size)
        Orthonormal basis of the same span.
    raw_condition : float
        2-norm condition number of the first triangular factor; a measure of
        how ill conditioned the raw family was.
    """
    c = np.asarray(raw, dtype=float) / np.linalg.norm(raw, axis=0)
    t, w = _gram_rule(c.shape[1])
    raw_condition = None
    for _ in range(int(passes)):
        _, r = la.qr(np.sqrt(w[:, None]) * np.polynomial.chebyshev.chebval(t, c).T, mode="economic")
        if raw_condition is None:
            raw_condition = float(np.linalg.cond(r))
        c = c @ la.solve_triangular(r, np.eye(c.shape[1]))
    return c, raw_condition


def trace_coordinates(orthonormal, tolerance=2e-11):
    """Endpoint-lift and zero-trace interior coordinates.

    The result has the lifts ``phi_0 = 1 - x`` and ``phi_{n-1} = x`` and, in
    between, an orthonormal basis of the subspace that vanishes at ``x = 0``
    and ``x = 1`` (the null space of the endpoint evaluation of the
    orthonormal basis). Interior functions are corrected with the lifts to
    remove endpoint rounding.

    Parameters
    ----------
    orthonormal : ndarray, shape (terms, size)
        Orthonormal coefficients from :func:`orthonormalize`; the span must
        contain the affine functions.
    tolerance : float, optional
        Maximum accepted deviation of the endpoint values from the identity
        pattern. Default ``2e-11``.

    Returns
    -------
    coefficients : ndarray, shape (terms, size)
        Columns ``[1 - x, interior_1, ..., interior_{n-2}, x]``.
    trace_defect : float
        Largest deviation of the endpoint values from the exact pattern.

    Raises
    ------
    ValueError
        If the trace defect exceeds ``tolerance``.
    """
    c = np.asarray(orthonormal)
    zero_trace = la.null_space(np.polynomial.chebyshev.chebval(np.array([-1.0, 1.0]), c).T)
    result = np.zeros_like(c)
    result[0, 0], result[1, 0] = 0.5, -0.5            # 1 - x in t = 2x - 1
    result[0, -1], result[1, -1] = 0.5, 0.5           # x
    result[:, 1:-1] = c @ zero_trace
    end = np.polynomial.chebyshev.chebval(np.array([-1.0, 1.0]), result).T[:, 1:-1]
    result[:, 1:-1] -= result[:, [0, -1]] @ end
    trace = chebyshev.evaluate(result, [0.0, 1.0])
    expected = np.zeros_like(trace)
    expected[0, 0] = expected[1, -1] = 1
    defect = float(np.max(abs(trace - expected)))
    if defect > tolerance:
        raise ValueError(f"unresolved trace coordinates (defect {defect:.2e})")
    return result, defect


def nodal_coordinates(orthonormal, nodes, tolerance=1e-10, tail_tolerance=1e-14):
    """Cardinal coordinates: ``phi_j(nodes[i]) = delta_ij``.

    Parameters
    ----------
    orthonormal : ndarray, shape (terms, size)
        Orthonormal coefficients from :func:`orthonormalize`.
    nodes : array_like, shape (size,)
        Distinct nodes in [0, 1], unisolvent for the span.
    tolerance : float, optional
        Maximum accepted deviation from the identity at the nodes.
    tail_tolerance : float, optional
        Maximum accepted magnitude of the last eight Chebyshev coefficients,
        a check that the representation is resolved.

    Returns
    -------
    coefficients : ndarray, shape (terms, size)
    nodal_condition : float
        Condition number of the matrix of orthonormal values at the nodes.
    cardinal_defect : float

    Raises
    ------
    ValueError
        If the cardinal defect or the tail exceeds its tolerance.
    """
    c = np.asarray(orthonormal)
    values = chebyshev.evaluate(c, nodes)
    condition = float(np.linalg.cond(values))
    c = c @ la.solve(values, np.eye(c.shape[1]))
    defect = float(np.max(abs(chebyshev.evaluate(c, nodes) - np.eye(c.shape[1]))))
    if defect > tolerance or np.max(abs(c[-8:])) > tail_tolerance:
        raise ValueError(f"unresolved or unstable nodal basis (defect {defect:.2e})")
    return c, condition, defect


def lobatto_nodes(size):
    """Gauss-Legendre-Lobatto nodes on [0, 1].

    The endpoints and the roots of :math:`P'_{size-1}`, mapped from [-1, 1].

    Parameters
    ----------
    size : int
        Number of nodes, ``>= 2``.

    Returns
    -------
    ndarray, shape (size,)
        Increasing nodes, including 0 and 1.
    """
    inner = np.polynomial.legendre.Legendre.basis(int(size) - 1).deriv().roots()
    return np.r_[0.0, (inner + 1) / 2, 1.0]
