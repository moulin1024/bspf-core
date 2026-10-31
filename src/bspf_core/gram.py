"""Element quadrature and weighted Gram matrices.

For an element ``[z0, z1]`` of length ``h`` and an :class:`~bspf_core.axis.Axis`
with reference functions :math:`\\phi_j(x)`, the physical basis is
:math:`\\phi_j((z - z0)/h)` and its ``k``-th derivative is
:math:`h^{-k} \\phi_j^{(k)}`. :func:`element_rule` returns the quadrature nodes,
weights and these physical values in one :class:`ElementRule`.

Gram matrices :math:`G_{ij} = \\int w(z)\\, f_i(z)\\, g_j(z)\\, dz` are then plain
weighted products (:func:`gram`). :func:`sturm_liouville_matrices` assembles the
element matrices of the scalar problem

.. math::

    -(a u')' + (\\lambda b - c) u = f,

namely the stiffness-like :math:`K = \\int a\\,\\phi'\\phi' - \\int c\\,\\phi\\phi` and the
parameter mass :math:`M = \\int b\\,\\phi\\phi`, so that the element operator is
:math:`K + \\lambda M`. For acoustics, ``a = b = 1/rho`` and
``c = (omega / c_p)^2 / rho``; ``lambda`` is the squared horizontal wavenumber.
"""
from dataclasses import dataclass

import numpy as np

from .quadrature import composite_gauss


@dataclass(frozen=True)
class ElementRule:
    """Quadrature and basis values on one physical element.

    Attributes
    ----------
    lower, upper : float
        Element end points ``z0 < z1``.
    nodes : ndarray, shape (points,)
        Physical quadrature nodes.
    weights : ndarray, shape (points,)
        Physical weights (summing to ``z1 - z0``).
    values : tuple of ndarray
        ``values[k]`` has shape ``(points, size)``: the ``k``-th physical
        derivative of every basis function at the nodes.
    """

    lower: float
    upper: float
    nodes: np.ndarray
    weights: np.ndarray
    values: tuple

    @property
    def length(self):
        """Element length ``z1 - z0``."""
        return self.upper - self.lower


def element_rule(axis, lower, upper, knots=(), order=None, derivatives=1):
    """Quadrature and physical basis values on ``[lower, upper]``.

    Parameters
    ----------
    axis : Axis
        Reference basis.
    lower, upper : float
        Element end points, ``lower < upper``.
    knots : array_like, optional
        Interior points where coefficients are not smooth (for example
        sound-speed profile knots). The rule is split there; points outside
        ``(lower, upper)`` are ignored.
    order : int, optional
        Gauss points per sub-interval; default ``max(2 * size + 4, 8)``,
        enough to integrate products of two basis functions with smooth
        coefficients accurately.
    derivatives : int, optional
        Highest physical derivative to tabulate (0 to 3). Default 1.

    Returns
    -------
    ElementRule

    Examples
    --------
    >>> from bspf_core import hybrid_axis
    >>> rule = element_rule(hybrid_axis(), 0.0, 2.0)
    >>> float(round(rule.weights.sum(), 14))
    2.0
    """
    if not lower < upper:
        raise ValueError("need lower < upper")
    inner = np.asarray(knots, dtype=float)
    inner = inner[(inner > lower) & (inner < upper)]
    order = max(2 * axis.size + 4, 8) if order is None else int(order)
    z, w = composite_gauss(np.r_[lower, inner, upper], order)
    h = upper - lower
    raw = axis.evaluate((z - lower) / h, derivatives)
    return ElementRule(float(lower), float(upper), z, w, tuple(v / h**k for k, v in enumerate(raw)))


def gram(left, right, weights):
    """Weighted Gram matrix :math:`G = L^T \\operatorname{diag}(w) R`.

    Parameters
    ----------
    left : ndarray, shape (points, m)
        Values of the test functions at the quadrature nodes.
    right : ndarray, shape (points, n)
        Values of the trial functions.
    weights : ndarray, shape (points,)
        Quadrature weights times any coefficient (real or complex).

    Returns
    -------
    ndarray, shape (m, n)
    """
    return np.asarray(left).T @ (np.asarray(weights)[:, None] * np.asarray(right))


def sturm_liouville_matrices(rule, a, b, c):
    """Element matrices ``K`` and ``M`` of :math:`-(a u')' + (\\lambda b - c) u`.

    Parameters
    ----------
    rule : ElementRule
        From :func:`element_rule` with ``derivatives >= 1``.
    a, b, c : array_like, shape (points,)
        Coefficient values at ``rule.nodes``; may be complex. ``b`` must be
        real and positive for the pole-form condensation of
        :class:`bspf_core.condensation.PoleCondensation`.

    Returns
    -------
    K : ndarray, shape (size, size)
        :math:`\\int a \\phi_i' \\phi_j' - \\int c \\phi_i \\phi_j`.
    M : ndarray, shape (size, size)
        :math:`\\int b \\phi_i \\phi_j`.

    Notes
    -----
    The weak form uses the same (unconjugated) test and trial functions, so
    ``K`` and ``M`` are complex symmetric when the coefficients are complex.
    """
    B, D = rule.values[0], rule.values[1]
    w = rule.weights
    K = gram(D, D, w * np.asarray(a)) - gram(B, B, w * np.asarray(c))
    M = gram(B, B, w * np.asarray(b))
    return K, M
