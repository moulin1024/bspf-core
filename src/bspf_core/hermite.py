"""Streamfunction bubbles and Hermite trace axes on compatible BSPF spaces.

A conforming streamfunction element needs ``C^1`` continuity between blocks,
so its natural unknowns at each interval end are the value and the slope.
:class:`HermiteAxis` provides exactly that: four cubic Hermite functions
carry the end values and slopes, and *bubbles* carry the interior. A bubble
:math:`\\psi` satisfies :math:`\\psi = \\psi' = 0` at both ends and lies in
the ``I`` space of a :class:`~bspf_core.compatible.CompatibleSpace`, so its
velocity :math:`\\psi'` is a zero-trace ``T`` function.

:class:`BubbleBasis` builds the bubbles in three steps, all fixed by a
quadrature rule:

1. orthonormalize ``T`` in the discrete :math:`L^2` inner product,
2. remove the rounding-level end values of ``T`` by a linear correction,
   applied consistently to ``I`` (its primitive) and ``T'``,
3. keep the combinations with :math:`\\psi(H) = \\int_0^H T = 0` (the null space
   of the end value of ``I``); :math:`\\psi(0) = 0` holds already.

This replaces ``ns_exponential_enrichment.Axis`` (without enrichment
dictionary) and the backward-step ``TraceAxis``, which the NACA C-grid
``Basis`` builds on.
"""
import numpy as np
import scipy.linalg as la
from numpy.polynomial import Polynomial

from . import compatible as _compatible
from . import quadrature as _quadrature


class BubbleBasis:
    """Streamfunction bubbles of a compatible space.

    Parameters
    ----------
    space : CompatibleSpace
    nodes, weights : array_like, shape (Q,)
        Quadrature rule on the reference interval [0, 1] defining the inner
        product of step 1.

    Attributes
    ----------
    size : int
        Number of bubbles, ``space.size - 1``.

    Notes
    -----
    With ``R`` from the QR factorization of ``sqrt(w) T(nodes)``, ``t0`` and
    ``t1`` the end values of ``T R^{-1}``, and ``Z`` an orthonormal basis of
    the null space of the corrected ``I(1) R^{-1}``, the bubbles at ``s`` are

    .. math::

        \\psi = \\big(I R^{-1} - I_0 - H(s - s^2/2)\\, t_0 - H (s^2/2)\\, t_1\\big) Z,\\quad
        \\psi' = \\big(T R^{-1} - (1-s)\\, t_0 - s\\, t_1\\big) Z,\\quad
        \\psi'' = \\big(T' R^{-1} + (t_0 - t_1)/H\\big) Z.
    """

    def __init__(self, space, nodes, weights):
        self.space = space
        nodes = np.asarray(nodes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        _, R = la.qr(np.sqrt(weights)[:, None] * space.values(nodes, "t"), mode="economic")
        self._transform = la.solve_triangular(R, np.eye(R.shape[0]))
        ends = np.array([0.0, 1.0])
        self._t0, self._t1 = space.values(ends, "t") @ self._transform
        I0, I1 = space.values(ends, "i") @ self._transform
        self._i0 = I0
        mu = I1 - I0 - space.height * (0.5 * self._t0 + 0.5 * self._t1)
        self._null = la.null_space(mu[None, :])
        self.size = self._null.shape[1]

    def evaluate(self, points):
        """Bubbles and their first two physical derivatives.

        Parameters
        ----------
        points : array_like, shape (M,)
            Reference coordinates in [0, 1].

        Returns
        -------
        list of ndarray
            ``[psi, psi', psi'']``, each of shape ``(M, size)``.
        """
        s = np.atleast_1d(np.asarray(points, dtype=float))[:, None]
        H, A, t0, t1 = self.space.height, self._transform, self._t0, self._t1
        psi = self.space.values(s[:, 0], "i") @ A - self._i0 - H * ((s - s * s / 2) * t0 + s * s / 2 * t1)
        velocity = self.space.values(s[:, 0], "t") @ A - (1 - s) * t0 - s * t1
        shear = self.space.values(s[:, 0], "t", 1) @ A + (t0 - t1) / H
        return [v @ self._null for v in (psi, velocity, shear)]


_HERMITE = [Polynomial([1, 0, -3, 2]), Polynomial([0, 1, -2, 1]), Polynomial([0, 0, 3, -2]), Polynomial([0, 0, -1, 1])]


class HermiteAxis:
    """Value/slope end unknowns plus streamfunction bubbles on an interval.

    Parameters
    ----------
    size : int
        Samples ``N`` of the underlying compatible line; the axis has
        ``4 + (N - 3) = N + 1`` functions.
    length : float
        Physical length ``L`` of the interval.
    extra : int, optional
        Extra Gauss points per panel. Default 0.
    space : CompatibleSpace, optional
        Underlying space on the reference interval (``height=1``); default
        ``compatible_space(size)``.

    Attributes
    ----------
    nodes, weights : ndarray
        Quadrature rule on [0, 1]: Gauss-Legendre with
        ``max(18, size + 6) + extra`` points on each knot panel of the line.
    values : list of ndarray
        ``evaluate(nodes)``.
    size : int
        Number of functions.

    Notes
    -----
    Function order: value at 0, slope at 0, value at 1, slope at 1 (slopes
    with respect to the physical coordinate), then the bubbles. Derivatives
    are physical: order ``k`` is divided by ``L**k``.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> axis = HermiteAxis(17, length=2.0)
    >>> V, dV, _ = axis.evaluate([0.0, 1.0])
    >>> bool(np.allclose(V, np.eye(axis.size)[[0, 2]], atol=1e-12))
    True
    >>> bool(np.allclose(dV, np.eye(axis.size)[[1, 3]], atol=1e-12))
    True
    """

    def __init__(self, size, length, extra=0, space=None):
        self.length = float(length)
        space = _compatible.compatible_space(size) if space is None else space
        panels = np.unique(np.asarray(space.line.knots))
        self.nodes, self.weights = _quadrature.composite_gauss(panels, max(18, int(size) + 6) + extra)
        self.bubbles = BubbleBasis(space, self.nodes, self.weights)
        self.values = self.evaluate(self.nodes)
        self.size = self.values[0].shape[1]

    def evaluate(self, points):
        """Functions and their first two physical derivatives at reference points.

        Returns
        -------
        list of ndarray
            Three arrays of shape ``(M, size)``.
        """
        z = np.atleast_1d(np.asarray(points, dtype=float))
        L = self.length
        bubbles = self.bubbles.evaluate(z)
        scale = np.array([1.0, L, 1.0, L])
        return [np.c_[np.column_stack([p.deriv(k)(z) for p in _HERMITE]) * scale, bubbles[k]] / L ** k
                for k in range(3)]
