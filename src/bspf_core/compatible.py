"""Derivative-compatible BSPF spaces: zero-trace functions and their primitives.

Streamfunction formulations need two spaces related by differentiation. On
an interval of physical length ``H`` (reference coordinate ``s`` in [0, 1],
``y = H s``):

``T`` (kind ``"t"``)
    The cardinal functions of a closed cardinal BSPF line at the interior
    samples ``1 .. N-2``. They vanish at both ends, so ``T`` holds velocity-like
    fields with zero trace.
``I`` (kind ``"i"``)
    Their primitives from the left end, :math:`I_j(y) = \\int_0^y T_j`. Then
    :math:`I' = T` exactly: a streamfunction in ``I`` has its velocity in ``T``.

Derivatives are physical (with respect to ``y``). ``I`` functions vanish at
``y = 0`` but not, in general, at ``y = H``; :mod:`bspf_core.hermite` removes
that last trace to build streamfunction bubbles.

The line uses a reflection-symmetrized fit (``symmetrize=True``) and
extended-precision remainder factors
(:class:`~bspf_core.cardinal.RemainderFactors`): the fits used here constrain
five endpoint orders and their gain grows like ``N^4``, so float64 factors
would lose up to six digits at ``N = 129``.

This replaces the hand-off ``line_space.CompatibleLine`` (its ``basis``
method). The hand-off class also carries circulant-plus-low-rank mass
operators and dealiased reconstruction, which the current Navier-Stokes
block solvers do not use; they are not part of this module.
"""
from dataclasses import dataclass

import numpy as np

from . import cardinal as _cardinal
from . import jets as _jets


@dataclass(frozen=True)
class CompatibleSpace:
    """The pair ``T``/``I`` on one interval (see the module docstring).

    Attributes
    ----------
    line : CardinalLine
        Closed, symmetrized cardinal line on the reference grid ``[0, 1]``.
    factors : RemainderFactors
        Extended-precision remainder factors of ``line``.
    height : float
        Physical length ``H``.
    """

    line: _cardinal.CardinalLine
    factors: _cardinal.RemainderFactors
    height: float

    @property
    def size(self):
        """Number of functions, ``N - 2``."""
        return self.line.size - 2

    def values(self, points, kind="t", order=0):
        """Basis functions or physical derivatives at reference points.

        Parameters
        ----------
        points : array_like, shape (M,)
            Reference coordinates ``s`` in [0, 1].
        kind : {"t", "i"}, optional
            Zero-trace functions or their primitives. Default ``"t"``.
        order : int, optional
            Derivative order with respect to ``y = H s``. Default 0.

        Returns
        -------
        ndarray, shape (M, N - 2)
        """
        if kind not in ("t", "i"):
            raise ValueError("kind must be 't' or 'i'")
        points = np.atleast_1d(np.asarray(points, dtype=float))
        if kind == "i":
            if order == 0:
                return self.height * _cardinal.primitive_matrix(self.line, points, self.factors)[:, 1:-1]
            return self.values(points, "t", order - 1)
        matrix = _cardinal.evaluation_matrix(self.line, points, order, self.factors)
        return matrix[:, 1:-1] / self.height ** order


def compatible_space(size, height=1.0, *, degree=7, n_basis=10, order=5, jets=None, digits=34):
    """Build a :class:`CompatibleSpace` on ``size`` uniform samples.

    Parameters
    ----------
    size : int
        Number of samples ``N`` of the underlying line (``N - 2`` functions).
    height : float, optional
        Physical length. Default 1.
    degree, n_basis, order : int, optional
        Spline degree, number of spline functions and constrained endpoint
        orders. Defaults (7, 10, 5) are the hand-off Navier-Stokes lines.
    jets : ndarray, shape (2 * order, N), optional
        Endpoint-jet estimator; default
        ``polynomial_jets(x, order, points=16, degree=12)`` (hand-off
        ``"fixed16d12"``).
    digits : int, optional
        Precision of the remainder factors. Default 34 (binary128).

    Returns
    -------
    CompatibleSpace

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> space = compatible_space(17, height=2.0)
    >>> s = np.linspace(0.0, 1.0, 5)
    >>> T, I = space.values(s, "t"), space.values(s, "i")
    >>> bool(np.allclose(T[[0, -1]], 0, atol=1e-12)), bool(np.allclose(I[0], 0, atol=1e-14))
    (True, True)
    >>> bool(np.allclose(space.values(s, "i", 1), T))      # I' = T
    True
    """
    x = np.linspace(0.0, 1.0, int(size))
    J = _jets.polynomial_jets(x, order, 16, 12) if jets is None else jets
    line = _cardinal.cardinal_line(x, degree=degree, n_basis=n_basis, order=order, jets=J,
                                   convention="closed", symmetrize=True)
    return CompatibleSpace(line, _cardinal.RemainderFactors(line, digits), float(height))
