"""One-dimensional bases: the :class:`Axis` object and its factories.

An :class:`Axis` is an immutable basis on the reference interval [0, 1] with
its derivatives, stored as Chebyshev coefficients. Physical elements
``[z0, z1]`` use it through the affine map ``x = (z - z0) / (z1 - z0)``; the
element layer (:mod:`bspf_core.gram`) applies the chain rule.

The factories cover the spaces in use today:

* :func:`hybrid_axis` - polynomial block plus Fourier pairs (the BSPF hybrid
  element). ``coordinates="trace"`` reproduces the acoustics
  ``BoundaryAxis("bspf", ...)``; ``coordinates="nodal"`` with
  ``period=1.5`` reproduces the vorticity solvers' ``HybridAxis``.
* :func:`sem_axis` - the polynomial (spectral-element) control in the same
  coordinates, for like-for-like comparisons.
* :func:`axis_from_family` - any raw family, including enrichments.
"""
from dataclasses import dataclass, field

import numpy as np

from . import chebyshev, coordinates as _coords, families

_SYSTEMS = ("trace", "nodal", "orthonormal")


@dataclass(frozen=True)
class Axis:
    """Immutable one-dimensional basis on [0, 1].

    Parameters
    ----------
    coefficients : ndarray, shape (terms, n)
        Chebyshev coefficients (variable ``t = 2x - 1``) of the ``n`` basis
        functions (``n`` is :attr:`size`), one per column.
    coordinates : str
        ``"trace"``, ``"nodal"`` or ``"orthonormal"`` (see
        :mod:`bspf_core.coordinates`).
    nodes : ndarray or None
        The cardinal nodes for nodal coordinates, else ``None``.
    metadata : dict
        Construction parameters and diagnostics (condition numbers, defects).

    Attributes
    ----------
    derivatives : tuple of ndarray
        Coefficients of the 0th to 3rd derivatives in ``x``; computed once at
        construction.

    Notes
    -----
    For ``coordinates="trace"`` column 0 is the only function non-zero at
    ``x = 0`` and column ``size - 1`` the only one non-zero at ``x = 1``.
    Element boundary unknowns are therefore ``[0, size - 1]`` and interior
    unknowns ``1 .. size - 2``; see :attr:`boundary` and :attr:`interior`.
    """

    coefficients: np.ndarray
    coordinates: str
    nodes: np.ndarray = None
    metadata: dict = field(default_factory=dict)
    derivatives: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.coordinates not in _SYSTEMS:
            raise ValueError(f"coordinates must be one of {_SYSTEMS}")
        c = np.asarray(self.coefficients, dtype=float)
        object.__setattr__(self, "coefficients", c)
        object.__setattr__(self, "derivatives", tuple(chebyshev.derivative(c, k) for k in range(4)))

    @property
    def size(self):
        """Number of basis functions."""
        return self.coefficients.shape[1]

    @property
    def boundary(self):
        """Indices of the two endpoint (lift) functions in trace coordinates."""
        return np.array([0, self.size - 1])

    @property
    def interior(self):
        """Indices of the zero-trace interior functions in trace coordinates."""
        return np.arange(1, self.size - 1)

    def evaluate(self, x, order=2):
        """Values and derivatives at points of [0, 1].

        Parameters
        ----------
        x : array_like, shape (points,)
            Reference coordinates.
        order : int, optional
            Highest derivative returned, 0 to 3. Default 2.

        Returns
        -------
        list of ndarray
            ``order + 1`` arrays of shape ``(points, size)``; entry ``k`` holds
            the ``k``-th derivative with respect to ``x``. Divide by
            ``h**k`` for an element of length ``h``.

        Examples
        --------
        >>> axis = hybrid_axis(degree=9, modes=2)
        >>> B, D = axis.evaluate([0.0, 1.0], order=1)
        >>> B.shape
        (2, 14)
        """
        if order not in (0, 1, 2, 3):
            raise ValueError("derivative order must be 0, 1, 2 or 3")
        return [chebyshev.evaluate(c, x) for c in self.derivatives[: order + 1]]


def axis_from_family(raw, coordinates="trace", nodes=None, **metadata):
    """Build an :class:`Axis` from a raw family.

    Parameters
    ----------
    raw : ndarray, shape (terms, n)
        Raw family from :mod:`bspf_core.families` (possibly several families
        concatenated column-wise).
    coordinates : {"trace", "nodal", "orthonormal"}, optional
        Coordinate system; default ``"trace"``.
    nodes : array_like, optional
        Cardinal nodes, required for ``"nodal"``.
    **metadata
        Extra entries stored in :attr:`Axis.metadata`.

    Returns
    -------
    Axis
    """
    c, raw_condition = _coords.orthonormalize(raw)
    meta = dict(metadata, raw_condition=raw_condition)
    if coordinates == "orthonormal":
        return Axis(c, coordinates, None, meta)
    if coordinates == "trace":
        c, defect = _coords.trace_coordinates(c)
        return Axis(c, coordinates, None, dict(meta, trace_defect=defect))
    if coordinates == "nodal":
        if nodes is None:
            raise ValueError("nodal coordinates need nodes")
        c, condition, defect = _coords.nodal_coordinates(c, nodes)
        return Axis(c, coordinates, np.asarray(nodes, float), dict(meta, nodal_condition=condition, cardinal_defect=defect))
    raise ValueError(f"coordinates must be one of {_SYSTEMS}")


def hybrid_axis(degree=9, modes=2, period=1.0, coordinates="trace", nodes=None):
    """BSPF hybrid element axis: :math:`P_p` plus ``modes`` Fourier pairs.

    Parameters
    ----------
    degree : int, optional
        Polynomial degree ``p``. Default 9.
    modes : int, optional
        Number of Fourier pairs ``m >= 1``. Default 2.
    period : float, optional
        Fourier period in element lengths. Default 1.
    coordinates : {"trace", "nodal", "orthonormal"}, optional
        Default ``"trace"``.
    nodes : array_like, optional
        Nodes for ``"nodal"``; default Gauss-Legendre-Lobatto nodes
        (:func:`bspf_core.coordinates.lobatto_nodes`).

    Returns
    -------
    Axis
        ``p + 1 + 2m`` functions.

    Notes
    -----
    Equivalences with existing implementations (verified to rounding in the
    test suite):

    ==============================================  ==========================================
    Existing                                        bspf-core
    ==============================================  ==========================================
    ``BoundaryAxis("bspf", n, p)`` (acoustics)      ``hybrid_axis(p, m, 1.0, "trace")``
    ``HybridAxis(modes=m, period=T, degree=p)``     ``hybrid_axis(p, m, T, "nodal")``
    ==============================================  ==========================================

    The space is the "independent hybrid" BSPF variant: Fourier coefficients
    are independent unknowns. It is not the cardinal BSPF construction (a
    spline fitted to endpoint jets plus an FFT remainder).

    Examples
    --------
    >>> axis = hybrid_axis(degree=9, modes=2)
    >>> axis.size, axis.coordinates
    (14, 'trace')
    """
    raw = families.hybrid_family(degree, modes, period)
    if coordinates == "nodal" and nodes is None:
        nodes = _coords.lobatto_nodes(raw.shape[1])
    return axis_from_family(raw, coordinates, nodes, family="hybrid", degree=int(degree), modes=int(modes), period=float(period))


def sem_axis(size, coordinates="trace", nodes=None):
    """Polynomial (spectral-element) axis of dimension ``size``.

    The span is :math:`P_{size-1}`; in ``"nodal"`` coordinates with the default
    Lobatto nodes it is the classical GLL spectral element. Using the same
    coordinate machinery as :func:`hybrid_axis` makes BSPF/SEM comparisons
    differ only in the span.

    Parameters
    ----------
    size : int
        Number of functions, ``>= 2``.
    coordinates : {"trace", "nodal", "orthonormal"}, optional
    nodes : array_like, optional

    Returns
    -------
    Axis
    """
    raw = families.polynomial_family(size)
    if coordinates == "nodal" and nodes is None:
        nodes = _coords.lobatto_nodes(size)
    return axis_from_family(raw, coordinates, nodes, family="sem", degree=int(size) - 1)
