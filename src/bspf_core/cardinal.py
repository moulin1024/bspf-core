"""Cardinal BSPF line: endpoint-constrained spline plus Fourier remainder.

This is the construction that gives BSPF its name. Data are samples
``f_0 .. f_{N-1}`` on a uniform grid of a non-periodic interval. A clamped
B-spline :math:`s` (few functions) is fitted to the samples by weighted least
squares, subject to reproducing estimated endpoint derivatives of orders
``0 .. q-1``:

.. math::

    \\min_c\\ \\|B c - f\\|_W^2 + \\lambda \\|c\\|^2 \\quad\\text{subject to}\\quad
    C c = J f,

where ``B`` holds the spline basis at the grid, ``W`` the trapezoid weights,
``C`` the spline's end derivatives and ``J`` an endpoint-jet estimator
(:mod:`bspf_core.jets`). The remainder :math:`r = f - s` then has matching
derivatives at both ends, so its periodic extension is smooth and its
trigonometric interpolant converges fast. Derivatives, values between grid
points and integrals are those of :math:`s + \\text{trig}(r)`.

Every step is linear in ``f``. The fit is precomputed as a matrix ``P``
(``c = P f``); the *cardinal functions* of the line are the reconstructions of
unit samples, ``phi_j = recon(e_j)``.

Fourier conventions
-------------------
``"samples"`` (pybspf ``plan_1d``)
    The remainder's FFT uses all ``N`` samples; the period is ``N h``.
``"closed"`` (hand-off Navier-Stokes solvers)
    The FFT uses the first ``N - 1`` samples; the period is the physical length
    ``(N - 1) h``. This assumes the remainder vanishes at both ends (``q >= 1``).

For real data the real part of the trigonometric interpolant is used, so odd
derivatives of the Nyquist mode vanish; complex data keep the complex result.

Runtime and setup
-----------------
:func:`cardinal_line` builds a :class:`CardinalLine` on the host. It is a JAX
PyTree, so :func:`fit`, :func:`split` and :func:`derivatives` can be traced and
batched; trailing array dimensions are batches. :func:`evaluation_matrix`,
:func:`differentiation_matrix` and :func:`integration_weights` return dense
host matrices for Galerkin assembly and post-processing.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as la

from . import jets as _jets
from . import splines

_CONVENTIONS = ("samples", "closed")


@partial(jax.tree_util.register_dataclass,
         data_fields=["x", "knots", "projection", "basis", "omega"],
         meta_fields=["degree", "order", "convention", "regularization"])
@dataclass(frozen=True)
class CardinalLine:
    """Precomputed cardinal BSPF line (immutable JAX PyTree).

    Attributes
    ----------
    x : jax.Array, shape (N,)
        Uniform grid.
    knots : jax.Array
        Clamped spline knots.
    projection : jax.Array, shape (n_basis, N)
        Fit matrix ``P``: spline coefficients ``c = P f``.
    basis : tuple of jax.Array
        ``basis[k]`` has shape ``(N, n_basis)``: ``k``-th derivative of the
        spline basis at the grid, for ``k = 0 .. max_derivative``.
    omega : jax.Array
        Angular frequencies of the remainder's FFT (length ``N`` or ``N - 1``).
    degree : int
        Spline degree.
    order : int
        Number of constrained endpoint orders ``q``.
    convention : str
        ``"samples"`` or ``"closed"``.
    regularization : float
        The ``lambda`` of the fit.
    """

    x: jax.Array
    knots: jax.Array
    projection: jax.Array
    basis: tuple
    omega: jax.Array
    degree: int
    order: int
    convention: str
    regularization: float

    @property
    def size(self):
        """Number of samples ``N``."""
        return self.x.shape[0]

    @property
    def n_basis(self):
        """Number of spline functions."""
        return self.projection.shape[0]

    @property
    def spacing(self):
        """Grid spacing ``h``."""
        return float(self.x[1] - self.x[0])


def cardinal_line(x, *, degree=5, n_basis=None, knots=None, order=None, jets=None,
                  regularization=0.0, convention="samples", max_derivative=4, clustering=0.0,
                  symmetrize=False):
    """Build a :class:`CardinalLine` on a uniform grid.

    Parameters
    ----------
    x : array_like, shape (N,)
        Strictly increasing uniform grid.
    degree : int, optional
        Spline degree. Default 5.
    n_basis : int, optional
        Number of spline functions; default ``min(4 * (degree + 1), N)``.
        Ignored when ``knots`` is given.
    knots : array_like, optional
        Explicit clamped knot vector on ``[x[0], x[-1]]``.
    order : int, optional
        Constrained endpoint orders ``q``; default ``degree - 1``.
    jets : ndarray, shape (2 * order, N), optional
        Endpoint-jet estimator matrix. Default
        ``taylor_jets(x, order, points=degree, fit="interpolation")``, the
        pybspf ``plan_1d`` default.
    regularization : float, optional
        ``lambda >= 0`` in the fit. Default 0.
    convention : {"samples", "closed"}, optional
        Fourier convention (see the module docstring). Default ``"samples"``.
    max_derivative : int, optional
        Highest derivative tabulated for :func:`derivatives`. Default 4.
    clustering : float, optional
        Knot clustering passed to :func:`bspf_core.splines.clamped_knots`.
    symmetrize : bool, optional
        Average ``P`` with its mirror image, ``(P + P[::-1, ::-1]) / 2``. The
        exact ``P`` of a mirror-symmetric setup (symmetric knots, jets that
        are reflections of each other) has this symmetry; averaging removes
        the rounding that breaks it, as the hand-off "reflection-balanced"
        lines do. Default False.

    Returns
    -------
    CardinalLine

    Raises
    ------
    ValueError
        For a non-uniform grid, inconsistent sizes, an unknown convention, or a
        singular constrained fit.

    Notes
    -----
    ``P`` is the exact solution operator of the constrained problem in the
    module docstring, the same as the Karush-Kuhn-Tucker system of pybspf
    ``plan_1d`` (whose ``lam`` is ``regularization`` here), but it is computed
    by the null-space method of the hand-off ``constrained_map``: with row
    scaling ``S``, ``(S C)^T = Q R``, ``Z`` spans the null space of ``C`` and
    ``T`` is a right inverse (``C T = I``). Writing ``c = Z y + T J f``, ``y``
    solves the unconstrained least-squares problem

    .. math::

        \\min_y \\left\\| \\begin{bmatrix} W^{1/2} B Z \\\\ \\sqrt{\\lambda}\\, Z \\end{bmatrix} y
        - \\begin{bmatrix} W^{1/2} (f - B T J f) \\\\ -\\sqrt{\\lambda}\\, T J f \\end{bmatrix} \\right\\|,

    solved by a second QR factorization. The constraint rows contain
    derivatives up to order ``q - 1`` and scale like ``h^{1-q}``; a direct KKT
    solve then loses several digits in ``P``, whose gain is large, so the
    QR form is used.

    A clamped spline's end values are its first and last coefficients, so
    the first and last rows of ``P`` equal the jet rows for order 0 exactly;
    they are set exactly rather than left to rounding.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> from bspf_core.jets import taylor_jets
    >>> x = np.linspace(0.0, 1.0, 33)
    >>> line = cardinal_line(x, degree=7, order=6, jets=taylor_jets(x, 6, 7))
    >>> d1 = derivatives(line, np.exp(x), orders=(1,))[1]
    >>> bool(np.max(abs(np.asarray(d1) - np.exp(x))) < 1e-7)
    True
    """
    x = np.asarray(x, dtype=float)
    _, h = _jets._uniform_spacing(x)
    if convention not in _CONVENTIONS:
        raise ValueError(f"convention must be one of {_CONVENTIONS}")
    q = degree - 1 if order is None else int(order)
    if knots is None:
        n_basis = min(4 * (degree + 1), len(x)) if n_basis is None else int(n_basis)
        knots = splines.clamped_knots(x[0], x[-1], degree=degree, n_basis=n_basis, clustering=clustering)
    knots = np.asarray(knots, dtype=float)
    n_basis = len(knots) - degree - 1
    if n_basis < 2 * q or (regularization == 0 and n_basis > len(x)):
        raise ValueError("need n_basis >= 2*order and, without regularization, n_basis <= N")
    if convention == "closed" and q < 1:
        raise ValueError("the closed convention needs order >= 1 (remainder zero at the ends)")
    J = _jets.taylor_jets(x, q, degree, "interpolation") if jets is None else np.asarray(jets, dtype=float)
    if J.shape != (2 * q, len(x)):
        raise ValueError(f"jets must have shape {(2 * q, len(x))}")
    basis = [splines.basis_matrix(knots, degree, x, k) for k in range(max(max_derivative, q) + 1)]
    weights = np.full(len(x), h)
    weights[[0, -1]] *= 0.5                                   # trapezoid rule
    C = np.vstack([b[0] for b in basis[:q]] + [b[-1] for b in basis[:q]]) if q else np.zeros((0, n_basis))
    try:
        P = _constrained_projection(basis[0], C, J, weights, regularization)
    except (la.LinAlgError, ValueError) as exc:
        raise ValueError("singular constrained fit; use fewer basis functions or regularization > 0") from exc
    if symmetrize:
        mirror = J[:q, ::-1] * (-1.0) ** np.arange(q)[:, None]
        if not (np.allclose(knots + knots[::-1], x[0] + x[-1], rtol=0, atol=1e-13 * (x[-1] - x[0]))
                and np.allclose(J[q:], mirror, rtol=0, atol=1e-12 * max(1.0, np.max(abs(J))))):
            raise ValueError("symmetrize needs symmetric knots and mirror-symmetric jets")
        P = 0.5 * (P + P[::-1, ::-1])
    if q and knots[degree] == x[0] and knots[-degree - 1] == x[-1]:
        P[0], P[-1] = J[0], J[q]                               # clamped end values are exact
    count = len(x) if convention == "samples" else len(x) - 1
    omega = 2 * np.pi * np.fft.fftfreq(count, d=h)
    return CardinalLine(jnp.asarray(x), jnp.asarray(knots), jnp.asarray(P),
                        tuple(jnp.asarray(b) for b in basis[: max_derivative + 1]), jnp.asarray(omega),
                        int(degree), q, convention, float(regularization))


def _constrained_projection(B, C, J, weights, regularization):
    """Null-space QR solution operator of the constrained fit (see cardinal_line)."""
    N, nb = B.shape
    r = C.shape[0]
    if r:
        scale = 1.0 / np.max(abs(C), axis=1)
        Q, R = la.qr((scale[:, None] * C).T)
        if np.min(abs(np.diag(R[:r, :r]))) < 1e-13 * np.max(abs(np.diag(R[:r, :r]))):
            raise ValueError("dependent endpoint constraints")
        Z = Q[:, r:]
        T = Q[:, :r] @ la.solve_triangular(R[:r, :r].T, np.diag(scale), lower=True)
    else:
        Z, T = np.eye(nb), np.zeros((nb, 0))
    if Z.shape[1] == 0:                                        # n_basis == 2q: the jets fix the spline
        return T @ J
    root = np.sqrt(weights)
    design = root[:, None] * (B @ Z)
    if regularization:
        design = np.vstack([design, np.sqrt(regularization) * Z])
    Qd, Rd = la.qr(design, mode="economic")
    if np.min(abs(np.diag(Rd))) < 1e-13 * np.max(abs(np.diag(Rd))):
        raise ValueError("rank-deficient spline fit")
    solve = la.solve_triangular(Rd, Qd.T)                     # (nb - r, rows of design)
    data = solve[:, :N] * root                                 # samples -> y
    lift = -data @ (B @ T)
    if regularization:
        lift -= np.sqrt(regularization) * solve[:, N:] @ T
    return Z @ data + (T + Z @ lift) @ J


def _contract(matrix, data):
    return jnp.tensordot(matrix, data, axes=(1, 0))


def fit(line, f):
    """Spline coefficients of the constrained fit.

    Parameters
    ----------
    line : CardinalLine
    f : array_like, shape (N, ...)
        Samples; trailing dimensions are batches.

    Returns
    -------
    jax.Array, shape (n_basis, ...)
    """
    return _contract(line.projection, jnp.asarray(f))


def split(line, f):
    """Coefficients, spline values and remainder at the grid.

    Parameters
    ----------
    line : CardinalLine
    f : array_like, shape (N, ...)

    Returns
    -------
    coefficients : jax.Array, shape (n_basis, ...)
    spline : jax.Array, shape (N, ...)
    remainder : jax.Array, shape (N, ...)
        ``f - spline``; smooth when periodically extended.
    """
    f = jnp.asarray(f)
    c = fit(line, f)
    s = _contract(line.basis[0], c)
    return c, s, f - s


def _fourier_derivative(line, remainder, k):
    """``k``-th derivative of the remainder's trigonometric interpolant at the grid."""
    closed = line.convention == "closed"
    r = remainder[:-1] if closed else remainder
    w = line.omega.reshape((-1,) + (1,) * (r.ndim - 1))
    out = jnp.fft.ifft(jnp.fft.fft(r, axis=0) * (1j * w) ** k, axis=0)
    out = out if jnp.iscomplexobj(remainder) else out.real
    return jnp.concatenate([out, out[:1]], axis=0) if closed else out


def derivatives(line, f, orders=(1, 2)):
    """Derivatives of the BSPF reconstruction at the grid.

    Parameters
    ----------
    line : CardinalLine
    f : array_like, shape (N, ...)
        Samples.
    orders : tuple of int, optional
        Derivative orders, each ``<= max_derivative``. Static under ``jit``.
        Order 0 returns the reconstruction (equal to ``f`` at the grid).

    Returns
    -------
    dict
        ``{order: jax.Array of shape (N, ...)}``. One fit and one forward FFT
        are shared by all orders.
    """
    c, _, r = split(line, f)
    return {k: _contract(line.basis[k], c) + _fourier_derivative(line, r, k) for k in orders}


def _trig_rows(line, points, derivative=0, primitive=False):
    """Rows mapping samples to the trigonometric interpolant (real-data semantics).

    Returns shape ``(len(points), N)``; for the closed convention the last
    column is zero because the last sample is not part of the FFT. With
    ``primitive=True`` the rows give :math:`\\int_{x_0}^{z}` of the interpolant.
    """
    x = np.asarray(line.x)
    omega = np.asarray(line.omega)
    count = len(omega)
    u = np.asarray(points)[:, None] - x[0]
    if primitive:
        safe = np.where(omega == 0, 1.0, omega)
        modes = np.where(omega == 0, u + 0j, np.expm1(1j * u * omega) / (1j * safe))
    else:
        modes = np.exp(1j * u * omega) * (1j * omega) ** derivative
    dft = np.exp(-2j * np.pi * np.outer(np.arange(count), np.arange(count)) / count)
    rows = ((modes @ dft) / count).real
    return np.pad(rows, ((0, 0), (0, line.size - count)))


def _remainder_factor(line, points, derivative=0, primitive=False, factors=None):
    """``G = spline basis - trig(spline basis samples)`` at points, shape (len(points), n_basis)."""
    if factors is not None:
        return factors.evaluate(points, derivative, primitive)
    knots = np.asarray(line.knots)
    spline = (splines.primitive_matrix(knots, line.degree, points) if primitive
              else splines.basis_matrix(knots, line.degree, points, derivative))
    return spline - _trig_rows(line, points, derivative, primitive) @ np.asarray(line.basis[0])


def evaluation_matrix(line, points, derivative=0, factors=None):
    """Cardinal functions (or derivatives) at arbitrary points, for real data.

    Parameters
    ----------
    line : CardinalLine
    points : array_like, shape (M,)
        Points in ``[x[0], x[-1]]``.
    derivative : int, optional
        Derivative order. Default 0.
    factors : RemainderFactors, optional
        Extended-precision remainder factors of ``line``. Without them the
        factors are formed in float64 (see Notes).

    Returns
    -------
    ndarray, shape (M, N)
        Row ``i`` holds ``phi_j^{(derivative)}(points[i])`` for every sample
        ``j``; applied to real samples it gives the BSPF reconstruction.

    Notes
    -----
    The reconstruction is written as

    .. math::

        \\operatorname{recon}(f)(z) = \\operatorname{trig}(f)(z) + G(z)\\, P f,
        \\qquad G(z) = B(z) - \\operatorname{trig}(B)(z),

    the trigonometric interpolant of the samples plus the interpolation error
    of each spline function, weighted by the fit. ``G`` is small while ``P``
    has a large gain for high constrained orders, so float64 rounding in
    ``G`` (absolute error about ``1e-16``) is amplified by ``|P|``. For such
    lines pass :class:`RemainderFactors`, which forms ``G`` in extended
    precision and rounds it once, as the hand-off quad-precision factors do.

    This is how a cardinal BSPF line becomes a Galerkin basis: evaluate the
    cardinal functions at quadrature points. Complex data should be split
    into real and imaginary parts.
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    G = _remainder_factor(line, points, derivative, factors=factors)
    return _trig_rows(line, points, derivative) + G @ np.asarray(line.projection)


def primitive_matrix(line, points, factors=None):
    """Primitives :math:`\\int_{x_0}^{z} \\phi_j` of the cardinal functions (real data).

    Parameters
    ----------
    line : CardinalLine
    points : array_like, shape (M,)
        Upper limits ``z`` in ``[x[0], x[-1]]``.
    factors : RemainderFactors, optional
        As in :func:`evaluation_matrix`.

    Returns
    -------
    ndarray, shape (M, N)
        ``primitive_matrix(line, z) @ f`` is the integral of the reconstruction
        of ``f`` from the left end to ``z``.
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    G = _remainder_factor(line, points, primitive=True, factors=factors)
    return _trig_rows(line, points, primitive=True) + G @ np.asarray(line.projection)


def differentiation_matrix(line, order=1):
    """Dense ``N x N`` matrix of the grid derivative of a given order (real data).

    Equivalent to :func:`derivatives` applied to the identity; useful for
    small dense operators. Prefer :func:`derivatives` for large ``N``.
    """
    return np.asarray(derivatives(line, jnp.eye(line.size), orders=(order,))[order])


def integration_weights(line, a=None, b=None, factors=None):
    """Quadrature weights of the BSPF reconstruction over ``[a, b]`` (real data).

    Parameters
    ----------
    line : CardinalLine
    a, b : float, optional
        Bounds inside ``[x[0], x[-1]]``; default the whole interval.
    factors : RemainderFactors, optional
        As in :func:`evaluation_matrix`.

    Returns
    -------
    ndarray, shape (N,)
        ``w @ f`` is the exact integral of the reconstruction of ``f``; the
        difference of two rows of :func:`primitive_matrix`.
    """
    x = np.asarray(line.x)
    a = x[0] if a is None else float(a)
    b = x[-1] if b is None else float(b)
    ends = primitive_matrix(line, [a, b], factors)
    return ends[1] - ends[0]


class RemainderFactors:
    """Spline interpolation errors ``G = B - trig(B)`` in extended precision.

    Setup computes the Fourier coefficients of the sampled spline basis with
    ``mpmath`` at ``digits`` significant digits; :meth:`evaluate` forms ``G``,
    its derivatives or its primitive at points and rounds the result to
    float64 once. Pass the object as ``factors=`` to :func:`evaluation_matrix`,
    :func:`primitive_matrix` and :func:`integration_weights`.

    Parameters
    ----------
    line : CardinalLine
    digits : int, optional
        Working precision in decimal digits. The default 34 matches IEEE
        binary128 (``__float128``), the precision of the hand-off factors.

    Notes
    -----
    Replaces the hand-off ``quad_basis.c``/``QuadFactors`` (GCC
    ``libquadmath``), which is fixed to degree 7, ten spline functions and
    knots at thirds. This version takes any clamped knots, degree and grid,
    at the cost of pure-Python arithmetic: setup is ``O(count^2 n_basis)`` and
    each point ``O(count n_basis)`` multiprecision operations. Results are
    cached per point set.

    Examples
    --------
    >>> import jax; jax.config.update("jax_enable_x64", True)
    >>> x = np.linspace(0.0, 1.0, 17)
    >>> line = cardinal_line(x, degree=3, n_basis=6, order=2, convention="closed")
    >>> factors = RemainderFactors(line)
    >>> z = np.array([0.1, 0.55])
    >>> bool(np.allclose(evaluation_matrix(line, z, factors=factors), evaluation_matrix(line, z), atol=1e-12))
    True
    """

    def __init__(self, line, digits=34):
        import mpmath
        self._mp = mpmath.mp.clone()
        self._mp.dps = int(digits)
        mp = self._mp
        self.line = line
        self.digits = int(digits)
        x = np.asarray(line.x)
        self._count = len(line.omega)
        self._x0 = mp.mpf(float(x[0]))
        self._h = (mp.mpf(float(x[-1])) - self._x0) / (line.size - 1)
        self._period = self._count * self._h
        self._knots = [mp.mpf(float(t)) for t in np.asarray(line.knots)]
        count, nb = self._count, line.n_basis
        samples = [splines._mp_basis(mp, self._knots, line.degree, self._x0 + n * self._h) for n in range(count)]
        roots = [mp.expjpi(-mp.mpf(2 * j) / count) for j in range(count)]
        self._coefficients = [[sum(samples[n][m] * roots[(k * n) % count] for n in range(count)) / count
                               for m in range(nb)] for k in range(count // 2 + 1)]
        self._cache = {}

    def evaluate(self, points, derivative=0, primitive=False):
        """``G`` (or its derivative, or its primitive from ``x[0]``) at points.

        Parameters
        ----------
        points : array_like, shape (M,)
        derivative : int, optional
        primitive : bool, optional
            Return :math:`\\int_{x_0}^{z} G` instead of a derivative.

        Returns
        -------
        ndarray, shape (M, n_basis)
        """
        points = np.atleast_1d(np.asarray(points, dtype=float))
        key = points.tobytes()
        orders = max(2, int(derivative))
        if key not in self._cache or len(self._cache[key]) - 1 < orders + 1:
            rows = [self._point(float(z), orders) for z in points]
            self._cache[key] = [np.array([r[j] for r in rows]) for j in range(orders + 2)]
        bundle = self._cache[key]
        return bundle[-1] if primitive else bundle[int(derivative)]

    def _point(self, z, orders):
        """``[G, G', ..., G^(orders), int G]`` at one point, from one pass over the modes."""
        mp = self._mp
        count, K = self._count, self._count // 2
        u = mp.mpf(z) - self._x0
        x = self._x0 + u
        degree = self.line.degree
        splines_ = [splines._mp_basis(mp, self._knots, degree, x, d) for d in range(orders + 1)]
        splines_.append(splines._mp_basis(mp, self._knots, degree, x, primitive=True))
        c0 = [c.real for c in self._coefficients[0]]
        trig = [list(c0)] + [[mp.zero] * len(c0) for _ in range(orders)] + [[c * u for c in c0]]
        step = 2 * mp.pi / self._period
        rotation = mp.expj(step * u)
        phase = mp.one
        for k in range(1, K + 1):
            omega = step * k
            phase *= rotation                                         # e^{i k step u}; 34 digits absorb the drift
            nyquist = count % 2 == 0 and k == K                       # real part, counted once
            weight = 1 if nyquist else 2
            powers = [weight * omega ** d for d in range(orders + 1)]
            for m, c in enumerate(self._coefficients[k]):
                c = mp.mpc(c.real, 0) if nyquist else c
                cp = c * phase
                parts = (cp.real, -cp.imag, -cp.real, cp.imag)        # Re(cp * i^d), d mod 4
                for d in range(orders + 1):
                    trig[d][m] += powers[d] * parts[d % 4]
                trig[-1][m] += weight * (cp.imag - c.imag) / omega    # Re(c (e^{i omega u} - 1) / (i omega))
        return [[float(a - b) for a, b in zip(sp, tr)] for sp, tr in zip(splines_, trig)]
