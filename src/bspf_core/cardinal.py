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
                  regularization=0.0, convention="samples", max_derivative=4, clustering=0.0):
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


def _host_dft_rows(line, points, derivative):
    """Trigonometric interpolation rows: points -> values from remainder samples."""
    x = np.asarray(line.x)
    omega = np.asarray(line.omega)
    count = len(omega)
    modes = np.exp(1j * (np.asarray(points)[:, None] - x[0]) * omega[None, :]) * (1j * omega[None, :]) ** derivative
    dft = np.exp(-2j * np.pi * np.outer(np.arange(count), np.arange(count)) / count)
    return (modes @ dft) / count                       # (points, count)


def _remainder_map(line):
    """Matrix mapping samples to remainder samples used by the FFT."""
    R = np.eye(line.size) - np.asarray(line.basis[0]) @ np.asarray(line.projection)
    return R[:-1] if line.convention == "closed" else R


def evaluation_matrix(line, points, derivative=0):
    """Cardinal functions (or derivatives) at arbitrary points, for real data.

    Parameters
    ----------
    line : CardinalLine
    points : array_like, shape (M,)
        Points in ``[x[0], x[-1]]``.
    derivative : int, optional
        Derivative order. Default 0.

    Returns
    -------
    ndarray, shape (M, N)
        Row ``i`` holds ``phi_j^{(derivative)}(points[i])`` for every sample
        ``j``; applied to real samples it gives the BSPF reconstruction.

    Notes
    -----
    This is how a cardinal BSPF line becomes a Galerkin basis: evaluate the
    cardinal functions at quadrature points. Complex data should be split
    into real and imaginary parts.
    """
    points = np.atleast_1d(np.asarray(points, dtype=float))
    spline = splines.basis_matrix(np.asarray(line.knots), line.degree, points, derivative) @ np.asarray(line.projection)
    return spline + (_host_dft_rows(line, points, derivative) @ _remainder_map(line)).real


def differentiation_matrix(line, order=1):
    """Dense ``N x N`` matrix of the grid derivative of a given order (real data).

    Equivalent to :func:`derivatives` applied to the identity; useful for
    small dense operators. Prefer :func:`derivatives` for large ``N``.
    """
    return np.asarray(derivatives(line, jnp.eye(line.size), orders=(order,))[order])


def integration_weights(line, a=None, b=None):
    """Quadrature weights of the BSPF reconstruction over ``[a, b]`` (real data).

    Parameters
    ----------
    line : CardinalLine
    a, b : float, optional
        Bounds inside ``[x[0], x[-1]]``; default the whole interval.

    Returns
    -------
    ndarray, shape (N,)
        ``w @ f`` is the exact integral of the reconstruction of ``f``.

    Notes
    -----
    The spline part uses the exact spline primitive; the remainder uses
    :math:`\\int_a^b e^{i\\omega (x - x_0)} dx = L\\, e^{i\\omega (m - x_0)}
    \\operatorname{sinc}(\\omega L / 2\\pi)` with ``L = b - a`` and midpoint ``m``.
    """
    x = np.asarray(line.x)
    a = x[0] if a is None else float(a)
    b = x[-1] if b is None else float(b)
    ends = splines.primitive_matrix(np.asarray(line.knots), line.degree, [a, b])
    spline = (ends[1] - ends[0]) @ np.asarray(line.projection)
    omega = np.asarray(line.omega)
    count = len(omega)
    L = b - a
    kernel = L * np.exp(1j * omega * ((a + b) / 2 - x[0])) * np.sinc(omega * L / (2 * np.pi))
    dft = np.exp(-2j * np.pi * np.outer(np.arange(count), np.arange(count)) / count)
    return spline + ((kernel @ dft) / count @ _remainder_map(line)).real
