"""Raw function families on [0, 1], as Chebyshev coefficient matrices.

A *family* is a set of functions whose span defines an approximation space.
Families here are not yet good bases: they are returned in a raw form and are
turned into well-conditioned coordinates by :mod:`bspf_core.coordinates`.

Hybrid (polynomial + Fourier) span
----------------------------------
The hybrid BSPF element space of degree ``p`` with ``m`` Fourier pairs of
period ``L`` (in units of the element length) is

.. math::

    V = P_p \\oplus \\operatorname{span}\\{\\cos(\\omega_j (x - 1/2)),\\,
        \\sin(\\omega_j (x - 1/2))\\}_{j=1}^{m}, \\qquad \\omega_j = 2\\pi j / L,

with dimension ``p + 1 + 2m``. With ``t = 2x - 1`` the Fourier functions are
``cos(omega t / 2)`` and ``sin(omega t / 2)``, whose Chebyshev series follow
from the Jacobi-Anger expansion

.. math::

    \\cos(z t) = J_0(z) + 2 \\sum_{k \\ge 1,\\ k\\ \\mathrm{even}} (-1)^{k/2} J_k(z) T_k(t),
    \\qquad
    \\sin(z t) = 2 \\sum_{k\\ \\mathrm{odd}} (-1)^{(k-1)/2} J_k(z) T_k(t).

Terms with ``k <= p`` lie in :math:`P_p` already, so only the *tail*
``k > p`` is stored. This leaves the span unchanged and avoids subtracting
two nearly equal columns, which is what made early hybrid bases ill
conditioned. :math:`J_k` is :func:`scipy.special.jv`.
"""
import numpy as np
from scipy.special import jv

from . import chebyshev


def default_terms(size):
    """Default Chebyshev length for a family of ``size`` functions.

    Returns ``max(128, 4 * size)``, the value used by all existing BSPF axes.
    It resolves Fourier pairs of moderate frequency to rounding.
    """
    return max(128, 4 * int(size))


def polynomial_block(degree, terms):
    """Polynomials of degree ``<= degree`` as Chebyshev unit vectors.

    Parameters
    ----------
    degree : int
        Polynomial degree ``p >= 0``.
    terms : int
        Length of the Chebyshev representation, ``>= degree + 1``.

    Returns
    -------
    ndarray, shape (terms, degree + 1)
        Column ``k`` is :math:`T_k(2x - 1)`.
    """
    if degree < 0 or terms < degree + 1:
        raise ValueError("need degree >= 0 and terms >= degree + 1")
    c = np.zeros((int(terms), int(degree) + 1))
    c[: degree + 1, : degree + 1] = np.eye(degree + 1)
    return c


def fourier_tails(modes, period, degree, terms):
    """Fourier pairs on [0, 1] with their degree-``<= degree`` part removed.

    Parameters
    ----------
    modes : int
        Number of pairs ``m >= 1``; frequencies ``omega_j = 2 pi j / period``.
    period : float
        Fourier period in units of the interval length. ``1`` gives one full
        period per element (acoustics default); ``1.5`` is used by the
        curved-domain Navier-Stokes solvers.
    degree : int
        Degree ``p`` of the accompanying polynomial block; coefficients of
        :math:`T_0, \\dots, T_p` are omitted (see the module docstring).
    terms : int
        Length of the Chebyshev representation.

    Returns
    -------
    ndarray, shape (terms, 2 * modes)
        The first ``modes`` columns are the cosine tails, the last ``modes``
        the sine tails, in order of increasing frequency.

    Raises
    ------
    ValueError
        If ``modes < 1`` or ``period`` is not positive and finite.
    """
    if modes < 1 or not np.isfinite(period) or period <= 0:
        raise ValueError("need modes >= 1 and a positive finite period")
    c = np.zeros((int(terms), 2 * int(modes)))
    k = np.arange(degree + 1, int(terms))
    even, odd = k[k % 2 == 0], k[k % 2 == 1]
    for j in range(int(modes)):
        z = np.pi * (j + 1) / period            # omega_j / 2
        c[even, j] = 2 * (-1.0) ** (even // 2) * jv(even, z)
        c[odd, modes + j] = 2 * (-1.0) ** ((odd - 1) // 2) * jv(odd, z)
    return c


def hybrid_family(degree, modes, period=1.0, terms=None):
    """Raw hybrid family: polynomial block followed by Fourier tails.

    Parameters
    ----------
    degree : int
        Polynomial degree ``p``.
    modes : int
        Number of Fourier pairs ``m``.
    period : float, optional
        Fourier period in element lengths. Default 1.
    terms : int, optional
        Chebyshev length; default :func:`default_terms` of ``p + 1 + 2m``.

    Returns
    -------
    ndarray, shape (terms, p + 1 + 2m)
        Columns ``[T_0 .. T_p, cos tails, sin tails]``.
    """
    size = int(degree) + 1 + 2 * int(modes)
    terms = default_terms(size) if terms is None else int(terms)
    return np.hstack([polynomial_block(degree, terms), fourier_tails(modes, period, degree, terms)])


def polynomial_family(size, terms=None):
    """Raw polynomial (spectral-element) family of dimension ``size``.

    Parameters
    ----------
    size : int
        Number of functions; the span is :math:`P_{size-1}`.
    terms : int, optional
        Chebyshev length; default :func:`default_terms`.

    Returns
    -------
    ndarray, shape (terms, size)
    """
    terms = default_terms(size) if terms is None else int(terms)
    return polynomial_block(int(size) - 1, terms)


def sampled_family(functions, terms):
    """Raw family from callables, e.g. enrichment functions.

    Parameters
    ----------
    functions : sequence of callable
        Functions of ``x`` in [0, 1]. They should be smooth on [0, 1]; their
        Chebyshev interpolants are used.
    terms : int
        Chebyshev length.

    Returns
    -------
    ndarray, shape (terms, len(functions))

    Notes
    -----
    Combine with another family by column concatenation, then apply a
    coordinate system; dependent directions are removed there. Check that the
    trailing coefficients are negligible before relying on the result.
    """
    return chebyshev.interpolate(functions, terms)
