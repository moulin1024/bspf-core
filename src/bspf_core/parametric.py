"""Chains of condensed elements solved for many parameter values.

A one-dimensional domain split into elements ``e = 0 .. E-1`` with
pole-condensed interiors (:class:`~bspf_core.condensation.PoleCondensation`)
couples only neighbouring nodes. For each parameter value ``lambda`` the
nodal system is tridiagonal:

* element ``e`` adds its 2x2 condensed matrix to nodes ``e`` and ``e + 1``;
* the ends receive optional Robin terms ``alpha_top(lambda)`` and
  ``alpha_bottom(lambda)`` on the diagonal, or are Dirichlet (value zero);
* point sources act on nodes with given weights.

:class:`ParametricChain` precomputes the partial-fraction constants once and
evaluates Green values at receiver points for a whole batch of ``lambda`` in
one compiled JAX call. The work per ``lambda`` is ``O(E n + E S + R n)`` for
``n`` functions per element, ``S`` sources and ``R`` receivers. This is the
engine of the native SCOOTER wavenumber integration.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import tridiagonal


class ParametricChain:
    """Prepared solver for a chain of pole-condensed elements.

    Parameters
    ----------
    elements : sequence of PoleCondensation
        One per element, in order; element ``e`` joins nodes ``e`` and ``e + 1``.
        All must have the same number of functions.
    sources : sequence of (int, complex)
        ``(node, weight)`` pairs; each defines one right-hand side.
    receivers : sequence of (int, ndarray)
        ``(element, row)`` pairs; ``row`` holds the element's basis values at
        the receiver point (trace coordinates).
    dirichlet : tuple of bool, optional
        ``(top, bottom)``: impose zero value at node 0 and/or node E.
        Default ``(False, False)``.

    Notes
    -----
    Requires ``jax_enable_x64``. Batches are processed in chunks of fixed
    length (padded) so the compiled kernel is reused across calls.

    Examples
    --------
    See ``tests/test_parametric.py`` for a complete example that checks the
    chain against a dense solve.
    """

    def __init__(self, elements, sources, receivers, dirichlet=(False, False)):
        sizes = {e.size for e in elements}
        if len(sizes) != 1:
            raise ValueError("all elements must have the same number of functions")
        parts = [e.partial_fractions() for e in elements]
        self._theta = np.array([p[0] for p in parts])
        self._residues = np.array([p[1] for p in parts])
        self._A0 = np.array([p[2] for p in parts])
        self._A1 = np.array([p[3] for p in parts])
        E = len(elements)
        rhs = np.zeros((E + 1, len(sources)), complex)
        for s, (node, weight) in enumerate(sources):
            rhs[node, s] = weight
        for node, fixed in zip((0, E), dirichlet):
            if fixed:
                rhs[node] = 0
        self._rhs = rhs
        sampling = [elements[e].sampling(np.asarray(row)) for e, row in receivers]
        self._rec = np.array([e for e, _ in receivers], dtype=int)
        self._Wc = np.array([s[0] for s in sampling])
        self._Wr = np.array([s[1] for s in sampling])
        self._Wt = np.array([s[2] for s in sampling])
        self.dirichlet = tuple(bool(d) for d in dirichlet)
        self.elements = E

    def green(self, lam, alpha_top=None, alpha_bottom=None, chunk=2048):
        """Green values at the receivers for every parameter value.

        Parameters
        ----------
        lam : array_like, shape (k,)
            Parameter values.
        alpha_top, alpha_bottom : callable, optional
            Map an array of ``lam`` to the Robin diagonal term at node 0 or
            node ``E``. Ignored at a Dirichlet end. Default: no term.
        chunk : int, optional
            Batch length per compiled call. Default 2048.

        Returns
        -------
        ndarray, shape (sources, receivers, k)
        """
        lam = np.asarray(lam, dtype=complex)
        out = np.empty((self._rhs.shape[1], len(self._rec), len(lam)), complex)
        args = tuple(jnp.asarray(a) for a in (self._theta, self._residues, self._A0, self._A1, self._rhs,
                                                self._rec, self._Wc, self._Wr, self._Wt))
        for start in range(0, len(lam), chunk):
            part = lam[start:start + chunk]
            n = len(part)
            if n < chunk:                        # pad to a fixed shape: one compilation
                part = np.r_[part, np.full(chunk - n, part[-1])]
            ends = [f(part) if f is not None else np.zeros(chunk, complex) for f in (alpha_top, alpha_bottom)]
            g = _chain_green(jnp.asarray(part), jnp.asarray(ends[0]), jnp.asarray(ends[1]), *args, dirichlet=self.dirichlet)
            out[..., start:start + n] = np.asarray(g)[..., :n]
        return out


@partial(jax.jit, static_argnames=("dirichlet",))
def _chain_green(lam, top, bottom, theta, residues, A0, A1, rhs, rec, Wc, Wr, Wt, *, dirichlet):
    """Compiled kernel of :meth:`ParametricChain.green` (node-major layout)."""
    E, K = theta.shape[0], lam.shape[0]

    def pole(j, acc):
        return acc + residues[:, j, :][:, None] / (theta[:, j][:, None, None] + lam[None, :, None])

    s = A0[:, None] + lam[None, :, None] * A1[:, None] - jax.lax.fori_loop(
        0, theta.shape[1], pole, jnp.zeros((E, K, 3), lam.dtype))
    d = jnp.zeros((E + 1, K), lam.dtype).at[:E].add(s[..., 0]).at[1:].add(s[..., 2])
    d = d.at[0].add(top).at[E].add(bottom)
    lower = upper = s[..., 1]
    if dirichlet[0]:
        d, upper = d.at[0].set(1), upper.at[0].set(0)
    if dirichlet[1]:
        d, lower = d.at[E].set(1), lower.at[E - 1].set(0)
    nodes = tridiagonal.solve(lower, d, upper, jnp.broadcast_to(rhs[:, None, :], (E + 1, K, rhs.shape[1])))

    def weight(j, acc):
        return acc - Wr[:, j][:, None] / (Wt[:, j][:, None, None] + lam[None, :, None])

    w = jax.lax.fori_loop(0, Wt.shape[1], weight, jnp.broadcast_to(Wc[:, None], (rec.shape[0], K, 2)).astype(lam.dtype))
    return jnp.einsum("rka,rkas->srk", w, jnp.stack((nodes[rec], nodes[rec + 1]), axis=2))
