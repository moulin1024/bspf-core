"""Batched tridiagonal solves with adjacent partial pivoting (JAX).

Condensing a chain of elements onto its nodes leaves a tridiagonal system per
parameter value. For wavenumber sweeps there are thousands of such systems of
the same size. :func:`solve` handles a whole batch at once.

Algorithm
---------
Gaussian elimination with *adjacent* row interchanges, the same algorithm as
LAPACK ``?gttrf``/``?gttrs``: at step ``i`` the remaining row ``i`` is swapped
with row ``i + 1`` when ``|a_i| < |l_i|``, which bounds the growth for
indefinite (Helmholtz-type) systems. Pivoting introduces one extra
super-diagonal in ``U``.

Elimination, forward substitution and back substitution are each a
:func:`jax.lax.scan` over the nodes, with the batch as the vector dimension.
Each step touches one row of every system, so no large array is rewritten
inside the loop; an earlier version that updated full arrays in place was
about 30 times slower.

Layout
------
Arrays are *node-major*: ``diag`` has shape ``(n, batch)``, ``lower`` and
``upper`` ``(n - 1, batch)``, ``rhs`` ``(n, batch, sources)``. Node-major
layout makes every scan step a contiguous slice.
"""
import jax
import jax.numpy as jnp


def solve(lower, diag, upper, rhs):
    """Solve a batch of tridiagonal systems.

    Parameters
    ----------
    lower : jax.Array, shape (n - 1, batch)
        Sub-diagonal; ``lower[i]`` is row ``i + 1``, column ``i``.
    diag : jax.Array, shape (n, batch)
        Main diagonal.
    upper : jax.Array, shape (n - 1, batch)
        Super-diagonal; ``upper[i]`` is row ``i``, column ``i + 1``.
    rhs : jax.Array, shape (n, batch, sources)
        Right-hand sides; several per system.

    Returns
    -------
    jax.Array, shape (n, batch, sources)
        Solutions.

    Notes
    -----
    Real or complex dtypes are supported. Singular systems produce
    non-finite values; no error is raised inside traced code.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> d = jnp.full((4, 1), 2.0); o = jnp.full((3, 1), -1.0)
    >>> x = solve(o, d, o, jnp.ones((4, 1, 1)))
    >>> [round(float(v), 12) for v in x[:, 0, 0]]
    [2.0, 3.0, 3.0, 2.0]
    """
    def eliminate(row, nxt):
        # row = (a, b): current diagonal and super-diagonal of remaining row i.
        a, b = row
        l, d, u = nxt
        swap = jnp.abs(a) < jnp.abs(l)
        p0, p1 = jnp.where(swap, l, a), jnp.where(swap, d, b)
        p2 = jnp.where(swap, u, 0)                       # second super-diagonal of U
        m = jnp.where(swap, a, l) / p0                   # multiplier
        remaining = (jnp.where(swap, b - m * d, d - m * b), jnp.where(swap, -m * u, u))
        return remaining, (p0, p1, p2, m, swap)

    up = jnp.concatenate((upper, jnp.zeros_like(upper[:1])))
    last, (p0, p1, p2, m, swap) = jax.lax.scan(eliminate, (diag[0], up[0]), (lower, diag[1:], up[1:]))

    def forward(b, step):
        m_, s_, nxt = step
        lead = jnp.where(s_[:, None], nxt, b)
        return jnp.where(s_[:, None], b, nxt) - m_[:, None] * lead, lead

    tail, y = jax.lax.scan(forward, rhs[0], (m, swap, rhs[1:]))
    y = jnp.concatenate((y, tail[None]))
    zero = jnp.zeros_like(p1[:1])
    U = (jnp.concatenate((p0, last[0][None])), jnp.concatenate((p1, zero)), jnp.concatenate((p2, zero)))

    def back(carry, step):
        x1, x2 = carry
        q0, q1, q2, yi = step
        x = (yi - q1[:, None] * x1 - q2[:, None] * x2) / q0[:, None]
        return (x, x1), x

    _, x = jax.lax.scan(back, (jnp.zeros_like(y[0]), jnp.zeros_like(y[0])), (*U, y), reverse=True)
    return x
