"""Tensor-product evaluation and fast diagonalization."""
import jax.numpy as jnp
import numpy as np
from bspf_core import tensor


def spd(rng, n):
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def test_evaluate_adjoint_and_apply_axes_agree():
    rng = np.random.default_rng(3)
    X, Y, C = rng.standard_normal((7, 4)), rng.standard_normal((6, 5)), rng.standard_normal((4, 5))
    assert np.allclose(tensor.evaluate(X, Y, C), (np.kron(X, Y) @ C.ravel()).reshape(7, 6))
    assert np.allclose(tensor.apply_axes([X, Y], C), tensor.evaluate(X, Y, C))
    F = rng.standard_normal((7, 6))
    assert np.allclose(tensor.adjoint(X, Y, F).ravel(), np.kron(X, Y).T @ F.ravel())
    batch = rng.standard_normal((4, 5, 3))
    out = tensor.apply_axes([X, None], jnp.asarray(batch))
    assert isinstance(out, jnp.ndarray) and out.shape == (7, 5, 3)


def test_fast_diagonalization_matches_dense_solve():
    rng = np.random.default_rng(4)
    Kx, Mx, Ky, My = spd(rng, 6), spd(rng, 6), spd(rng, 5), spd(rng, 5)
    fd = tensor.FastDiagonalization(Kx, Mx, Ky, My, sigma=-0.3)
    A = np.kron(Kx, My) + np.kron(Mx, Ky) - 0.3 * np.kron(Mx, My)
    F = rng.standard_normal((6, 5))
    assert np.allclose(fd.solve(F).ravel(), np.linalg.solve(A, F.ravel()), atol=1e-12)
    assert np.allclose(fd.solve(jnp.asarray(F)), fd.solve(F), atol=1e-14)
