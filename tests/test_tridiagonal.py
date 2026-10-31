import numpy as np
import jax.numpy as jnp
from bspf_core import tridiagonal


def test_batched_pivoted_solve_matches_dense():
    rng = np.random.default_rng(1)
    n, batch, sources = 25, 7, 3
    lower = rng.standard_normal((n - 1, batch)) + 1j * rng.standard_normal((n - 1, batch))
    upper = rng.standard_normal((n - 1, batch)) + 1j * rng.standard_normal((n - 1, batch))
    diag = 0.05 * rng.standard_normal((n, batch))           # small diagonal forces row swaps
    rhs = rng.standard_normal((n, batch, sources)) + 0j
    x = np.asarray(tridiagonal.solve(*(jnp.asarray(a) for a in (lower, diag + 0j, upper, rhs))))
    for k in range(batch):
        A = np.diag(diag[:, k]) + np.diag(lower[:, k], -1) + np.diag(upper[:, k], 1)
        ref = np.linalg.solve(A, rhs[:, k, :])
        assert np.max(abs(x[:, k, :] - ref)) < 1e-10 * np.max(abs(ref))
