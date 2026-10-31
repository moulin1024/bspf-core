import numpy as np
import pytest
import scipy.sparse as sp
from bspf_core import hybrid_axis, element_rule, sturm_liouville_matrices, PoleCondensation, StaticCondensation


def lossy_element(length=7.0):
    rule = element_rule(hybrid_axis(), 0.0, length)
    z = rule.nodes
    k = (2 * np.pi * 50 / (1500 + 3 * z)) * (1 - 2e-3j)          # lossy, depth-varying medium
    rho = 1.0 + 0.02 * z
    return rule, sturm_liouville_matrices(rule, 1 / rho, 1 / rho, k * k / rho)


def dense_schur(K, M, lam):
    n = len(K); b, i = [0, n - 1], list(range(1, n - 1))
    A = K + lam * M
    return A[np.ix_(b, b)] - A[np.ix_(b, i)] @ np.linalg.solve(A[np.ix_(i, i)], A[np.ix_(i, b)])


def test_pole_schur_matches_dense_condensation():
    _, (K, M) = lossy_element()
    pc = PoleCondensation(K, M)
    lam = np.array([0.0, 0.03 + 1e-4j, 0.4 - 2e-3j, 5.0])
    S = pc.schur(lam)
    for q, l in enumerate(lam):
        ref = dense_schur(K, M, l)
        assert np.max(abs(S[q] - ref)) < 1e-10 * np.max(abs(ref))


def test_partial_fractions_reproduce_schur():
    _, (K, M) = lossy_element()
    pc = PoleCondensation(K, M)
    theta, r, A0, A1 = pc.partial_fractions()
    lam = np.array([0.02 + 3e-4j, 1.7])
    s = A0 + lam[:, None] * A1 - np.sum(r[None] / (theta[None, :, None] + lam[:, None, None]), axis=1)
    S = pc.schur(lam)
    assert np.allclose(s, np.stack([S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]], -1), rtol=1e-10, atol=1e-12)


def test_sampling_reproduces_interior_reconstruction():
    rule, (K, M) = lossy_element()
    pc = PoleCondensation(K, M)
    row = hybrid_axis().evaluate([0.37], 0)[0][0]
    lam, ends = 0.05 + 1e-3j, np.array([0.3 - 0.1j, -1.2 + 0.4j])
    n = len(K); i = list(range(1, n - 1)); A = K + lam * M
    interior = -np.linalg.solve(A[np.ix_(i, i)], A[np.ix_(i, [0, n - 1])] @ ends)
    full = np.r_[ends[0], interior, ends[1]]
    c, res, theta = pc.sampling(row)
    w = c - np.sum(res / (theta + lam)[:, None], axis=0)
    assert np.isclose(w @ ends, row @ full, rtol=1e-11)


def test_pole_condensation_rejects_complex_mass():
    _, (K, M) = lossy_element()
    with pytest.raises(ValueError):
        PoleCondensation(K, M * (1 + 1e-3j))


def test_static_condensation_is_exact():
    rng = np.random.default_rng(0)
    N, shared = 30, np.array([9, 10, 19, 20])
    private = [np.arange(0, 9), np.arange(11, 19), np.arange(21, 30)]
    A = np.zeros((N, N))
    for block in private:                                   # couple each block only to the shared set
        ids = np.r_[block, shared]
        X = rng.standard_normal((len(ids), len(ids)))
        A[np.ix_(ids, ids)] += X @ X.T + len(ids) * np.eye(len(ids))
    rhs = rng.standard_normal(N)
    solver = StaticCondensation(sp.csr_matrix(A), shared, private)
    assert np.allclose(solver.solve(rhs), np.linalg.solve(A, rhs), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        StaticCondensation(A, shared, private[:2])
