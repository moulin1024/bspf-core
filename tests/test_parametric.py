"""ParametricChain against a dense global solve and against native SCOOTER."""
from pathlib import Path

import numpy as np
import pytest
from bspf_core import hybrid_axis, element_rule, sturm_liouville_matrices, PoleCondensation
from bspf_core.parametric import ParametricChain
from conftest import load_reference

AXIS = hybrid_axis()


def build_chain(edges, frequency=50.0, speed=lambda z: 1500 + 0.2 * z):
    elements, mats = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rule = element_rule(AXIS, lo, hi)
        k = 2 * np.pi * frequency / speed(rule.nodes) * (1 - 1e-3j)
        K, M = sturm_liouville_matrices(rule, np.ones_like(rule.nodes), np.ones_like(rule.nodes), k * k)
        elements.append(PoleCondensation(K, M)); mats.append((K, M))
    return elements, mats


def dense_green(mats, lam, source_node, receiver, alpha_bottom):
    """Assemble the full C0 global system (all element functions) and solve it."""
    n, E = AXIS.size, len(mats)
    size = E * (n - 1) + 1
    A = np.zeros((size, size), complex)
    for e, (K, M) in enumerate(mats):
        ids = np.r_[e * (n - 1), e * (n - 1) + 1 + np.arange(n - 2), (e + 1) * (n - 1)]
        A[np.ix_(ids, ids)] += K + lam * M
    A[-1, -1] += alpha_bottom
    A[0, :] = 0; A[:, 0] = 0; A[0, 0] = 1                                # Dirichlet top
    f = np.zeros(size, complex); f[source_node * (n - 1)] = 1.0
    u = np.linalg.solve(A, f)
    e, row = receiver
    ids = np.r_[e * (n - 1), e * (n - 1) + 1 + np.arange(n - 2), (e + 1) * (n - 1)]
    return row @ u[ids]


def test_chain_matches_dense_global_solve():
    edges = np.linspace(0, 100, 6)
    elements, mats = build_chain(edges)
    row = AXIS.evaluate([0.4], 0)[0][0]
    alpha = lambda lam: 0.3 * np.sqrt(lam + 0j)
    chain = ParametricChain(elements, [(2, 1.0)], [(3, row)], dirichlet=(True, False))
    lam = np.array([0.01 + 1e-4j, 0.04 + 2e-4j, 0.2 + 1e-3j])
    g = chain.green(lam, alpha_bottom=alpha, chunk=4)
    for q, l in enumerate(lam):
        ref = dense_green(mats, l, 2, (3, row), alpha(np.array([l]))[0])
        assert abs(g[0, 0, q] - ref) < 1e-9 * abs(ref)


def test_chain_reproduces_native_scooter_kernel():
    """bspf-core pole condensation + chain == native SCOOTER 0.30 kernel on a Pekeris case."""
    import sys
    from conftest import HANDOFF
    path = str(HANDOFF / "release/bspf-scooter/src")
    if not Path(path).exists():
        pytest.skip("bspf-scooter not available")
    sys.path.insert(0, path)
    from bspf_scooter.env_io import _read_env_text
    from bspf_scooter.layered import LayeredSpace
    from bspf_scooter.native_scooter import WavenumberKernel, native_wavenumbers, native_edges
    from bspf_core.gram import ElementRule
    text = ("'pekeris'\n50\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n 100.0 1500.0 /\n'A' 0.0\n"
            " 100.0 1800.0 0.0 1.8 0.5 /\n1400 1e9\n2.0\n1\n30.0 /\n3\n10.0 90.0 /\n")
    doc = _read_env_text(text, model="scooter", layered=True)
    k, atten = native_wavenumbers(doc, 50.0)
    h = k[::25] + 1j * atten
    space = LayeredSpace(doc.environment, edges=native_edges(doc.environment, 50.0, k[-1], doc.source_depths))
    kernel = WavenumberKernel(space, 50.0, doc.source_depths, doc.receiver_depths)
    ref = kernel.green(h)
    elements = []
    for lo, hi, (layer, z, w, B, D) in zip(space.edges[:-1], space.edges[1:], space.rules):
        _, rho, kz = layer.values(z, 50.0)
        K, M = sturm_liouville_matrices(ElementRule(lo, hi, z, w, (B, D)), 1 / rho, 1 / rho, kz * kz / rho)
        elements.append(PoleCondensation(K, M))
    sources = [(j // 3, weight) for j, weight in kernel.sources]       # trace slot 3*node -> node
    chain = ParametricChain(elements, sources, kernel.receivers, dirichlet=kernel._dirichlet)
    g = chain.green(h * h, alpha_top=lambda lam: kernel._end_terms(np.sqrt(lam))[0],
                    alpha_bottom=lambda lam: kernel._end_terms(np.sqrt(lam))[1], chunk=64)
    assert np.max(abs(g - ref)) < 1e-10 * np.max(abs(ref))
