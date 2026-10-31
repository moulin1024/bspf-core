"""Extended-precision factors, compatible spaces and Hermite axes (Navier-Stokes bases)."""
import warnings

import numpy as np
import pytest
from bspf_core import cardinal, jets
from bspf_core.compatible import compatible_space
from bspf_core.hermite import HermiteAxis
from conftest import import_from, load_reference

NS_CORE = "src/current_ns/core"


def ns_line(size):
    x = np.linspace(0.0, 1.0, size)
    line = cardinal.cardinal_line(x, degree=7, n_basis=10, order=5, jets=jets.polynomial_jets(x, 5, 16, 12),
                                  convention="closed", symmetrize=True)
    return line


def test_polynomial_jets_match_handoff_build_stencil():
    estimators = import_from(f"{NS_CORE}/original", "boundary_estimators")
    for size in (17, 49, 129):
        x = np.linspace(0.0, 1.0, size)
        ref = estimators.build_stencil(x, 5, "fixed16d12")[0]
        assert np.max(abs(jets.polynomial_jets(x, 5, 16, 12) - ref)) < 1e-14 * np.max(abs(ref))


def test_symmetrize_and_exact_end_rows():
    line = ns_line(33)
    P = np.asarray(line.projection)
    assert np.array_equal(P, P[::-1, ::-1])
    assert np.array_equal(P[0], np.eye(33)[0]) and np.array_equal(P[-1], np.eye(33)[-1])
    x = np.linspace(0.0, 1.0, 33)
    with pytest.raises(ValueError):
        cardinal.cardinal_line(x, degree=3, n_basis=8, order=2, knots=np.r_[[0.0] * 4, 0.2, 0.3, 0.6, 0.8, [1.0] * 4],
                               symmetrize=True)


def test_remainder_factors_converged_and_needed():
    line = ns_line(129)
    z = np.array([0.0, 0.0123, 0.4, 0.999, 1.0])
    quad, fine = cardinal.RemainderFactors(line, 34), cardinal.RemainderFactors(line, 50)
    for k in (0, 1, 2):
        a = cardinal.evaluation_matrix(line, z, k, quad)
        assert np.max(abs(a - cardinal.evaluation_matrix(line, z, k, fine))) < 1e-14 * np.max(abs(a))
    # float64 factors lose digits for this line (|P| ~ 5e7): the reason the class exists.
    a = cardinal.evaluation_matrix(line, z, 2, quad)
    assert np.max(abs(cardinal.evaluation_matrix(line, z, 2) - a)) > 1e-10 * np.max(abs(a))
    assert np.allclose(cardinal.primitive_matrix(line, z, quad)[-1], cardinal.integration_weights(line, factors=quad))


@pytest.mark.parametrize("size", [17, 49, 129])
def test_compatible_space_matches_handoff_compatible_line(size):
    line_space = import_from(NS_CORE, "line_space")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = line_space.CompatibleLine(size, 3.0)
    space = compatible_space(size, height=3.0)
    s = np.r_[np.linspace(0.0, 1.0, 41), 0.123456]
    for kind, order in (("t", 0), ("t", 1), ("t", 2), ("i", 0), ("i", 1)):
        expected = np.asarray(ref.basis(s, kind, order), float)
        assert np.max(abs(space.values(s, kind, order) - expected)) < 1e-13 * np.max(abs(expected)), (kind, order)


@pytest.mark.parametrize("size, length, extra", [(17, 2.0, 0), (49, 8.0, 2)])
def test_hermite_axis_matches_handoff_trace_axis(size, length, extra):
    solver = load_reference("benchmarks/backward_step/solver.py", "backward_step_solver")
    ref = solver.TraceAxis(size, length, extra)
    axis = HermiteAxis(size, length, extra)
    assert axis.size == ref.n
    assert np.allclose(axis.nodes, ref.q, rtol=0, atol=1e-15) and np.allclose(axis.weights, ref.w, rtol=0, atol=1e-16)
    z = np.array([0.0, 0.05, 0.3333, 0.5, 0.97, 1.0])
    for ours, theirs in ((axis.values, ref.V), (axis.evaluate(z), ref.evaluate(z))):
        for k in range(3):
            assert np.max(abs(ours[k] - theirs[k])) < 1e-13 * np.max(abs(theirs[k]))


def test_bubbles_have_zero_value_and_slope_and_orthonormal_slopes():
    axis = HermiteAxis(17, 1.0)
    psi, velocity, _ = axis.bubbles.evaluate([0.0, 1.0])
    assert np.max(abs(psi)) < 1e-13 and np.max(abs(velocity)) < 1e-13
    V = axis.bubbles.evaluate(axis.nodes)[1]
    assert np.allclose(V.T @ (axis.weights[:, None] * V), np.eye(axis.bubbles.size), atol=1e-12)
    # psi' is the derivative of psi: compare with the quadrature primitive.
    psi_q, vel_q, shear_q = axis.bubbles.evaluate(axis.nodes)
    assert np.allclose(axis.weights @ vel_q, 0, atol=1e-13)                     # psi(1) - psi(0) = 0
    assert np.allclose(axis.weights @ shear_q, 0, atol=1e-11)                   # psi'(1) - psi'(0) = 0
