"""Splines, endpoint jets, the cardinal BSPF line, and parity with pybspf and the hand-off."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bspf_core import cardinal, jets, splines
from conftest import PYBSPF, import_from


def smooth(x):
    """Non-periodic test function and its first two derivatives."""
    f = np.exp(np.sin(2 * x))
    d1 = 2 * np.cos(2 * x) * f
    d2 = (4 * np.cos(2 * x) ** 2 - 4 * np.sin(2 * x)) * f
    return f, d1, d2


# --- splines -----------------------------------------------------------------

def test_clamped_basis_partition_of_unity_and_primitive():
    t = splines.clamped_knots(-1.0, 2.0, degree=5, n_basis=14, clustering=1.5)
    x = np.linspace(-1, 2, 101)
    assert np.allclose(splines.basis_matrix(t, 5, x).sum(axis=1), 1, atol=1e-14)
    ends = splines.primitive_matrix(t, 5, [-1.0, 2.0])
    assert np.isclose((ends[1] - ends[0]).sum(), 3.0, atol=1e-13)   # integral of 1
    with pytest.raises(ValueError):
        splines.basis_matrix(t, 5, [2.5])


# --- jets ----------------------------------------------------------------------

@pytest.mark.parametrize("fit", ["interpolation", "least_squares"])
def test_taylor_jets_exact_on_polynomials(fit):
    x = np.linspace(0.5, 2.0, 21)
    order, points = 4, (6 if fit == "interpolation" else 4)
    J = jets.taylor_jets(x, order, points, fit)
    poly = np.polynomial.Polynomial([0.3, -1.0, 2.0, 0.5])            # degree 3
    exact = [poly.deriv(k)(x[0]) if k else poly(x[0]) for k in range(order)]
    exact += [poly.deriv(k)(x[-1]) if k else poly(x[-1]) for k in range(order)]
    assert np.allclose(J @ poly(x), exact, rtol=1e-9, atol=1e-9)


def test_taylor_fits_coincide_on_square_window():
    x = np.linspace(0, 1, 30)
    assert np.allclose(jets.taylor_jets(x, 5, 5), jets.taylor_jets(x, 5, 5, "least_squares"), atol=1e-8)


def test_chebyshev_jets_match_pybspf():
    endpoints = import_from(PYBSPF, "pybspf.endpoints")
    x = np.linspace(0.0, 1.0, 40)
    ours = jets.chebyshev_jets(x, 4, 12, 6)
    left, right = np.asarray(endpoints.chebyshev_boundary_blocks(
        jnp.asarray(x), order=4, points=12, modes=6, alpha=1e-12, penalty_power=4))
    assert np.allclose(ours[:4, :12], left, rtol=1e-12, atol=1e-9)
    assert np.allclose(ours[4:, -12:], right, rtol=1e-12, atol=1e-9)


# --- cardinal line ---------------------------------------------------------------

@pytest.mark.parametrize("convention", ["samples", "closed"])
def test_cardinal_accuracy_and_dense_helpers(convention):
    x = np.linspace(0.0, 1.5, 65)
    line = cardinal.cardinal_line(x, degree=7, order=6, jets=jets.taylor_jets(x, 6, 7),
                                  convention=convention)
    f, d1, d2 = smooth(x)
    d = cardinal.derivatives(line, f, orders=(0, 1, 2))
    assert np.max(abs(np.asarray(d[0]) - f)) < 1e-13                  # interpolates the samples
    # Endpoint errors are set by the 7-point jet stencil (truncation), not the line.
    assert np.max(abs(np.asarray(d[1]) - d1)) < 1e-6
    assert np.max(abs(np.asarray(d[2]) - d2)) < 2e-4
    D = cardinal.differentiation_matrix(line, 1)
    assert np.allclose(D @ f, d[1], atol=1e-11)
    points = np.array([0.0, 0.123, 0.77, 1.5])
    fp, d1p, _ = smooth(points)
    assert np.max(abs(cardinal.evaluation_matrix(line, points) @ f - fp)) < 1e-9
    assert np.max(abs(cardinal.evaluation_matrix(line, points, 1) @ f - d1p)) < 1e-6
    assert np.allclose(cardinal.evaluation_matrix(line, x), np.eye(len(x)), atol=1e-11)
    fine = np.linspace(0.2, 1.1, 200001)
    exact = np.trapezoid(smooth(fine)[0], fine)                         # trapezoid error ~ 1e-12
    assert abs(cardinal.integration_weights(line, 0.2, 1.1) @ f - exact) < 1e-9


def test_cardinal_jit_vmap_and_complex_batches():
    x = np.linspace(0, 1, 48)
    line = cardinal.cardinal_line(x)
    f, d1, _ = smooth(x)
    F = np.stack([f, 2 * f], axis=1)
    out = jax.jit(lambda l, g: cardinal.derivatives(l, g, orders=(1,))[1])(line, F)
    assert np.allclose(out[:, 1], 2 * out[:, 0])
    vm = jax.vmap(lambda g: cardinal.derivatives(line, g, orders=(1,))[1], in_axes=1, out_axes=1)(F)
    assert np.allclose(vm, out, atol=1e-12)
    z = cardinal.derivatives(line, f + 1j * f, orders=(1,))[1]
    assert np.allclose(z.real, out[:, 0]) and np.allclose(z.imag, out[:, 0])


def test_regularization_and_validation():
    x = np.linspace(0, 1, 20)
    line = cardinal.cardinal_line(x, degree=3, n_basis=30, order=2, regularization=1e-8)
    assert line.n_basis == 30                                          # n_basis > N allowed with lambda > 0
    with pytest.raises(ValueError):
        cardinal.cardinal_line(x, degree=3, n_basis=30, order=2)
    with pytest.raises(ValueError):
        cardinal.cardinal_line(x, convention="periodic")
    with pytest.raises(ValueError):
        cardinal.cardinal_line(x ** 2)


@pytest.mark.parametrize("size", [64, 65])
@pytest.mark.parametrize("method", ["finite_difference", "chebyshev"])
@pytest.mark.parametrize("lam", [0.0, 1e-4])
def test_parity_with_pybspf_plan_1d(size, method, lam):
    import_from(PYBSPF, "pybspf")
    from pybspf import calculus, operators, plans
    x = np.linspace(-0.3, 1.2, size)
    f = smooth(x)[0]
    plan = plans.plan_1d(jnp.asarray(x), endpoint_method=method, lam=lam)
    J = None if method == "finite_difference" else jets.chebyshev_jets(x, 4, 12, 6)
    line = cardinal.cardinal_line(x, jets=J, regularization=lam)
    ref = operators.derivatives(plan, jnp.asarray(f), orders=(1, 2, 3))
    ours = cardinal.derivatives(line, f, orders=(1, 2, 3))
    for k in (1, 2, 3):
        scale = np.max(abs(np.asarray(ref[k])))
        assert np.max(abs(np.asarray(ours[k]) - np.asarray(ref[k]))) < 1e-9 * scale
    points = np.linspace(-0.3, 1.2, 23)
    assert np.allclose(cardinal.evaluation_matrix(line, points) @ f,
                       calculus.interpolate(plan, jnp.asarray(f), jnp.asarray(points)), atol=1e-13)
    assert np.isclose(cardinal.integration_weights(line, 0.1, 0.9) @ f,
                      calculus.integrate(plan, jnp.asarray(f), a=0.1, b=0.9), atol=1e-13)


@pytest.mark.parametrize("size", [65, 129])
def test_parity_with_handoff_second_derivative(size):
    spatial = import_from("src/current_ns/core/original", "spatial_bspf")
    estimators = import_from("src/current_ns/core/original", "boundary_estimators")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = spatial.HighPrecisionBSPFSecondDerivative(size)
    J = estimators.build_stencil(op.x, 8, "fixed16d12")[0]
    line = cardinal.cardinal_line(op.x, degree=8, n_basis=24, order=8, jets=J, convention="closed")
    # The fit has a large gain (|P| ~ 1e6 at N=65, 3e8 at N=129): compare relative to it.
    assert np.max(abs(np.asarray(line.projection) - op.P)) < 1e-13 * np.max(abs(op.P))
    f = np.exp(np.cos(3 * op.x))
    exact = f * (9 * np.sin(3 * op.x) ** 2 - 9 * np.cos(3 * op.x))
    ours = np.asarray(cardinal.derivatives(line, f, orders=(2,))[2])
    assert np.max(abs(ours - exact)) <= 1.1 * np.max(abs(op.apply(f) - exact)) + 1e-12
