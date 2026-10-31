"""Axis construction, and parity with the implementations it replaces."""
import numpy as np
import pytest
from bspf_core import hybrid_axis, sem_axis, axis_from_family
from bspf_core import families
from conftest import load_reference


def test_trace_coordinates_structure_and_orthonormal_interior():
    axis = hybrid_axis(degree=9, modes=2)
    B = axis.evaluate([0.0, 1.0], 0)[0]
    assert np.allclose(B, np.eye(axis.size)[[0, -1]], atol=1e-12)
    x, w = np.polynomial.legendre.leggauss(80)
    V = axis.evaluate((x + 1) / 2, 0)[0][:, axis.interior]
    assert np.allclose(V.T @ (w[:, None] / 2 * V), np.eye(axis.size - 2), atol=1e-12)


def test_hybrid_span_contains_its_fourier_functions():
    axis = hybrid_axis(degree=5, modes=3, period=1.5, coordinates="orthonormal")
    x = np.linspace(0, 1, 40)
    B = axis.evaluate(x, 0)[0]
    target = np.cos(2 * np.pi * 3 / 1.5 * (x - 0.5))
    coef = np.linalg.lstsq(B, target, rcond=None)[0]
    assert np.max(abs(B @ coef - target)) < 1e-12


def test_derivatives_match_finite_differences():
    axis = hybrid_axis()
    x, h = np.array([0.37]), 1e-5
    B0, B1 = axis.evaluate(x, 1)
    fd = (axis.evaluate(x + h, 0)[0] - axis.evaluate(x - h, 0)[0]) / (2 * h)
    assert np.max(abs(fd - B1)) < 1e-6 * np.max(abs(B1))


def test_enriched_family_through_axis_from_family():
    raw = np.hstack([families.polynomial_family(6, 64),
                     families.sampled_family([lambda x: np.exp(-x / 0.05) * x * (1 - x)], 64)])
    axis = axis_from_family(raw, "trace")
    assert axis.size == 7 and axis.metadata["trace_defect"] < 1e-11


def test_parity_with_acoustics_boundary_axis():
    ref = load_reference("release/bspf-scooter/src/bspf_scooter/basis.py", "ref_basis")
    for family, n, p in (("bspf", 14, 9), ("bspf", 11, 6), ("sem", 12, 7)):
        old = ref.BoundaryAxis(family, n, p)
        new = hybrid_axis(p, (n - p - 1) // 2, 1.0, "trace") if family == "bspf" else sem_axis(n, "trace")
        assert new.coefficients.shape == old.coefficients.shape
        assert np.max(abs(new.coefficients - old.coefficients)) < 1e-13


def test_parity_with_vorticity_hybrid_axis():
    ref = load_reference("src/optimized_bvp/hybrid_axis.py", "ref_hybrid")
    for modes, period, degree in ((4, 1.5, 7), (6, 1.5, 6), (3, 2.0, 5)):
        old = ref.HybridAxis(modes=modes, period=period, degree=degree)
        new = hybrid_axis(degree, modes, period, "nodal")
        assert np.allclose(new.nodes, old.x, atol=1e-15)
        assert np.max(abs(new.coefficients - old.coefficients)) < 1e-12 * np.max(abs(old.coefficients))
