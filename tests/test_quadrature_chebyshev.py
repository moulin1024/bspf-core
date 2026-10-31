import numpy as np
import pytest
from bspf_core import gauss_legendre, composite_gauss
from bspf_core import chebyshev


def test_gauss_legendre_exactness_on_interval():
    x, w = gauss_legendre(5, (2.0, 5.0))
    for k in range(10):
        assert np.isclose(w @ x**k, (5**(k + 1) - 2**(k + 1)) / (k + 1), rtol=1e-14)


def test_composite_rule_is_exact_for_kinked_function():
    x, w = composite_gauss([0, 0.3, 0.3, 1.0], 3)
    assert np.isclose(w @ np.abs(x - 0.3) ** 3, (0.3**4 + 0.7**4) / 4, rtol=1e-14)


@pytest.mark.parametrize("bad", [[1.0], [0.0, -1.0], [0.0, np.inf]])
def test_composite_rule_rejects_bad_breakpoints(bad):
    with pytest.raises(ValueError):
        composite_gauss(bad, 3)


def test_chebyshev_derivative_and_interpolation():
    c = chebyshev.interpolate([np.sin, np.exp], 30)
    x = np.linspace(0, 1, 7)
    assert np.allclose(chebyshev.evaluate(c, x), np.c_[np.sin(x), np.exp(x)], atol=1e-14)
    d = chebyshev.derivative(c, 2)
    # Differentiating an interpolant amplifies its rounding by roughly terms^4.
    assert np.allclose(chebyshev.evaluate(d, x), np.c_[-np.sin(x), np.exp(x)], atol=1e-7)
    p = chebyshev.chebyshev_points(6)
    assert np.all(np.diff(p) > 0) and 0 < p[0] and p[-1] < 1
