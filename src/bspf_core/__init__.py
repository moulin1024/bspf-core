"""bspf-core: one library for the BSPF function families and their solvers.

BSPF approximations combine a polynomial or B-spline block, which carries the
behaviour at interval ends, with Fourier functions, which carry the interior.
Several variants of that idea grew up in separate solvers (acoustics, Navier-
Stokes, curved-domain vorticity, pybspf models). This package is the single,
tested implementation they are being migrated onto.

The package is organised in layers. Each layer only uses the layers above it
in this list:

``bspf_core.quadrature``
    Gauss-Legendre rules and composite rules split at breakpoints.
``bspf_core.chebyshev``
    Chebyshev-series evaluation, differentiation and interpolation; the common
    representation of every function family.
``bspf_core.families``
    Raw function families on the reference interval: polynomial blocks,
    Fourier pairs of any period (as analytic Chebyshev tails), sampled
    enrichment functions.
``bspf_core.coordinates``
    Bases of a family's span: L2-orthonormal, endpoint-trace (two affine lifts
    plus a zero-trace orthonormal interior) and nodal (cardinal) coordinates.
``bspf_core.axis``
    :class:`~bspf_core.axis.Axis`, an immutable one-dimensional basis with
    derivatives, and factories such as :func:`~bspf_core.axis.hybrid_axis`.
``bspf_core.gram``
    Element quadrature and weighted derivative Gram matrices, including the
    Sturm-Liouville element matrices used by depth boundary-value problems.
``bspf_core.condensation``
    Static condensation: the pole (partial-fraction) form for parameter sweeps
    and dense multi-block condensation with a shared trace system.
``bspf_core.tridiagonal``
    Batched tridiagonal solves with adjacent partial pivoting, in JAX.
``bspf_core.parametric``
    A chain of condensed elements solved for many values of a parameter,
    the engine behind wavenumber integration.

A second, independent group implements the cardinal (sampled) BSPF line:

``bspf_core.splines``
    Clamped knot vectors and B-spline basis matrices (via SciPy).
``bspf_core.jets``
    Endpoint-jet estimators as linear maps from samples.
``bspf_core.cardinal``
    :class:`~bspf_core.cardinal.CardinalLine`: endpoint-constrained spline
    fit plus FFT remainder; derivatives, evaluation and integration.
``bspf_core.tensor``
    Tensor-product evaluation and adjoints, and fast diagonalization.

The Navier-Stokes streamfunction bases build on the cardinal line:

``bspf_core.compatible``
    :class:`~bspf_core.compatible.CompatibleSpace`: zero-trace functions and
    their primitives (``I' = T``), with extended-precision remainder factors.
``bspf_core.hermite``
    Streamfunction bubbles and :class:`~bspf_core.hermite.HermiteAxis`
    (end values and slopes plus bubbles).

Setup (basis construction, eigendecompositions) runs on the host with
NumPy/SciPy. Repeated online work (parameter sweeps, batched solves) runs in
JAX and requires 64-bit mode::

    import jax
    jax.config.update("jax_enable_x64", True)

Importing :mod:`bspf_core` never changes JAX's precision or device settings.
"""
from .quadrature import gauss_legendre, composite_gauss
from .axis import Axis, hybrid_axis, sem_axis, axis_from_family
from .gram import ElementRule, element_rule, gram, sturm_liouville_matrices
from .condensation import PoleCondensation, StaticCondensation
from .cardinal import CardinalLine, cardinal_line
from .tensor import FastDiagonalization
from .compatible import CompatibleSpace, compatible_space
from .hermite import BubbleBasis, HermiteAxis

__version__ = "0.1.0.dev0"
__all__ = ["Axis", "BubbleBasis", "CardinalLine", "CompatibleSpace", "ElementRule", "FastDiagonalization",
           "HermiteAxis", "PoleCondensation", "StaticCondensation", "axis_from_family", "cardinal_line",
           "compatible_space", "composite_gauss", "element_rule", "gauss_legendre", "gram", "hybrid_axis", "sem_axis",
           "sturm_liouville_matrices"]
