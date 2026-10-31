# Overview

## Representation

Every basis is a matrix of Chebyshev coefficients on the reference interval
$[0, 1]$, one column per function. Families (raw spans) become bases through
one of three coordinate systems:

| Coordinates | Property | Used for |
| --- | --- | --- |
| `trace` | lifts $1-x$, $x$ plus an orthonormal zero-trace interior | element Galerkin methods, condensation |
| `nodal` | cardinal at given nodes ($\phi_j(x_i) = \delta_{ij}$) | nodal Galerkin / collocation solvers |
| `orthonormal` | $L^2(0,1)$-orthonormal | projections, enrichment studies |

## Layers

| Module | Role | Runs on |
| --- | --- | --- |
| `quadrature` | Gauss rules, composite rules split at knots | host (NumPy) |
| `chebyshev` | series evaluation, derivatives, interpolation | host |
| `families` | polynomial blocks, Fourier tails, sampled enrichments | host |
| `coordinates` | orthonormal / trace / nodal coordinates | host |
| `axis` | `Axis` objects and factories | host |
| `gram` | element rules, weighted Gram matrices, Sturm-Liouville matrices | host |
| `condensation` | pole-form and multi-block static condensation | host setup |
| `tridiagonal` | batched pivoted tridiagonal solves | JAX |
| `parametric` | chains of condensed elements for parameter sweeps | JAX |
| `splines` | clamped knots, B-spline basis and primitive matrices (SciPy) | host |
| `jets` | endpoint-jet estimators as $2q \times N$ matrices | host |
| `cardinal` | cardinal BSPF line: constrained spline fit + FFT remainder | host setup, JAX apply |
| `tensor` | tensor-product evaluation/adjoint, fast diagonalization | NumPy or JAX |
| `compatible` | zero-trace spaces `T` and primitives `I` ($I' = T$) | host |
| `hermite` | streamfunction bubbles, value/slope Hermite axes | host |

Setup is done once on the host; repeated work runs in compiled JAX with
64-bit floats.

## Which BSPF?

Two constructions share the name, and both are here:

*Independent hybrid element* (`axis`, `gram`, `condensation`)
: $P_p$ plus independent Fourier pairs on each element, used as a Galerkin
  basis. This is what the SCOOTER reimplementation uses.

*Cardinal line* (`cardinal`)
: samples on a uniform grid; a clamped B-spline fitted to estimated endpoint
  jets absorbs the non-periodic part and the FFT handles the smooth remainder.
  This is the pybspf construction and the derivative operator of the
  hand-off Navier-Stokes solvers. It supports two Fourier conventions:
  `"samples"` (pybspf, period $N h$) and `"closed"` (hand-off, period
  $(N-1) h$).

The fit is computed by the row-scaled null-space QR method rather than a KKT
solve. The fit map has a large gain (about $10^8$ for degree 8 at $N=129$)
and a KKT solve loses accuracy there; see {func}`bspf_core.cardinal.cardinal_line`.

## Navier-Stokes bases and extended precision

The streamfunction solvers use the closed cardinal line through
{mod}`bspf_core.compatible` (the `T`/`I` pair) and {mod}`bspf_core.hermite`
(bubbles with zero end value and slope, plus cubic Hermite end functions).
Their lines constrain five endpoint orders, and the fit gain grows like
$N^4$ ($5 \times 10^7$ at $N = 129$). Values between grid points are
$\operatorname{trig}(f) + G P f$ with $G = B - \operatorname{trig}(B)$ small,
so $G$ must be accurate relative to its own size: in float64 the second
derivative loses six digits at $N = 129$.
{class}`~bspf_core.cardinal.RemainderFactors` computes $G$ with `gmpy2` (MPFR)
at binary128 precision and rounds once, reproducing the hand-off
quad-precision C factors to about $10^{-14}$. It is setup-only work: a
{class}`~bspf_core.hermite.HermiteAxis` takes about 0.04 s for 17 samples
and 0.13 s for 49 (hand-off C version: 0.02 s and 0.05 s).
