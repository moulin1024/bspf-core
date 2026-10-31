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

Smooth B-spline spaces with derivative closure are still planned; see {doc}`migration`.
