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

Setup is done once on the host; repeated work runs in compiled JAX with
64-bit floats.

## Which BSPF?

This release implements the *independent hybrid* BSPF element
($P_p$ plus independent Fourier pairs). The *cardinal* construction of pybspf
(a B-spline fitted to endpoint jets plus an FFT remainder) and smooth
B-spline spaces are planned; see {doc}`migration`.
