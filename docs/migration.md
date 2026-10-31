# Migration map

What each part of bspf-core replaces. "Parity" means a test in `tests/`
reproduces the old implementation to rounding.

## Implemented in 0.1

| bspf-core | Replaces | Parity |
| --- | --- | --- |
| `quadrature.composite_gauss` | about 12 private split Gauss rules (pybspf models and scratch, acoustics `split_rule`, NS `quadrature`, `rule`) | exactness tests |
| `axis.hybrid_axis(..., "trace")` | acoustics `BoundaryAxis("bspf")` | yes, to 1e-13 |
| `axis.sem_axis(..., "trace")` | acoustics `BoundaryAxis("sem")` | yes, to 1e-13 |
| `axis.hybrid_axis(..., "nodal")` | `optimized_bvp.HybridAxis` (vorticity solvers) | yes, to 1e-12 |
| `axis.axis_from_family` + `families.sampled_family` | ad hoc enrichment bases | structure tests |
| `gram.sturm_liouville_matrices` | element assembly in native SCOOTER and the layered acoustic solvers | via the SCOOTER parity test |
| `condensation.PoleCondensation` | native SCOOTER `_Fluid` and its partial fractions | yes, to 1e-10 |
| `condensation.StaticCondensation` | `CondensedInverse` (backward step, NACA C-grid) | exactness test |
| `tridiagonal.solve` | `scooter_fluid_jax._tridiagonal`, `banded_jax` | dense comparison |
| `parametric.ParametricChain` | `scooter_fluid_jax.fluid_solve` | yes, native SCOOTER kernel to 1e-10 |

## Planned

| Next component | Candidates to consolidate (from the pybspf survey) |
| --- | --- |
| Cardinal BSPF line (QR-constrained endpoint-jet fit + FFT remainder) | core `pybspf._qr_trial.qr_spline_fit`; `bspf_models._numerics.trial_spaces._line_projector`; hand-off `spatial_bspf.constrained_map` |
| Cardinal point evaluation | `_numerics._weak_basis` (MPFR), `_numerics._gpu_basis` (JAX), four scratch `Axis` copies |
| Endpoint-jet estimation | `pybspf.endpoints.chebyshev_boundary_blocks`, `_qr_trial.chebyshev_jets`, Taylor jets in `_line_projector` |
| Clamped / derivative-closed spaces (I' = T) | `pybspf.trial_spaces.ClosedBSPFLine`, `_numerics.trial_spaces._stream_line`, hand-off `CompatibleLine`, `OpenCompatibleLine` |
| Hermite trace axis with clamped bubbles | backward-step `TraceAxis`, NACA `Basis` |
| Independent hybrid variants still outside | `bspf_models.fluids._embedded.stable_line.StableLine` (and its scratch twin), `free_bspf.FreeLine`, `stable_hybrid.StableAxis` |
| Tensor-product evaluation / adjoint | backward-step and NACA `Block.fields` / `adjoint`, `pybspf.tensor.tensor_product` |
| Fast diagonalization | `pybspf.tensor.tensor_elliptic_solve`; reimplementations in `elliptic.pressure`, `pressure3d`, `weak_navier_stokes`, `smooth_extension`; hand-off `TensorMass` |
| Circulant + low-rank solves | `pybspf.fast_axis`, hand-off `StructuredMass` |
| Multi-patch bookkeeping and interface terms | backward-step / NACA union-find joins and SIP faces; `scratch/mapped_spline_bvp.py` |

Each step adds the component with a parity test against the listed
implementations, then switches callers one at a time.
