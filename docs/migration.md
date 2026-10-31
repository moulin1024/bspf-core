# Migration map

What each part of bspf-core replaces. "Parity" means a test in `tests/`
reproduces the old implementation to rounding.

## Implemented in 0.1 (first batch)

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

## Implemented in 0.1 (second batch)

| bspf-core | Replaces | Parity |
| --- | --- | --- |
| `splines.clamped_knots`, `basis_matrix`, `primitive_matrix` | `pybspf.basis.open_knots` / `basis_matrix` / `spline_primitive`; hand-off `open_uniform_knots` | via the cardinal parity tests |
| `jets.taylor_jets` | `pybspf.plan_1d` finite-difference jets (`"interpolation"`); `_qr_trial` Taylor jets (`"least_squares"`) | exactness tests; via `plan_1d` parity |
| `jets.chebyshev_jets` | `pybspf.endpoints.chebyshev_boundary_blocks`, `_qr_trial.chebyshev_jets` | yes, to 1e-12 relative |
| any $2q \times N$ matrix as `jets=` | hand-off `boundary_estimators.build_stencil` (all schemes) | yes (`fixed16d12`) |
| `cardinal.cardinal_line` + `derivatives` (`"samples"`) | `pybspf.plan_1d` + `operators.derivatives`, including `lam > 0` | yes, orders 1-3 to 1e-9 relative |
| `cardinal.evaluation_matrix`, `integration_weights` | `pybspf.calculus.interpolate`, `integrate` | yes, to 1e-13 |
| `cardinal.cardinal_line(..., convention="closed")` | hand-off `HighPrecisionBSPFSecondDerivative` (`constrained_map`) | `P` to 1e-13 relative; error vs exact no worse |
| `tensor.evaluate`, `adjoint`, `apply_axes` | `pybspf.tensor.tensor_product`; backward-step / NACA `Block.fields` / `adjoint` | dense Kronecker tests |
| `tensor.FastDiagonalization` | `pybspf.tensor.tensor_elliptic_solve` and its reimplementations | dense solve test |

Not yet covered from these sources: pybspf noisy-data smoothing
(`noise_std`), `antiderivative` with boundary values, and `with_regularization`
as a traced function of `lam`.

## Implemented in 0.1 (third batch: Navier-Stokes bases)

| bspf-core | Replaces | Parity |
| --- | --- | --- |
| `jets.polynomial_jets` | hand-off `build_stencil("fixed{m}d{d}")` | yes, to 1e-14 relative |
| `cardinal_line(..., symmetrize=True)` | hand-off reflection balancing `(P + P[::-1, ::-1]) / 2` | via the compatible-space test |
| `cardinal.RemainderFactors`, `primitive_matrix` | hand-off `quad_basis.c` / `QuadFactors` (libquadmath; fixed degree 7, ten functions, knots at thirds) | via the compatible-space test; 34 vs 50 digits agree to 1e-14 |
| `compatible.compatible_space(...).values` | `line_space.CompatibleLine.basis` (`"t"` orders 0-2, `"i"` orders 0-1) | yes, to 1e-13 relative (N = 17, 49, 129) |
| `hermite.BubbleBasis` | `ns_exponential_enrichment.Axis` without dictionary | via the Hermite test |
| `hermite.HermiteAxis` | backward-step `TraceAxis` (and the base of NACA `Basis`) | yes, same functions to 1e-13 relative |

Not carried over from `CompatibleLine`: circulant-plus-low-rank mass and
stiffness operators, the pressure space and dealiased uniform
reconstruction; the block solvers do not use them. Wall exponential
enrichment (`Axis` with a dictionary, NACA radial enrichment) is not in the
core yet.

## Planned

| Next component | Candidates to consolidate (from the pybspf survey) |
| --- | --- |
| Switch remaining cardinal callers to `bspf_core.cardinal` | `_qr_trial.qr_spline_fit`, `_numerics.trial_spaces._line_projector`, `_numerics._weak_basis` (MPFR), `_numerics._gpu_basis`, four scratch `Axis` copies |
| Remaining derivative-closed spaces (I' = T) | `pybspf.trial_spaces.ClosedBSPFLine`, `_numerics.trial_spaces._stream_line`, hand-off `OpenCompatibleLine` |
| Enriched Hermite axes | NACA `Basis` (reflection, radial exponential enrichment), `ns_exponential_enrichment.Axis` dictionary |
| Independent hybrid variants still outside | `bspf_models.fluids._embedded.stable_line.StableLine` (and its scratch twin), `free_bspf.FreeLine`, `stable_hybrid.StableAxis` |
| Switch fast-diagonalization callers | reimplementations in `elliptic.pressure`, `pressure3d`, `weak_navier_stokes`, `smooth_extension`; hand-off `TensorMass` |
| Circulant + low-rank solves | `pybspf.fast_axis`, hand-off `StructuredMass` |
| Multi-patch bookkeeping and interface terms | backward-step / NACA union-find joins and SIP faces; `scratch/mapped_spline_bvp.py` |

Each step adds the component with a parity test against the listed
implementations, then switches callers one at a time.
