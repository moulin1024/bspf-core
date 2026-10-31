> **Superseded (2026-10-31).** bspf-core is being implemented in Python/JAX instead of C++,
> as a refactor of the existing BSPF code. The current design is the documentation in
> `docs/` (overview, API generated from docstrings, migration map). This C++ plan is kept for
> reference; its algorithms (pole-form condensation, pivoted tridiagonal solves) are implemented
> in `bspf_core.condensation`, `bspf_core.tridiagonal` and `bspf_core.parametric`.

# bspf-core: design plan

A compact C++20 core for the BSPF discretization of parametric one-dimensional
boundary-value problems. SCOOTER, and later other Acoustics Toolbox models,
are built on top of it. Status: design, no code yet. Date: 2026-10-31.

## 1. Purpose

1. Implement the BSPF numerical algorithm once, in a small, readable,
   dependency-light library that can be benchmarked fairly.
2. Make the method the only variable in comparisons: BSPF and SEM (and other
   element families) share every line of code except the basis.
3. Serve as the compute engine for a native SCOOTER implementation and, later,
   for modal (KRAKEN-type) and elastic solvers.

### Non-goals (for the first version)

- No acoustics inside the core: no sound speeds, densities, attenuation laws
  or file formats. Those belong to the SCOOTER layer.
- No 2-D/3-D problems, no sparse global solvers.
- No automatic differentiation, no GPU. The design must not prevent adding
  them later (pure functions, explicit data, no hidden global state).

## 2. Problem class

On a depth interval [z_top, z_bot] partitioned into elements, find u(z) with

    -(a(z) u')' + (lambda * b(z) - c(z)) u = f(z) + sum_s w_s delta(z - z_s)

for many values of a complex parameter lambda (for acoustics, lambda = k^2,
the squared horizontal wavenumber including any contour shift).

- a(z), b(z): complex coefficients; b real and positive is required for the
  fast condensation path (acoustics: a = b = 1/rho).
- c(z): complex (acoustics: c = (omega/c_p(z))^2 / rho, lossy).
- Coefficients may jump at element edges (media interfaces). Inside an element
  they may have kinks at known knot positions (piecewise-linear SSPs); the
  quadrature splits there.
- Ends: Dirichlet (u = 0), Neumann, or Robin with a lambda-dependent
  coefficient alpha_end(lambda) added to the end diagonal (radiating
  halfspaces, rough surfaces, tabulated reflection coefficients).
- Point sources sit on element edges; distributed loads f are supported (they
  are needed for manufactured-solution tests).

The weak form, with v in the same space, is

    int a u' v' + int (lambda b - c) u v + alpha_top u v |_top + alpha_bot u v |_bot
        = int f v + sum_s w_s v(z_s)

which is the operator already used by the Python implementation (the negative
of the original SCOOTER finite-difference operator).

## 3. The BSPF element space

Per element, mapped to the reference coordinate x in [0, 1] (z = z0 + h x):

    V = P_p  (+)  span{ cos(2 pi j (x - 1/2)), sin(2 pi j (x - 1/2)) : j = 1..m }

with dimension n = p + 1 + 2m (defaults p = 9, m = 2, n = 14). This must
reproduce `BoundaryAxis('bspf', n, p)` of the Python package to <= 1e-13.

Coordinates (identical to the Python implementation):

1. Represent every function by Chebyshev coefficients in t = 2x - 1, length
   N_c = max(128, 4n). The polynomial part uses T_0..T_p. Each Fourier
   function uses its Chebyshev series with the first p+1 terms removed (they
   lie in P_p already). The C++ version computes these coefficients by
   Chebyshev interpolation (DCT) at N_c points instead of the Jacobi-Anger
   Bessel series, so no special functions are needed; both are exact to
   rounding for these entire functions.
2. Orthonormalize in L2(0,1) with Gauss-Legendre quadrature of
   max(80, 4n) points: three passes of QR plus triangular solve.
3. Interior functions: an orthonormal basis of the zero-trace subspace (null
   space of the endpoint evaluation). Lifts: phi_0 = 1 - x, phi_{n-1} = x.
   Endpoint roundoff of the interior functions is removed with the lifts.
4. Store Chebyshev coefficients and their first two derivatives; evaluation is
   Clenshaw on demand, plus cached values at quadrature nodes.

Other families plug in through the same interface:

- `sem`: P_{n-1} only (consistent polynomial SEM in the same coordinates),
  used as the fair baseline.
- `enriched`: P_p plus user-supplied functions sampled at Chebyshev points
  (prior-informed bases such as WKB phases). If the enrichment does not
  depend on lambda, the fast path below still applies; lambda-dependent
  enrichments use the dense path.

## 4. Algorithm

### 4.1 Preparation (once per coefficient set, e.g. per frequency)

For each element e:

1. Quadrature: Gauss-Legendre of order q = max(2n + 4, user value), composite
   over the sub-intervals split at the element's knots.
2. Caller supplies a, b, c (and optionally f) at the quadrature nodes. The
   core never calls back into user code during assembly.
3. Element matrices: K = int a phi' phi' - int c phi phi, M = int b phi phi,
   load F = int f phi.
4. Interior condensation in pole form. With i = interior (n - 2 functions) and
   b = the two lifts: M_ii = L L^T (real Cholesky); C = L^-1 K_ii L^-T is
   complex symmetric; eigendecomposition C = Q diag(theta) Q^T with the
   complex-symmetric normalization Q^T Q = I. Then

       A_ii(lambda)^-1 = L^-T Q diag(1/(theta + lambda)) Q^T L^-1

   and every condensed quantity is a partial-fraction sum over the
   n - 2 poles -theta_j:

       S_ab(lambda) = A0_ab + lambda A1_ab - sum_j r_jab / (theta_j + lambda)
       g_a(lambda)  = F_a  - sum_j (s0_ja + lambda s1_ja) / (theta_j + lambda)   (condensed load)
       w_a(lambda)  = B_a  - sum_j t_ja / (theta_j + lambda)                    (receiver weights)

   Stored per element: theta (n-2), r (n-2 x 3), A0, A1 (3 each), plus load and
   receiver residues. Memory O(E n).
5. Safety: accept the eigenbasis only if ||Q diag(theta) Q^T - C|| <= 1e-10 ||C||
   (and Q^T Q is well conditioned). Otherwise the element is marked dense
   and is condensed directly for every lambda (O(n^3) per lambda). This
   fallback is the same code path used by elastic and lambda-dependent elements.

### 4.2 Online solve (per batch of lambda values)

For every lambda in the batch:

1. Evaluate the condensed 2x2 blocks: O(E n) operations.
2. Assemble the nodal tridiagonal system (E + 1 unknowns), add end terms,
   apply Dirichlet rows.
3. LU with adjacent partial pivoting (LAPACK gttrf semantics) and solves for
   all sources: O(E) per source.
4. Sample receivers with the precomputed residues: O(n) per receiver.

Cost per lambda: O(E n + E S + R n) for S sources and R receivers. Batches
are processed in fixed-size chunks; the lambda loop is the parallel loop
(OpenMP, optional). Arithmetic is complex<double>; the scalar type is a
template parameter so a quad-precision build can verify roundoff.

### 4.3 Outputs

- Green values g(s, r, lambda) at receivers (primary).
- Optional: u'(z_r), full nodal solution, element reconstruction.
- For future modal solvers: the trace-matrix determinant (with scaling to
  avoid overflow) and the Sturm/inertia count of the LU factorization for
  real lambda, so eigenvalue search can use counting and root finding.

## 5. Public API (sketch)

```cpp
namespace bspf {

enum class Family { bspf, sem, enriched };
struct BasisSpec { Family family = Family::bspf; int degree = 9; int fourier_pairs = 2; };

class ReferenceBasis {                       // section 3; immutable, shareable
 public:
  explicit ReferenceBasis(const BasisSpec&);
  int size() const;
  // values and derivatives (order 0..2) at reference points in [0, 1]
  Matrix evaluate(std::span<const double> x, int order) const;
};

struct Mesh {                                // element edges plus interior knots per element
  std::vector<double> edges;
  std::vector<std::vector<double>> knots;
};

class Discretization {                       // basis + mesh + quadrature; no coefficients yet
 public:
  Discretization(std::shared_ptr<const ReferenceBasis>, Mesh, int quadrature_order = 0);
  std::span<const double> quadrature_points() const;   // caller samples a, b, c, f here
};

struct Coefficients {                        // samples at quadrature_points()
  std::vector<complex> a, b, c, f;           // f may be empty
};

enum class EndKind { dirichlet, neumann, robin };
struct EndCondition { EndKind kind; };       // Robin values are passed per lambda batch

class Prepared {                             // section 4.1; immutable after construction
 public:
  Prepared(const Discretization&, const Coefficients&,
           std::span<const double> source_depths, std::span<const double> receiver_depths,
           EndCondition top, EndCondition bottom);
  // section 4.2: out has shape (sources, receivers, lambdas), row-major
  void green(std::span<const complex> lambda,
             std::span<const complex> alpha_top, std::span<const complex> alpha_bottom,
             std::span<const complex> source_weights, std::span<complex> out) const;
  Diagnostics diagnostics() const;           // dense-fallback count, eigen residuals, condition estimates
};

}  // namespace bspf
```

Design rules: no virtual calls in the inner loop; element kinds are resolved
at preparation into structure-of-arrays storage; all inputs explicit; no
global state; exceptions only at preparation, never inside the lambda loop.

## 6. Extension points (later milestones)

- **Vector unknowns (elastic media).** Solid elements carry two displacement
  components; their matrices are quadratic in h = sqrt(lambda), so they use
  the dense condensation path. The trace system becomes banded (half-bandwidth
  4) with fluid-solid coupling terms; a small banded LU with partial pivoting
  replaces the tridiagonal solver. The API keeps the scalar case unchanged.
- **Modal solvers.** Determinant and inertia outputs (4.3) support counting
  and root finding for normal modes without a sparse eigensolver.
- **Prior-informed bases.** `Family::enriched` (section 3).
- **Python bindings** with nanobind for testing and for the Python library.

## 7. Verification

Every check runs in CI-style unit tests (`doctest`, single header):

1. **Basis parity:** Chebyshev coefficients, orthonormality (<= 1e-13), trace
   exactness, derivatives; reproduce Python `BoundaryAxis` fixtures to 1e-13.
2. **Quadrature:** exactness for polynomials; split-rule integrals of
   piecewise-linear functions.
3. **Condensation:** pole form against direct dense condensation for random
   complex lambda, including near-pole lambda; dense fallback triggered and
   correct on a constructed defective case.
4. **Tridiagonal solver:** against a dense solve, including pivoting cases.
5. **Manufactured solutions:** smooth and piecewise-smooth u with interface
   continuity of u and a u', variable complex coefficients, Robin/Dirichlet
   ends, distributed load f. Report L2 and H1 errors; fit rates under
   h-refinement (algebraic) and p-refinement (exponential). Same tests for
   `sem`.
6. **Analytic Green functions:** constant and piecewise-constant layers
   (exact exponential/trigonometric solutions with transfer matrices),
   radiating ends, strongly evanescent lambda.
7. **Parity with the Python implementation:** Green values from the Python
   native SCOOTER kernel on suite-like problems to <= 1e-12.
8. **Determinism:** identical results with 1 and N threads.

## 8. Benchmark methodology

- Single-thread CPU time as the primary measure; wall time and thread scaling
  reported separately.
- Report preparation and per-lambda costs separately; per-lambda cost in ns
  and in operation counts.
- BSPF and SEM compared at matched accuracy against independent references
  (section 7.6 and ODE references), with the same mesh options and the same
  kernel.
- Original SCOOTER timed as a process with its own settings; any comparison
  at matched accuracy states the original's mesh.

## 9. Build and dependencies

- C++20, CMake >= 3.25, Ninja; GCC >= 13 (Homebrew g++-15 here) or Clang >= 16.
- Eigen 3.4 (header-only): Cholesky, QR, SVD, complex eigensolver.
- doctest (vendored single header) for tests.
- Optional: OpenMP for the lambda loop; nanobind for Python bindings.
- No LAPACK requirement for the scalar core.

## 10. Layout

```
bspf-core/
  CMakeLists.txt
  include/bspf/
    chebyshev.hpp     Clenshaw evaluation, derivatives, DCT interpolation
    quadrature.hpp    Gauss-Legendre, split composite rules
    basis.hpp         ReferenceBasis (bspf, sem, enriched)
    mesh.hpp          Mesh, Discretization
    condense.hpp      pole-form and dense element condensation
    tridiagonal.hpp   pivoted tridiagonal LU (batched)
    prepared.hpp      Prepared: assembly, green(), diagnostics
  src/                non-template implementations
  tests/              doctest units, MMS, analytic Green, Python fixtures
  bench/              kernel micro-benchmarks
  python/             nanobind module (optional target)
  docs/DESIGN.md      this document
```

Expected size of the scalar core: about 800-1,200 lines plus tests.

## 11. Milestones

| # | Deliverable | Acceptance |
|---|---|---|
| M1 | chebyshev, quadrature, basis (bspf, sem) | basis parity with Python <= 1e-13; tests 7.1-7.2 |
| M2 | scalar Prepared: assembly, pole condensation, tridiagonal solve, receivers | tests 7.3-7.6 pass; MMS rates documented for bspf and sem |
| M3 | Python bindings + parity | <= 1e-12 against the Python native SCOOTER kernel on fluid suite cases |
| M4 | SCOOTER layer (fluid): media, loads, native k grid, .grn writer; .env parsed by the existing Python reader | the 44 all-fluid suite cases match the Python implementation <= 1e-12; single-thread timing vs scooter.exe |
| M5 | vector unknowns and banded LU (elastic) | the 8 elastic suite cases |
| M6 | native .env parser and FIELDS in C++; stand-alone executables | all 52 cases end to end without Python |

## 12. Risks and open questions

1. **Eigenbasis conditioning.** Complex-symmetric interior pencils can be
   close to defective in lossy layers. Mitigation: residual check and the dense
   fallback (4.1 step 5); diagnostics count how often it triggers.
2. **Eigen's complex eigensolver accuracy** compared with LAPACK zgeev for
   n - 2 = 12. To be measured in M2; LAPACK can be added if needed.
3. **Toolchain on macOS.** Apple clang 14 lacks parts of C++20; the reference
   build uses GCC. OpenMP needs libomp with clang.
4. **Name of the method.** The current element space is polynomial plus
   Fourier per element (section 3). If the method definition used in the paper
   is a global B-spline space with Fourier enrichment, the basis layer must
   support it; the rest of the design is unchanged because only element
   matrices depend on the basis.
5. **Scope of the first SCOOTER layer.** M4 keeps .env parsing in Python to
   reach timing results early; M6 removes the dependency.
