# bspf-core

Consolidated BSPF function families, Galerkin assembly and structured solvers, in NumPy (setup) and
JAX (repeated work). Development version 0.1.0.dev0.

- Bases: hybrid polynomial + Fourier elements and spectral elements, in trace, nodal or orthonormal
  coordinates (`bspf_core.axis`).
- Assembly: element quadrature split at knots, weighted Gram and Sturm-Liouville matrices
  (`bspf_core.gram`).
- Solvers: pole-form condensation for parameter sweeps, multi-block static condensation, batched
  pivoted tridiagonal solves, chains of condensed elements (`bspf_core.condensation`,
  `bspf_core.tridiagonal`, `bspf_core.parametric`).
- Cardinal BSPF lines: a clamped B-spline fitted to estimated endpoint jets plus an FFT remainder;
  derivatives, interpolation and integration of sampled data, in the pybspf and hand-off Fourier
  conventions (`bspf_core.splines`, `bspf_core.jets`, `bspf_core.cardinal`).
- Tensor products: evaluation, Galerkin adjoints and fast diagonalization (`bspf_core.tensor`).
- Navier-Stokes streamfunction bases: derivative-compatible zero-trace spaces and Hermite
  value/slope axes with bubbles, using extended-precision (mpmath) remainder factors
  (`bspf_core.compatible`, `bspf_core.hermite`).

Each component reproduces the implementation it replaces to rounding; see `docs/migration.md`.

```sh
pip install -e '.[test,docs]'
pytest                                          # unit and parity tests
pytest --doctest-modules src/bspf_core          # docstring examples (JAX_ENABLE_X64=1)
sphinx-build -b html docs docs/_build/html      # documentation from docstrings
```

License: GPL-3.0 (see `LICENSE`).

Parity tests compare against implementations in the BSPF hand-off workspace and in pybspf (a sibling
checkout) and are skipped when those are not present.
