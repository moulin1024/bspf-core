# bspf-core

One library for the BSPF function families (polynomial or B-spline block plus
Fourier functions), their Galerkin assembly and their structured solvers.
It consolidates implementations that grew separately in the acoustics,
Navier-Stokes and pybspf code bases.

```{toctree}
:maxdepth: 2

overview
api
migration
```

## Quick start

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from bspf_core import hybrid_axis, element_rule, sturm_liouville_matrices, PoleCondensation

axis = hybrid_axis(degree=9, modes=2)                 # 14 functions, trace coordinates
rule = element_rule(axis, 0.0, 10.0)                  # quadrature + physical values
one = np.ones_like(rule.nodes)
K, M = sturm_liouville_matrices(rule, one, one, 0.04 * one)
S = PoleCondensation(K, M).schur(np.array([0.01, 0.02]))   # condensed 2x2 per lambda
```

Building this documentation: `sphinx-build -b html docs docs/_build/html`.
