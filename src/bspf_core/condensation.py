"""Static condensation of element interiors.

Two forms are provided.

Pole (partial-fraction) form, for parameter sweeps
---------------------------------------------------
For a trace-coordinate element (see :class:`bspf_core.axis.Axis`) with
matrices ``K`` and ``M`` the operator is :math:`A(\\lambda) = K + \\lambda M`.
Split the unknowns into the two boundary (lift) functions ``b`` and the
interior functions ``i``. With :math:`M_{ii} = L L^T` (real Cholesky) and the
complex-symmetric eigendecomposition

.. math::

    L^{-1} K_{ii} L^{-T} = Q\\, \\operatorname{diag}(\\theta)\\, Q^T, \\qquad Q^T Q = I,

the interior inverse is
:math:`A_{ii}(\\lambda)^{-1} = L^{-T} Q\\,\\operatorname{diag}(1/(\\theta + \\lambda))\\,Q^T L^{-1}`.
Every condensed quantity is then a sum over the ``n - 2`` fixed poles
:math:`-\\theta_j`. For the 2x2 Schur complement,

.. math::

    S_{ab}(\\lambda) = A^{0}_{ab} + \\lambda A^{1}_{ab} - \\sum_j \\frac{r_{jab}}{\\theta_j + \\lambda},

with constants computed once (:meth:`PoleCondensation.partial_fractions`).
Evaluating ``S`` for another ``lambda`` costs ``O(n)`` instead of a dense
``O(n^3)`` solve. Values at interior points (receivers) have the same form
(:meth:`PoleCondensation.sampling`).

This requires ``M_ii`` real symmetric positive definite and ``K_ii``
diagonalizable with a well-conditioned eigenbasis; the constructor checks the
eigendecomposition residual and raises if it is not met, so callers can fall
back to dense condensation.

Dense multi-block form, for fixed operators
--------------------------------------------
:class:`StaticCondensation` factors a symmetric positive-definite global
matrix whose unknowns split into *private* groups (one per block, coupled only
through the shared unknowns) and *shared* unknowns (interfaces). Each private
block is Cholesky-factored, the Schur complement on the shared unknowns is
formed and factored, and every later solve is direct.
"""
import numpy as np
import scipy.linalg as la

_PAIRS = ((0, 0), (0, 1), (1, 1))


class PoleCondensation:
    """Pole-form condensation of one trace-coordinate element.

    Parameters
    ----------
    K, M : ndarray, shape (n, n)
        Element matrices with :math:`A(\\lambda) = K + \\lambda M`, in trace
        coordinates (boundary unknowns ``0`` and ``n - 1``). ``K`` may be
        complex symmetric; ``M`` must be real symmetric positive definite on
        the interior block. Typically from
        :func:`bspf_core.gram.sturm_liouville_matrices`.
    tolerance : float, optional
        Relative residual accepted for the interior eigendecomposition.
        Default ``1e-10``.

    Attributes
    ----------
    theta : ndarray, shape (n - 2,)
        Pole parameters; the poles in ``lambda`` are ``-theta``.
    interior : ndarray
        Interior indices ``1 .. n - 2``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the interior eigenbasis does not reproduce the reduced matrix to
        ``tolerance`` (nearly defective interior pencil).
    ValueError
        If the interior mass block is not real.

    Examples
    --------
    >>> import numpy as np
    >>> from bspf_core import hybrid_axis, element_rule, sturm_liouville_matrices
    >>> rule = element_rule(hybrid_axis(), 0.0, 10.0)
    >>> one = np.ones_like(rule.nodes)
    >>> K, M = sturm_liouville_matrices(rule, one, one, 0.04 * one)
    >>> pc = PoleCondensation(K, M)
    >>> S = pc.schur(np.array([0.01 + 1e-4j]))      # shape (1, 2, 2)
    >>> S.shape
    (1, 2, 2)
    """

    def __init__(self, K, M, tolerance=1e-10):
        K, M = np.asarray(K), np.asarray(M)
        n = len(K)
        b, i = np.array([0, n - 1]), np.arange(1, n - 1)
        Mii = M[np.ix_(i, i)]
        if np.any(np.iscomplex(Mii)):
            raise ValueError("pole condensation needs a real interior mass block")
        # Reduce the generalized problem to a standard complex-symmetric one.
        L = np.linalg.cholesky(Mii.real)
        Linv = la.solve_triangular(L, np.eye(len(i)), lower=True)
        C = Linv @ K[np.ix_(i, i)] @ Linv.T
        theta, Q = la.eig(C)
        # Complex-symmetric normalisation q^T q = 1 (not the Hermitian norm).
        Q = Q / np.sqrt(np.sum(Q * Q, axis=0))
        residual = np.linalg.norm(Q * theta @ Q.T - C) / np.linalg.norm(C)
        if residual > tolerance:
            raise np.linalg.LinAlgError(f"ill-conditioned interior eigenbasis (residual {residual:.1e})")
        G = Q.T @ Linv
        self.theta = theta
        self.interior = i
        self.size = n
        self._P0, self._P1 = G @ K[np.ix_(i, b)], G @ M[np.ix_(i, b)]
        self._Kbb, self._Mbb = K[np.ix_(b, b)], M[np.ix_(b, b)]
        self._basis = Linv.T @ Q            # maps pole coordinates to interior coefficients
        self.residual = float(residual)

    def schur(self, lam):
        """Condensed 2x2 element matrices for a batch of parameters.

        Parameters
        ----------
        lam : array_like, shape (k,)
            Parameter values (complex allowed).

        Returns
        -------
        ndarray, shape (k, 2, 2)
            :math:`S(\\lambda) = A_{bb} - A_{bi} A_{ii}^{-1} A_{ib}`.
        """
        lam = np.asarray(lam)
        U = self._P0 + lam[:, None, None] * self._P1
        inv = 1 / (self.theta + lam[:, None])
        return self._Kbb + lam[:, None, None] * self._Mbb - np.einsum("kja,kjb,kj->kab", U, U, inv)

    def partial_fractions(self):
        """Constants of the 2x2 Schur complement in partial-fraction form.

        Returns
        -------
        theta : ndarray, shape (n - 2,)
        residues : ndarray, shape (n - 2, 3)
            ``r_j`` for the entries ``(0, 0), (0, 1), (1, 1)``.
        A0, A1 : ndarray, shape (3,)
            Constant and linear parts for the same entries, such that
            ``S_ab(lam) = A0 + lam * A1 - sum_j r_j / (theta_j + lam)``.

        Notes
        -----
        Derived from the identity
        :math:`(c_0 + c_1\\lambda + c_2\\lambda^2)/(\\theta+\\lambda)
        = c_2\\lambda + (c_1 - c_2\\theta) + (c_0 - c_1\\theta + c_2\\theta^2)/(\\theta+\\lambda)`
        with :math:`c_0 = p_{0a} p_{0b}`, :math:`c_1 = p_{0a} p_{1b} + p_{1a} p_{0b}`,
        :math:`c_2 = p_{1a} p_{1b}`.
        """
        P0, P1, t = self._P0, self._P1, self.theta[:, None]
        c0 = np.stack([P0[:, a] * P0[:, b] for a, b in _PAIRS], -1)
        c1 = np.stack([P0[:, a] * P1[:, b] + P1[:, a] * P0[:, b] for a, b in _PAIRS], -1)
        c2 = np.stack([P1[:, a] * P1[:, b] for a, b in _PAIRS], -1)
        residues = c0 - c1 * t + c2 * t * t
        A0 = np.array([self._Kbb[a, b] for a, b in _PAIRS]) - np.sum(c1 - c2 * t, 0)
        A1 = np.array([self._Mbb[a, b] for a, b in _PAIRS]) - np.sum(c2, 0)
        return self.theta, residues, A0, A1

    def sampling(self, row):
        """Partial-fraction form of a point evaluation inside the element.

        For basis values ``row`` at a point (shape ``(n,)``), the value of the
        condensed solution is ``w_0 u_0 + w_1 u_1`` with end values
        ``u_0, u_1`` and

        ``w_a(lam) = constant_a - sum_j residues_ja / (theta_j + lam)``.

        Parameters
        ----------
        row : ndarray, shape (n,)
            Basis function values at the point (trace coordinates).

        Returns
        -------
        constant : ndarray, shape (2,)
        residues : ndarray, shape (n - 2, 2)
        theta : ndarray, shape (n - 2,)
        """
        R = row[self.interior] @ self._basis
        constant = row[[0, self.size - 1]] - R @ self._P1
        residues = R[:, None] * (self._P0 - self.theta[:, None] * self._P1)
        return constant, residues, self.theta


class StaticCondensation:
    """Exact direct solver by condensation onto shared unknowns.

    Parameters
    ----------
    A : array_like or scipy.sparse matrix, shape (N, N)
        Symmetric positive-definite matrix.
    shared : array_like of int
        Indices of unknowns that couple different private groups (interface
        unknowns).
    private : sequence of array_like of int
        One index group per block. Groups must be disjoint, must not overlap
        ``shared``, and together with ``shared`` must cover ``0 .. N-1``.
        Unknowns of different groups must not be coupled directly.

    Notes
    -----
    Setup factors each private block ``A_pp`` (dense Cholesky) and the Schur
    complement :math:`S = A_{ss} - \\sum_p A_{sp} A_{pp}^{-1} A_{ps}` (dense
    Cholesky). Each :meth:`solve` is then one private solve per block, one
    shared solve and one back-substitution per block. This is the solver
    used by the rectangular backward-step and NACA C-grid prototypes.

    Raises
    ------
    ValueError
        If the index sets do not partition the unknowns.
    numpy.linalg.LinAlgError
        If a factorization fails (matrix not positive definite).
    """

    def __init__(self, A, shared, private):
        shared = np.asarray(shared, dtype=int)
        private = [np.asarray(p, dtype=int) for p in private]
        N = A.shape[0]
        everything = np.concatenate([shared, *private])
        if len(everything) != N or len(np.unique(everything)) != N:
            raise ValueError("shared and private indices must partition the unknowns")
        dense = (lambda X: X.toarray()) if hasattr(A, "toarray") else np.asarray
        Ar = A.tocsr() if hasattr(A, "tocsr") else np.asarray(A)
        rows = lambda r, c: dense(Ar[r][:, c])
        self.shared, self.private, self.size = shared, private, N
        schur = rows(shared, shared)
        self._local = []
        for ids in private:
            factor = la.cho_factor(rows(ids, ids), check_finite=False)
            coupling = rows(ids, shared)
            W = la.cho_solve(factor, coupling, check_finite=False)
            schur = schur - coupling.T @ W
            self._local.append((factor, coupling, W))
        self._schur = la.cho_factor((schur + schur.T) / 2, check_finite=False)

    def solve(self, rhs):
        """Solve ``A z = rhs`` directly.

        Parameters
        ----------
        rhs : ndarray, shape (N,)

        Returns
        -------
        ndarray, shape (N,)
        """
        rhs = np.asarray(rhs)
        z = np.empty(self.size, dtype=np.result_type(rhs, float))
        r = rhs[self.shared].copy()
        partial = []
        for ids, (factor, coupling, _) in zip(self.private, self._local):
            a = la.cho_solve(factor, rhs[ids], check_finite=False)
            partial.append(a)
            r = r - coupling.T @ a
        z[self.shared] = la.cho_solve(self._schur, r, check_finite=False)
        for ids, a, (_, _, W) in zip(self.private, partial, self._local):
            z[ids] = a - W @ z[self.shared]
        return z
