"""
Utility classes and functions for exact inference on the difference between
two independent binomial proportions.
"""

import numpy as np
from scipy.stats import binom


def qbinom(alpha: float, n: int, pi: float) -> int:
    """
    Conservatively invert the binomial CDF.

    Returns the smallest k* such that:
      - F(k*) >= alpha  if alpha >= 0.5   (upper quantile)
      - F(k*) <= alpha  if alpha < 0.5    (lower quantile — add 1 before using as rejection bound)

    This matches the conservative pivoting described in the post.
    """
    assert 0 <= alpha <= 1
    kstar = int(binom.ppf(q=alpha, n=n, p=pi))
    Fk = binom.cdf(k=kstar, n=n, p=pi)
    check = bool(np.where(alpha > 0.5, Fk >= alpha, Fk <= alpha))
    while not check:
        kstar = int(np.where(alpha > 0.5, kstar + 1, kstar - 1))
        Fk = binom.cdf(k=kstar, n=n, p=pi)
        check = bool(np.where(alpha > 0.5, Fk >= alpha, Fk <= alpha))
    return kstar


class dist_binom2:
    """
    Joint distribution of (y1, y2) where y1 ~ B(n1, pi1) and y2 ~ B(n2, pi2)
    are independent. Pre-computes the full PMF and CDF over the support grid.
    """

    def __init__(self, n1: int, pi1: float, n2: int, pi2: float):
        assert isinstance(n1, int) and isinstance(n2, int)
        assert n1 >= 0 and n2 >= 0
        assert 0 <= pi1 <= 1 and 0 <= pi2 <= 1

        self.n1, self.pi1, self.n2, self.pi2 = n1, pi1, n2, pi2

        k1 = np.arange(n1 + 1)
        k2 = np.arange(n2 + 1)

        pmf1 = binom.pmf(k=k1, n=n1, p=pi1)
        pmf2 = binom.pmf(k=k2, n=n2, p=pi2)
        cdf1 = binom.cdf(k=k1, n=n1, p=pi1)
        cdf2 = binom.cdf(k=k2, n=n2, p=pi2)

        # Outer products: rows = k2 values, cols = k1 values
        pmf_mat = np.outer(pmf2, pmf1)   # shape (n2+1, n1+1)
        cdf_mat = np.outer(cdf2, cdf1)   # shape (n2+1, n1+1)

        # Build tidy data frame for lookup
        rows = []
        for i2, v2 in enumerate(k2):
            for i1, v1 in enumerate(k1):
                rows.append((int(v1), int(v2),
                              pmf_mat[i2, i1], cdf_mat[i2, i1]))
        import pandas as pd
        self.mat_dist = pd.DataFrame(rows, columns=['v1', 'v2', 'pmf', 'cdf'])

    def pmf(self, y1: int, y2: int) -> float:
        assert 0 <= y1 <= self.n1 and 0 <= y2 <= self.n2
        return (binom.pmf(k=y1, n=self.n1, p=self.pi1) *
                binom.pmf(k=y2, n=self.n2, p=self.pi2))

    def cdf(self, y1: int, y2: int) -> float:
        assert 0 <= y1 <= self.n1 and 0 <= y2 <= self.n2
        return (binom.cdf(k=y1, n=self.n1, p=self.pi1) *
                binom.cdf(k=y2, n=self.n2, p=self.pi2))


class binom_diff:
    """
    Marginal distribution of the difference delta = y1 - y2 under the null
    hypothesis H0: pi1 = pi2 = pi0.

    The support of delta is {-n2, ..., n1}.

    Pre-computes the full PMF and CDF by convolving the two marginals.
    """

    def __init__(self, n1: int, pi0: float, n2: int):
        assert isinstance(n1, int) and isinstance(n2, int)
        assert n1 >= 0 and n2 >= 0
        assert 0 <= pi0 <= 1

        self.n1, self.n2, self.pi0 = n1, n2, pi0

        k1 = np.arange(n1 + 1)
        k2 = np.arange(n2 + 1)
        pmf1 = binom.pmf(k=k1, n=n1, p=pi0)
        pmf2 = binom.pmf(k=k2, n=n2, p=pi0)

        # delta = k1 - k2 ranges from -n2 to n1
        delta_min = -n2
        delta_max = n1
        delta_vals = np.arange(delta_min, delta_max + 1)
        pmf_delta = np.zeros(len(delta_vals))

        for i1, v1 in enumerate(k1):
            for i2, v2 in enumerate(k2):
                d = v1 - v2
                idx = d - delta_min
                pmf_delta[idx] += pmf1[i1] * pmf2[i2]

        cdf_delta = np.cumsum(pmf_delta)
        # Clamp floating-point drift at the top
        cdf_delta = np.minimum(cdf_delta, 1.0)

        self._delta_vals = delta_vals
        self._pmf = pmf_delta
        self._cdf = cdf_delta

    def pmf(self, d: int) -> float:
        """P(delta == d)"""
        if d < self._delta_vals[0] or d > self._delta_vals[-1]:
            return 0.0
        return float(self._pmf[d - self._delta_vals[0]])

    def cdf(self, d: int) -> float:
        """P(delta <= d)"""
        if d < self._delta_vals[0]:
            return 0.0
        if d >= self._delta_vals[-1]:
            return 1.0
        return float(self._cdf[d - self._delta_vals[0]])

    def qdf(self, alpha: float) -> int:
        """
        Conservatively invert the CDF of delta, analogous to qbinom.

        Returns k* such that:
          - F(k*) >= alpha  if alpha >= 0.5
          - F(k*) <= alpha  if alpha < 0.5
        """
        assert 0 <= alpha <= 1
        if alpha >= 0.5:
            # Find smallest k with F(k) >= alpha
            for k, c in zip(self._delta_vals, self._cdf):
                if c >= alpha:
                    return int(k)
            return int(self._delta_vals[-1])
        else:
            # Find largest k with F(k) <= alpha
            best = self._delta_vals[0]
            for k, c in zip(self._delta_vals, self._cdf):
                if c <= alpha:
                    best = k
                else:
                    break
            return int(best)

    @property
    def support(self) -> np.ndarray:
        return self._delta_vals.copy()
