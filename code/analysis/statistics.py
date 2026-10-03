"""OLS with two absorbed fixed effects and full-rank cluster correction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import f, t


def codes(values: pd.Series) -> tuple[np.ndarray, int]:
    if values.isna().any():
        raise ValueError("Missing fixed effect or cluster")
    labels, levels = pd.factorize(values, sort=True)
    return labels, len(levels)


def fixed_effect_rank(groups: list[pd.Series]) -> int:
    """Rank of the full two-way indicator matrix, including its intercept."""
    if len(groups) != 2:
        raise ValueError("Exactly two fixed effects are required")
    a, na = codes(groups[0])
    b, nb = codes(groups[1])
    graph = sparse.coo_matrix((np.ones(len(a)), (a, na + b)), shape=(na + nb, na + nb))
    components = connected_components(graph, directed=False, return_labels=False)
    return int(na + nb - components)


def absorb(values: np.ndarray, groups: list[pd.Series], tolerance: float = 1e-11) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite absorption input")
    if result.ndim == 1:
        result = result[:, None]
    info = [codes(g) for g in groups]
    for _ in range(10000):
        before = result.copy()
        for labels, count in info:
            sizes = np.bincount(labels, minlength=count)
            for col in range(result.shape[1]):
                sums = np.bincount(labels, weights=result[:, col], minlength=count)
                result[:, col] -= (sums / sizes)[labels]
        if np.max(np.abs(result - before)) <= tolerance:
            return result
    raise ValueError("Fixed-effect absorption did not converge")


def peel_singletons(
    frame: pd.DataFrame,
    area_column: str = "base_primary_topic_id",
) -> tuple[pd.DataFrame, pd.Series]:
    work = frame.copy()
    removed = pd.Series(0, index=frame.index, dtype=int)
    iteration = 0
    while len(work):
        drop = (
            work.groupby("venue_year").paper_id.transform("size").eq(1)
            | work.groupby(area_column).paper_id.transform("size").eq(1)
        )
        if not drop.any():
            break
        iteration += 1
        removed.loc[work.index[drop]] = iteration
        work = work.loc[~drop]
    return work, removed


@dataclass
class Fit:
    names: list[str]
    beta: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    dropped: list[str]
    n: int
    clusters: int
    fe_rank: int
    full_rank: int
    sse: float
    tss: float
    within_tss: float
    correction: float

    def coefficients(self) -> list[dict]:
        se = np.sqrt(np.diag(self.covariance))
        if not np.isfinite(se).all() or (se <= 0).any():
            raise ValueError("Invalid coefficient standard errors")
        critical = t.ppf(.975, self.clusters - 1)
        return [dict(term=name, coefficient=float(b), standard_error=float(s),
                     ci_low=float(b - critical * s), ci_high=float(b + critical * s),
                     statistic=float(b / s), p_value=float(2 * t.sf(abs(b / s), self.clusters - 1)))
                for name, b, s in zip(self.names, self.beta, se)]

    def wald(self, terms: list[str]) -> tuple[float, float]:
        idx = [self.names.index(term) for term in terms]
        cov = self.covariance[np.ix_(idx, idx)]
        if np.linalg.matrix_rank(cov) != len(idx):
            raise ValueError("Singular Wald covariance")
        stat = float(self.beta[idx] @ np.linalg.solve(cov, self.beta[idx]) / len(idx))
        return stat, float(f.sf(stat, len(idx), self.clusters - 1))


def fit_absorbed(y_raw: np.ndarray, y: np.ndarray, design: pd.DataFrame,
                 cluster: pd.Series, fe_rank: int, required: list[str]) -> Fit:
    """Design and y are already FE-residualized; cluster correction uses full rank."""
    kept: list[str] = []
    dropped: list[str] = []
    for name in design:
        candidate = design[kept + [name]].to_numpy(dtype=float)
        norms = np.linalg.norm(candidate, axis=0)
        if norms[-1] < 1e-9:
            dropped.append(name)
            continue
        normalized = candidate / norms
        if np.linalg.matrix_rank(normalized, tol=1e-9) < len(kept) + 1:
            dropped.append(name)
        else:
            kept.append(name)
    if set(required) - set(kept):
        raise ValueError(f"Unidentified core terms: {set(required) - set(kept)}")
    x = design[kept].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite regression input")
    labels, g = codes(cluster)
    n, k = x.shape
    rank = fe_rank + k
    if g < 2 or n <= rank:
        raise ValueError("Insufficient cluster or residual degrees of freedom")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    scores = np.zeros((g, k))
    np.add.at(scores, labels, x * residuals[:, None])
    correction = g / (g - 1) * (n - 1) / (n - rank)
    covariance = correction * bread @ (scores.T @ scores) @ bread
    covariance = (covariance + covariance.T) / 2
    return Fit(kept, beta, covariance, residuals, dropped, n, g, fe_rank, rank,
               float(residuals @ residuals), float(np.sum((y_raw - np.mean(y_raw)) ** 2)),
               float(y @ y), correction)


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Holm family contains missing p-values")
    order = np.argsort(p)
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p))
    result[order] = adjusted
    return result.tolist()


def compare_nested(restricted: Fit, full: Fit, tested: list[str]) -> dict:
    if restricted.n != full.n or abs(restricted.tss - full.tss) > 1e-9:
        raise ValueError("Nested comparison changed sample/outcome")
    delta = restricted.sse - full.sse
    if delta < -1e-9 * max(1., restricted.sse):
        raise ValueError("Nested SSE increased")
    stat, p = full.wald(tested)
    return dict(delta_r2=delta / full.tss, delta_within_r2=delta / full.within_tss,
                partial_r2=delta / restricted.sse, wald_f=stat, wald_p=p,
                numerator_df=len(tested), denominator_df=full.clusters - 1)
