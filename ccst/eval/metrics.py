"""Metrics from proposal Sec. 7.5 (classification, calibration, selective risk, alignment, retrieval,
clustering, dimensionality-reduction faithfulness) plus bootstrap CIs and paired significance tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    adjusted_rand_score, average_precision_score, f1_score, normalized_mutual_info_score, precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from ..config import RELATION_LABELS


# ----------------------------------------------------------------------------- classification
def classification_report(y_true: list[str], y_pred: list[str], proba: np.ndarray | None = None,
                          labels: tuple = RELATION_LABELS) -> dict:
    labels = list(labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
           "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred)))}
    for k, lab in enumerate(labels):
        out[f"f1_{lab}"] = float(f[k])
        out[f"p_{lab}"] = float(p[k])
        out[f"r_{lab}"] = float(r[k])
    if proba is not None:
        yi = np.array([labels.index(v) for v in y_true])
        out["ece"] = expected_calibration_error(proba, yi)
        out["brier"] = brier(proba, yi)
        ci = labels.index("contradiction")
        if (yi == ci).any():
            out["auprc_contradiction"] = float(average_precision_score(yi == ci, proba[:, ci]))
    return out


def false_contradiction_rate(y_true: list[str], y_pred: list[str], mask: np.ndarray) -> float:
    """Critical safety metric: share of context-shift hard negatives predicted as contradiction."""
    yt = np.array(y_pred)[mask]
    return float(np.mean(yt == "contradiction")) if len(yt) else float("nan")


def expected_calibration_error(P: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = P.max(1)
    acc = (P.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def brier(P: np.ndarray, y: np.ndarray) -> float:
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1
    return float(((P - Y) ** 2).sum(1).mean())


def risk_coverage(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Selective risk as a function of coverage; returns (coverage, risk, AURC)."""
    conf = P.max(1)
    err = (P.argmax(1) != y).astype(float)
    order = np.argsort(-conf)
    cum_err = np.cumsum(err[order])
    n = np.arange(1, len(y) + 1)
    cov = n / len(y)
    risk = cum_err / n
    return cov, risk, float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") else float(np.trapz(risk, cov))


# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    if len(values) == 0:
        return float("nan"), float("nan")
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_metric_ci(y_true, y_pred, fn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        vals.append(fn(list(yt[idx]), list(yp[idx])))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_pvalue(y_true, pred_a, pred_b, fn, n_boot: int = 2000, seed: int = 0) -> float:
    """One-sided p-value that system A is not better than B under metric fn (higher is better)."""
    rng = np.random.default_rng(seed)
    yt, pa, pb = map(np.asarray, (y_true, pred_a, pred_b))
    delta = fn(list(yt), list(pa)) - fn(list(yt), list(pb))
    cnt = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        d = fn(list(yt[idx]), list(pa[idx])) - fn(list(yt[idx]), list(pb[idx]))
        cnt += d <= 0
    return float((cnt + 1) / (n_boot + 1)) if delta > 0 else 1.0


def mcnemar_pvalue(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


# ----------------------------------------------------------------------------- alignment
def alignment_prf(T: np.ndarray, pred_slots1: list, pred_slots2: list, threshold: float = 0.5) -> dict:
    """Node alignment P/R/F1 given gold slot ids for nodes of both graphs (None = no gold slot)."""
    if T.size == 0:
        return {"align_p": 0.0, "align_r": 0.0, "align_f1": 0.0, "gold_mass": 0.0}
    gold = {(i, j) for i, si in enumerate(pred_slots1) for j, sj in enumerate(pred_slots2)
            if si is not None and si == sj}
    rowmax = T.max(1, keepdims=True) + 1e-12
    pred = {(i, j) for i, j in zip(*np.nonzero(T >= threshold * rowmax)) if T[i, j] > 1e-6}
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    gm = sum(T[i, j] for i, j in gold) / max(T.sum(), 1e-12)
    return {"align_p": p, "align_r": r, "align_f1": 2 * p * r / (p + r) if p + r else 0.0, "gold_mass": float(gm)}


# ----------------------------------------------------------------------------- retrieval
def retrieval_metrics(rankings: dict, relevant: dict, ks=(1, 5, 10)) -> dict:
    mrr, rec, ndcg = [], {k: [] for k in ks}, {k: [] for k in ks}
    for q, ranked in rankings.items():
        rel = relevant.get(q, set())
        if not rel:
            continue
        rr = 0.0
        for pos, d in enumerate(ranked, start=1):
            if d in rel:
                rr = 1.0 / pos
                break
        mrr.append(rr)
        for k in ks:
            top = ranked[:k]
            rec[k].append(len(set(top) & rel) / len(rel))
            dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(top) if d in rel)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(rel))))
            ndcg[k].append(dcg / idcg)
    out = {"MRR": float(np.mean(mrr)) if mrr else float("nan")}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(rec[k])) if rec[k] else float("nan")
        out[f"nDCG@{k}"] = float(np.mean(ndcg[k])) if ndcg[k] else float("nan")
    return out


# ----------------------------------------------------------------------------- clustering
def purity(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    tot = 0
    for c in np.unique(yp):
        _, cnt = np.unique(yt[yp == c], return_counts=True)
        tot += cnt.max()
    return tot / len(yt)


def clustering_metrics(y_true, y_pred, X: np.ndarray | None = None, metric: str = "cosine") -> dict:
    out = {"NMI": float(normalized_mutual_info_score(y_true, y_pred)), "ARI": float(adjusted_rand_score(y_true, y_pred)),
           "purity": float(purity(y_true, y_pred)), "n_clusters": int(len(set(y_pred)))}
    if X is not None and 1 < len(set(y_pred)) < len(y_pred):
        n = min(len(X), 5000)
        idx = np.random.default_rng(0).choice(len(X), n, replace=False)
        out["silhouette"] = float(silhouette_score(X[idx], np.asarray(y_pred)[idx], metric=metric))
    return out


# ----------------------------------------------------------------------------- projections
def _knn_sets(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def trustworthiness_continuity(X_high: np.ndarray, X_low: np.ndarray, k: int = 10) -> dict:
    """Venna & Kaski (2001) trustworthiness T(k) and continuity C(k), plus kNN preservation."""
    from sklearn.manifold import trustworthiness

    n = len(X_high)
    T = float(trustworthiness(X_high, X_low, n_neighbors=k))
    C = float(trustworthiness(X_low, X_high, n_neighbors=k))  # continuity = trustworthiness with roles swapped
    nh, nl = _knn_sets(X_high, k), _knn_sets(X_low, k)
    pres = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nh, nl)]))
    return {"trustworthiness": T, "continuity": C, "knn_preservation": pres, "n": n, "k": k}
