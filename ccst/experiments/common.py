"""Shared experiment machinery: CCST featurisation of pairs and grouped cross-validation."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from ..config import RELATION_LABELS, CCSTConfig, ContextConfig
from ..relation.classifier import RelationClassifier
from ..relation.features import FEATURE_NAMES, domain_separation, pair_features
from ..relation.rules import rule_predict, tune_rules
from ..transport.ccst import CCSTSolver


def _solve_chunk(chunk, graphs, metas, cfg, ccfg, keep_results):
    solver = CCSTSolver(cfg, ccfg)
    out = []
    for a, b in chunk:
        t0 = time.perf_counter()
        res = solver.solve(graphs[a], graphs[b])
        f = pair_features(graphs[a], graphs[b], res, cfg, ccfg)
        f["S_domain"] = domain_separation(metas.get(a, {}), metas.get(b, {}))
        f["seconds"] = time.perf_counter() - t0
        f["iters"] = res.iters
        if not keep_results:
            f.pop("_evidence", None)
            res = None
        out.append((f, res))
    return out


def featurize(pair_list: list[tuple[str, str]], graphs: dict, metas: dict, cfg: CCSTConfig,
              ccfg: ContextConfig | None = None, n_jobs: int = 1, keep_results: bool = False):
    """Solve CCST for each (a, b) and return (feature dicts, results or None)."""
    ccfg = ccfg or ContextConfig()
    if n_jobs == 1 or keep_results:
        out = _solve_chunk(pair_list, graphs, metas, cfg, ccfg, keep_results)
    else:
        k = max(1, len(pair_list) // (n_jobs * 4))
        chunks = [pair_list[i:i + k] for i in range(0, len(pair_list), k)]
        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_solve_chunk)(c, {x: graphs[x] for p in c for x in p}, {x: metas.get(x, {}) for p in c for x in p},
                                  cfg, ccfg, False) for c in chunks)
        out = [x for p in parts for x in p]
    return [f for f, _ in out], [r for _, r in out]


def feature_matrix(feats: list[dict], names: list[str] | None = None) -> np.ndarray:
    names = names or FEATURE_NAMES + ["S_domain"]
    return np.array([[float(f.get(n, 0.0)) for n in names] for f in feats])


def ccst_variant(base: CCSTConfig, **kw) -> CCSTConfig:
    return replace(base, **kw)


def cross_validate(X: np.ndarray, y: list[str], groups: np.ndarray, n_splits: int = 5, seed: int = 0,
                   calibration: str = "temperature", risk_target: float = 0.1, feats: list[dict] | None = None,
                   rule_mode: bool = False, labels: tuple = RELATION_LABELS) -> dict:
    """Grouped K-fold. Inside each training portion, 25% of the groups are held out to fit the
    temperature / abstention threshold (and rule thresholds), so the test fold is never touched."""
    y = np.asarray(y)
    n = len(y)
    proba = np.zeros((n, len(labels)))
    pred = np.empty(n, dtype=object)
    pred_abst = np.empty(n, dtype=object)
    fold_of = np.zeros(n, int)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        fold_of[te] = fold
        if rule_mode:
            Ftr = [_with_sigma(feats[i], feats, tr) for i in tr]
            th = tune_rules(Ftr, list(y[tr]), seed=seed + fold)
            for i in te:
                p = rule_predict(_with_sigma(feats[i], feats, tr), th)
                pred[i] = pred_abst[i] = p
                proba[i, RELATION_LABELS.index(p)] = 1.0
            continue
        g_tr = np.unique(groups[tr])
        rng = np.random.default_rng(seed + fold)
        cal_groups = set(rng.choice(g_tr, size=max(1, len(g_tr) // 4), replace=False))
        cal = np.array([i for i in tr if groups[i] in cal_groups])
        fit = np.array([i for i in tr if groups[i] not in cal_groups])
        clf = RelationClassifier(calibration=calibration, risk_target=risk_target, seed=seed, labels=labels)
        clf.fit(X[fit], list(y[fit]))
        clf.calibrate(X[cal], list(y[cal]))
        P = clf.predict_proba(X[te])
        proba[te] = P
        pred[te] = [labels[k] for k in P.argmax(1)]
        pred_abst[te] = [labels[k] if c >= clf.tau else "abstain" for k, c in zip(P.argmax(1), P.max(1))]
    return {"proba": proba, "pred": list(pred), "pred_abstain": list(pred_abst), "fold": fold_of}


_SIGMA_CACHE: dict = {}


def _with_sigma(f: dict, feats: list[dict], tr: np.ndarray) -> dict:
    key = (id(feats), len(tr), int(tr[0]), int(tr[-1]))
    if key not in _SIGMA_CACHE:
        _SIGMA_CACHE[key] = float(np.median([feats[i]["D_mech"] for i in tr if np.isfinite(feats[i]["D_mech"])]) or 1.0)
    sigma = max(_SIGMA_CACHE[key], 1e-3)
    return {**f, "S_mech": float(np.exp(-f["D_mech"] / sigma))}
