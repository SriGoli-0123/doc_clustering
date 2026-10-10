"""Multinomial logistic-regression relation model with calibration and selective abstention.

* Calibration: temperature scaling (Guo et al., 2017) or one-vs-rest isotonic regression.
* Abstention: selective-risk control (Geifman & El-Yaniv, 2017): pick the lowest confidence
  threshold whose Clopper-Pearson upper bound on the selective error is <= the target risk.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..config import RELATION_LABELS


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(beta_dist.ppf(1 - delta, k + 1, n - k))


class RelationClassifier:
    def __init__(self, C: float = 1.0, calibration: str = "temperature", risk_target: float = 0.10,
                 delta: float = 0.1, labels: tuple = RELATION_LABELS, seed: int = 0):
        self.labels = list(labels)
        self.C, self.calibration, self.risk_target, self.delta = C, calibration, risk_target, delta
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=C, max_iter=5000, class_weight="balanced", random_state=seed)
        self.temperature = 1.0
        self.iso: list[IsotonicRegression] | None = None
        self.tau = 0.0  # abstention threshold on max probability

    def fit(self, X: np.ndarray, y: list[str]) -> "RelationClassifier":
        yi = np.array([self.labels.index(v) for v in y])
        Xs = self.scaler.fit_transform(np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0))
        self.lr.fit(Xs, yi)
        return self

    def _logits(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0))
        z = np.full((len(X), len(self.labels)), -1e4)
        dec = self.lr.decision_function(Xs)
        if dec.ndim == 1:  # binary: sigmoid(d) == softmax([0, d])[1]
            dec = np.stack([np.zeros_like(dec), dec], 1)
        z[:, self.lr.classes_] = dec
        return z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self._logits(X)
        if self.calibration == "isotonic" and self.iso is not None:
            P = softmax(z)
            Q = np.stack([iso.predict(P[:, k]) for k, iso in enumerate(self.iso)], 1) + 1e-6
            return Q / Q.sum(1, keepdims=True)
        return softmax(z / self.temperature)

    def calibrate(self, X: np.ndarray, y: list[str]) -> "RelationClassifier":
        yi = np.array([self.labels.index(v) for v in y])
        z = self._logits(X)
        if self.calibration == "temperature":
            def nll(t):
                P = softmax(z / t)
                return -np.log(P[np.arange(len(yi)), yi] + 1e-12).mean()

            self.temperature = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)
        elif self.calibration == "isotonic":
            P = softmax(z)
            self.iso = [IsotonicRegression(out_of_bounds="clip").fit(P[:, k], (yi == k).astype(float))
                        for k in range(len(self.labels))]
        P = self.predict_proba(X)
        self.tau = self._risk_threshold(P, yi)
        return self

    def _risk_threshold(self, P: np.ndarray, yi: np.ndarray) -> float:
        conf = P.max(1)
        err = (P.argmax(1) != yi).astype(int)
        order = np.argsort(-conf)
        best = 1.0 + 1e-9
        k = 0
        for n_acc, i in enumerate(order, start=1):
            k += err[i]
            if clopper_pearson_upper(k, n_acc, self.delta) <= self.risk_target:
                best = conf[i]
        return float(best)

    def predict(self, X: np.ndarray, abstain: bool = False) -> list[str]:
        P = self.predict_proba(X)
        pred = [self.labels[i] for i in P.argmax(1)]
        if abstain:
            pred = [p if c >= self.tau else "abstain" for p, c in zip(pred, P.max(1))]
        return pred
