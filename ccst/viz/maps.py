"""Corpus maps (proposal Sec. 4.10, Step J): PCA(50) -> t-SNE in 2D/3D with a perplexity sweep chosen
by trustworthiness; maps coloured by predicted cluster and by true subject label; relation links
(analogy, contradiction, dashed heterogeneity); a 3D map. t-SNE preserves local neighbourhoods only,
so the maps are read for local structure and trustworthiness / continuity are reported.

Colour: categorical identity uses the validated 8-hue order; label sets larger than 8 are
composite-encoded (hue x marker shape) and every subject label is also direct-labelled at its
centroid, so identity never relies on colour alone.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..eval.metrics import trustworthiness_continuity
from .plots import GRID, MUTED, SERIES, SURFACE, TEXT, TEXT2, _save, plt

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def tsne(X: np.ndarray, dims: int = 2, perplexity: float = 30, seed: int = 0, pca_dims: int = 50) -> np.ndarray:
    Z = PCA(n_components=min(pca_dims, X.shape[1], len(X) - 1), random_state=seed).fit_transform(X)
    return TSNE(n_components=dims, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed,
                max_iter=1000).fit_transform(Z)


def perplexity_sweep(X: np.ndarray, perplexities=(15, 30, 50), dims: int = 2, seed: int = 0, k: int = 10,
                     max_eval: int = 3000) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    ev = rng.choice(len(X), size=min(max_eval, len(X)), replace=False)
    best, best_t, rows = None, -1, []
    for p in perplexities:
        Y = tsne(X, dims, p, seed)
        m = trustworthiness_continuity(X[ev], Y[ev], k=k)
        rows.append({"perplexity": p, "dims": dims, **m})
        if m["trustworthiness"] > best_t:
            best, best_t = Y, m["trustworthiness"]
    return best, rows


def _style(i: int) -> tuple[str, str]:
    return SERIES[i % len(SERIES)], MARKERS[(i // len(SERIES)) % len(MARKERS)]


def _scatter_labels(ax, Y, labels, title, annotate=True, s=6):
    labs = sorted(set(labels), key=lambda x: str(x))
    lab_arr = np.asarray(labels)
    for i, lab in enumerate(labs):
        m = lab_arr == lab
        c, mk = _style(i)
        ax.scatter(Y[m, 0], Y[m, 1], s=s, c=c, marker=mk, linewidths=0, alpha=0.75, label=str(lab))
        if annotate:
            cx, cy = np.median(Y[m, 0]), np.median(Y[m, 1])
            ax.text(cx, cy, str(lab), fontsize=7, color=TEXT, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec=GRID, alpha=0.85))
    ax.set_title(title, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def side_by_side(Y: np.ndarray, pred, true, path: str, title_pred: str = "predicted cluster",
                 title_true: str = "true subject label") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    _scatter_labels(axes[0], Y, pred, title_pred, annotate=len(set(pred)) <= 30)
    _scatter_labels(axes[1], Y, true, title_true)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=2, fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def relation_links(Y: np.ndarray, links: dict, path: str, title: str, background=None) -> str:
    """links: {'analogy': [(i,j,w)], 'contradiction': [...], 'heterogeneity': [...]}"""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(Y[:, 0], Y[:, 1], s=4, c=MUTED, linewidths=0, alpha=0.5)
    styles = {"analogy": (SERIES[0], "-"), "contradiction": (SERIES[1], "-"), "heterogeneity": (SERIES[2], "--")}
    for kind, edges in links.items():
        col, ls = styles[kind]
        for n, (i, j, w) in enumerate(edges):
            ax.plot([Y[i, 0], Y[j, 0]], [Y[i, 1], Y[j, 1]], color=col, linestyle=ls, linewidth=0.6 + 1.2 * w,
                    alpha=0.75, label=f"{kind} ({len(edges)})" if n == 0 else None)
    ax.set_title(title, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(loc="upper right")
    return _save(fig, path)


def map_3d(Y3: np.ndarray, labels, path: str, title: str) -> str:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    labs = sorted(set(labels), key=str)
    arr = np.asarray(labels)
    for i, lab in enumerate(labs):
        m = arr == lab
        c, mk = _style(i)
        ax.scatter(Y3[m, 0], Y3[m, 1], Y3[m, 2], s=4, c=c, marker=mk, alpha=0.7, label=str(lab), linewidths=0)
    ax.set_title(title, loc="left", color=TEXT)
    ax.legend(loc="upper left", fontsize=7, markerscale=2.5, bbox_to_anchor=(1.02, 1))
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticklabels([])
        a.label.set_color(TEXT2)
    return _save(fig, path)
