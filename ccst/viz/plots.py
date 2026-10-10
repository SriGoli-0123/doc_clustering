"""Static Matplotlib charts for every experiment (proposal Sec. 4.10 supporting charts).

Palette: a validated categorical order (fixed, never cycled), one-hue blue ramp for magnitudes,
blue<->red diverging with a neutral midpoint. Many-method comparisons use a single highlight hue
for the proposed method and neutral gray for baselines instead of >8 categorical hues.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..utils import ensure_dir  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
HIGHLIGHT, MUTED = "#2a78d6", "#a3a29d"
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#b4302f"])


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT, "axes.titlesize": 12,
        "axes.titleweight": "semibold", "axes.labelsize": 10, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 9, "font.family": "DejaVu Sans", "lines.linewidth": 2.0, "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


setup_style()


def _save(fig, path) -> str:
    ensure_dir(Path(path).parent)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def bar_ci(names: list[str], values, lo=None, hi=None, highlight: set | None = None, xlabel: str = "",
           title: str = "", path: str = "bar.png", fmt: str = "{:.3f}", xlim: tuple | None = None) -> str:
    """Horizontal bars sorted by value; proposed method(s) in the highlight hue, baselines in gray."""
    values = np.asarray(values, float)
    order = np.argsort(values)
    highlight = highlight or set()
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(names) + 1.2))
    ys = np.arange(len(names))
    cols = [HIGHLIGHT if names[i] in highlight else MUTED for i in order]
    ax.barh(ys, values[order], color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
    if lo is not None and hi is not None:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        ax.errorbar(values[order], ys, xerr=[values[order] - lo[order], hi[order] - values[order]], fmt="none",
                    ecolor=TEXT2, elinewidth=1, capsize=2)
    ax.set_yticks(ys, [names[i] for i in order])
    end = np.maximum(values, hi) if (lo is not None and hi is not None) else values
    for y, i in zip(ys, order):  # label sits past the CI whisker so the two never collide
        ax.text(end[i] + (0.012 * (xlim[1] - xlim[0]) if xlim else 0.005), y, fmt.format(values[i]), va="center",
                fontsize=8, color=TEXT)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    if xlim:
        ax.set_xlim(*xlim)
    return _save(fig, path)


def grouped_bars(categories: list[str], series: dict, ylabel: str = "", title: str = "", path: str = "g.png",
                 ylim: tuple | None = None) -> str:
    k = len(series)
    assert k <= len(SERIES), "fold extra series into small multiples"
    x = np.arange(len(categories))
    w = 0.8 / k
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(categories) + 2), 3.8))
    for s, (name, vals) in enumerate(series.items()):
        ax.bar(x + (s - (k - 1) / 2) * w, vals, w * 0.92, label=name, color=SERIES[s], edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, categories, rotation=20 if len(categories) > 5 else 0, ha="right" if len(categories) > 5 else "center")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    rotated = len(categories) > 5 or max(len(c) for c in categories) > 14
    if rotated:  # keep long category names clear of the legend
        ax.set_xticks(x, categories, rotation=25, ha="right")
    ax.legend(ncol=min(k, 4), loc="upper center", bbox_to_anchor=(0.5, -0.38 if rotated else -0.1))
    return _save(fig, path)


def confusion(cm: np.ndarray, labels: list[str], title: str, path: str) -> str:
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cmn, cmap=SEQ, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.55 else TEXT)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def lines(x, ys: dict, xlabel: str, ylabel: str, title: str, path: str, logx: bool = False, logy: bool = False,
          markers: bool = True, bands: dict | None = None, ylim: tuple | None = None, ref_line: float | None = None) -> str:
    assert len(ys) <= len(SERIES)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s, (name, y) in enumerate(ys.items()):
        xx = x[name] if isinstance(x, dict) else x
        ax.plot(xx, y, label=name, color=SERIES[s], marker="o" if markers else None, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(xx, lo, hi, color=SERIES[s], alpha=0.12, linewidth=0)
    if ref_line is not None:
        ax.axhline(ref_line, color=TEXT2, linewidth=1, linestyle="--")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="best")
    return _save(fig, path)


def reliability(P: np.ndarray, y: np.ndarray, title: str, path: str, n_bins: int = 10) -> str:
    conf, acc = P.max(1), (P.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    mids, accs, cnt = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            mids.append(conf[m].mean())
            accs.append(acc[m].mean())
            cnt.append(m.sum())
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color=TEXT2, linewidth=1, linestyle="--", label="perfect calibration")
    ax.plot(mids, accs, color=HIGHLIGHT, marker="o", markersize=5, label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    return _save(fig, path)


def heatmap(M: np.ndarray, rows: list[str], cols: list[str], title: str, path: str, fmt: str = "{:.0f}",
            cmap=None, vmin=None, vmax=None, xlabel: str = "", ylabel: str = "") -> str:
    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 2.5, 0.42 * len(rows) + 1.6))
    im = ax.imshow(M, cmap=cmap or SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    hi = np.nanmax(M) if M.size else 1
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 * hi else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    return _save(fig, path)


def topword_bars(topics: list[list[tuple[str, float]]], title: str, path: str, ncols: int = 5) -> str:
    n = len(topics)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.3 * nrows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        words = topics[k][:8][::-1]
        ax.barh([w for w, _ in words], [p for _, p in words], color=HIGHLIGHT, height=0.6)
        ax.set_title(f"topic {k}", loc="left", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold", color=TEXT)
    fig.tight_layout()
    return _save(fig, path)
