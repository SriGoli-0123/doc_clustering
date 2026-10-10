"""Shared helpers: logging, seeding, timing, memory tracking, JSON/IO."""
from __future__ import annotations

import json
import logging
import os
import random
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CCST_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("CCST_RESULTS_DIR", ROOT / "results"))


def get_logger(name: str = "ccst") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(os.environ.get("CCST_LOGLEVEL", "INFO"))
        logger.propagate = False
    return logger


log = get_logger()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, cls=_Encoder, indent=indent)


def load_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def write_jsonl(rows: Iterable[dict], path: Path | str) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, cls=_Encoder) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (macOS reports bytes, Linux KB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


@contextmanager
def timer(label: str, store: dict | None = None, quiet: bool = False) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if store is not None:
        store[label] = store.get(label, 0.0) + dt
    if not quiet:
        log.info(f"{label}: {dt:.2f}s")


def device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def md_table(rows: list[dict], cols: list[str] | None = None, floatfmt: str = ".3f") -> str:
    """Render a list of dicts as a GitHub markdown table."""
    if not rows:
        return "_(empty)_"
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
