"""arXiv abstracts via the public API (proposal Sec. 7.2).

Categories describe similar mechanisms in different vocabularies (security, quantitative biology,
control, ...). Only papers whose *primary* category matches are kept, so the primary category is a
clean subject label. The API asks for >= 3 s between requests.
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET

import requests

from ..utils import DATA_DIR, log, read_jsonl, write_jsonl

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
CATEGORIES = ["cs.CR", "q-bio.PE", "q-bio.MN", "q-bio.CB", "eess.SY", "physics.soc-ph", "econ.GN", "cs.NE", "nlin.AO",
              "q-bio.NC"]


def _page(cat: str, start: int, n: int) -> list[dict]:
    params = {"search_query": f"cat:{cat}", "start": start, "max_results": n, "sortBy": "submittedDate",
              "sortOrder": "descending"}
    for k in range(5):
        try:
            r = requests.get(API, params=params, timeout=90)
            if r.status_code == 200 and b"<entry>" in r.content:
                break
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    else:
        return []
    root = ET.fromstring(r.content)
    out = []
    for e in root.findall("a:entry", NS):
        prim = e.find("arxiv:primary_category", NS)
        out.append({"arxiv_id": e.findtext("a:id", namespaces=NS).rsplit("/", 1)[-1],
                    "title": " ".join(e.findtext("a:title", namespaces=NS).split()),
                    "abstract": " ".join(e.findtext("a:summary", namespaces=NS).split()),
                    "primary": prim.get("term") if prim is not None else cat,
                    "categories": [c.get("term") for c in e.findall("a:category", NS)],
                    "year": e.findtext("a:published", namespaces=NS)[:4]})
    return out


def download(per_category: int = 500, out: str | None = None) -> str:
    out = out or str(DATA_DIR / "raw" / "arxiv.jsonl")
    rows = [r for r in read_jsonl(out) if r["label"] in CATEGORIES] if os.path.exists(out) else []
    have = {c: sum(r["label"] == c for r in rows) for c in CATEGORIES}
    seen = {r["arxiv_id"] for r in rows}
    todo = [c for c in CATEGORIES if have[c] < per_category // 2]  # resume: only missing / short categories
    if not todo:
        log.info(f"{out} complete; skipping download")
        return out
    rows = [r for r in rows if r["label"] not in todo]
    for cat in todo:
        got, start = [], 0
        while len(got) < per_category and start < per_category * 5:
            page = _page(cat, start, 250)
            time.sleep(3.1)
            if not page:
                break
            for p in page:
                if p["primary"] == cat and p["arxiv_id"] not in seen and len(p["abstract"]) > 200:
                    seen.add(p["arxiv_id"])
                    got.append(p)
            start += 250
        got = got[:per_category]
        rows += [{"doc_id": f"ax{p['arxiv_id']}", "source": "arxiv", "label": cat, **p} for p in got]
        log.info(f"arxiv {cat}: {len(got)} abstracts")
    write_jsonl(rows, out)
    return out


def load(path: str | None = None) -> list[dict]:
    return read_jsonl(path or DATA_DIR / "raw" / "arxiv.jsonl")
