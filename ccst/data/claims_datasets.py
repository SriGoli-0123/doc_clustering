"""Expert-labelled contradiction datasets (proposal Sec. 7.2, stratum 3).

SciFact (Wadden et al., 2020): claims paired with abstracts that SUPPORT / CONTRADICT them, plus
claims whose cited abstract holds no evidence (NEI). Test labels are hidden, so train + dev are used.

ManConCorpus (Alamri & Stevenson, 2016): 259 conclusion claims answering 24 clinical questions with
YES / NO assertions. Two claims answering the same question with opposite assertions are a
potential contradiction; same assertions are agreement. Full abstracts can be fetched by PMID so
that CCST can read the scope conditions (population, dose, duration) stated outside the claim.
"""
from __future__ import annotations

import io
import itertools
import json
import os
import tarfile
import xml.etree.ElementTree as ET

import requests

from ..utils import DATA_DIR, ensure_dir, log, read_jsonl, write_jsonl

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
MANCON_URL = "https://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/corpus.xml"


def download_scifact() -> str:
    out = ensure_dir(DATA_DIR / "raw" / "scifact")
    if (out / "corpus.jsonl").exists():
        return str(out)
    log.info("downloading SciFact")
    r = requests.get(SCIFACT_URL, timeout=120)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name in ("corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"):
                with open(out / name, "wb") as f:
                    f.write(tf.extractfile(m).read())
    return str(out)


def scifact_pairs() -> tuple[dict[str, str], list[dict]]:
    """Returns (doc texts, pairs). Pair label in {agreement, contradiction, incomparable}."""
    d = DATA_DIR / "raw" / "scifact"
    corpus = {str(r["doc_id"]): r for r in read_jsonl(d / "corpus.jsonl")}
    texts: dict[str, str] = {}
    pairs: list[dict] = []
    for split in ("train", "dev"):
        for c in read_jsonl(d / f"claims_{split}.jsonl"):
            cid = f"sfc{c['id']}"
            texts[cid] = c["claim"].rstrip(".") + "."
            ev = c.get("evidence", {}) or {}
            for doc_id in map(str, c.get("cited_doc_ids", [])):
                if doc_id not in corpus:
                    continue
                did = f"sfd{doc_id}"
                doc = corpus[doc_id]
                texts[did] = doc["title"].rstrip(".") + ". " + " ".join(doc["abstract"])
                if doc_id in ev and ev[doc_id]:
                    lab = ev[doc_id][0]["label"]
                    label = "agreement" if lab == "SUPPORT" else "contradiction"
                    rationale = sorted({s for e in ev[doc_id] for s in e["sentences"]})
                else:
                    label, rationale = "incomparable", []
                pairs.append({"pair_id": f"sf{c['id']}_{doc_id}", "a": cid, "b": did, "label": label,
                              "group": cid, "split": split, "rationale": rationale})
    log.info(f"SciFact: {len(pairs)} claim-abstract pairs")
    return texts, pairs


def download_mancon() -> str:
    out = ensure_dir(DATA_DIR / "raw" / "mancon") / "corpus.xml"
    if not out.exists():
        log.info("downloading ManConCorpus")
        r = requests.get(MANCON_URL, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
    return str(out)


def mancon_claims() -> list[dict]:
    root = ET.parse(DATA_DIR / "raw" / "mancon" / "corpus.xml").getroot()
    claims = []
    for qi, rev in enumerate(root.iter("REVIEW")):
        for k, c in enumerate(rev.iter("CLAIM")):
            claims.append({"cid": f"mc{qi}_{k}", "question": c.get("QUESTION"), "qid": qi, "pmid": c.get("PMID"),
                           "assertion": c.get("ASSERTION"), "type": c.get("TYPE"), "text": (c.text or "").strip()})
    return claims


def mancon_pairs(with_abstracts: bool = True) -> tuple[dict[str, str], list[dict], list[dict]]:
    """Pairs within the same clinical question: YS vs NO -> 'conflict', same assertion -> 'agreement'."""
    claims = mancon_claims()
    texts = {c["cid"]: c["text"] for c in claims}
    if with_abstracts:
        cache = DATA_DIR / "raw" / "mancon" / "abstracts.json"
        if cache.exists():
            abstracts = json.loads(cache.read_text())
        else:
            from .pubmed import fetch_by_pmid

            recs = fetch_by_pmid([c["pmid"] for c in claims if c["pmid"]])
            abstracts = {k: v["title"] + " " + v["abstract"] for k, v in recs.items()}
            cache.write_text(json.dumps(abstracts))
        for c in claims:
            ab = abstracts.get(c["pmid"])
            if ab:
                # keep the annotated claim as the final sentence so its relation is always present
                texts[c["cid"]] = ab if c["text"][:40] in ab else ab + " " + c["text"]
    pairs = []
    for qid, grp in itertools.groupby(sorted(claims, key=lambda c: c["qid"]), key=lambda c: c["qid"]):
        grp = list(grp)
        for a, b in itertools.combinations(grp, 2):
            if a["pmid"] == b["pmid"]:
                continue
            pairs.append({"pair_id": f"{a['cid']}_{b['cid']}", "a": a["cid"], "b": b["cid"], "group": qid,
                          "label": "conflict" if a["assertion"] != b["assertion"] else "agreement",
                          "question": a["question"]})
    log.info(f"ManConCorpus: {len(claims)} claims, {len(pairs)} within-question pairs")
    return texts, pairs, claims


def save_pairs(name: str, texts: dict, pairs: list[dict]) -> None:
    out = ensure_dir(DATA_DIR / "processed" / name)
    write_jsonl([{"doc_id": k, "text": v} for k, v in texts.items()], out / "docs.jsonl")
    write_jsonl(pairs, out / "pairs.jsonl")
