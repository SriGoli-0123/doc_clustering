"""PubMed abstracts via NCBI E-utilities (proposal Sec. 7.2).

Topics are chosen for known conflicting findings; the query topic is the subject label used only for
evaluating clustering, and MeSH major headings are stored too. Respects NCBI's 3 req/s limit
(10 req/s with NCBI_API_KEY).
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET

import requests

from ..utils import DATA_DIR, log, read_jsonl, write_jsonl

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOPICS = {
    "statins_cardiovascular": '(statin[tiab] OR statins[tiab]) AND (cardiovascular[tiab] OR "myocardial infarction"[tiab] OR stroke[tiab])',
    "vitamin_d_supplementation": '"vitamin D"[tiab] AND (supplementation[tiab] OR supplement[tiab])',
    "hormone_replacement": '"hormone replacement therapy"[tiab] OR "menopausal hormone therapy"[tiab]',
    "coffee_health": 'coffee[tiab] AND (mortality[tiab] OR cardiovascular[tiab] OR cancer[tiab] OR diabetes[tiab])',
    "alcohol_cardiovascular": '"alcohol consumption"[tiab] AND (cardiovascular[tiab] OR mortality[tiab])',
    "aspirin_prevention": 'aspirin[tiab] AND ("primary prevention"[tiab] OR "cancer prevention"[tiab])',
    "omega3_fish_oil": '("omega-3"[tiab] OR "fish oil"[tiab]) AND (cardiovascular[tiab] OR mortality[tiab] OR inflammation[tiab])',
    "salt_blood_pressure": '(sodium[tiab] OR salt[tiab]) AND intake[tiab] AND ("blood pressure"[tiab] OR hypertension[tiab])',
    "antioxidant_supplements": '(antioxidant[tiab] OR "vitamin E"[tiab] OR "beta-carotene"[tiab]) AND supplementation[tiab]',
    "metformin_outcomes": 'metformin[tiab] AND (cancer[tiab] OR aging[tiab] OR mortality[tiab] OR dementia[tiab])',
}


def _params(extra: dict) -> dict:
    p = {"tool": "ccst-g16", **extra}
    if os.environ.get("NCBI_API_KEY"):
        p["api_key"] = os.environ["NCBI_API_KEY"]
    if os.environ.get("NCBI_EMAIL"):
        p["email"] = os.environ["NCBI_EMAIL"]
    return p


def _sleep():
    time.sleep(0.12 if os.environ.get("NCBI_API_KEY") else 0.4)


def _get(url: str, params: dict, retries: int = 4) -> requests.Response:
    for k in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"PubMed request failed: {url}")


def esearch(term: str, retmax: int) -> list[str]:
    r = _get(f"{EUTILS}/esearch.fcgi", _params({"db": "pubmed", "term": f"({term}) AND hasabstract AND english[lang]",
                                                 "retmax": retmax, "retmode": "json", "sort": "relevance"}))
    _sleep()
    return r.json()["esearchresult"]["idlist"]


def efetch(pmids: list[str]) -> list[dict]:
    out = []
    for i in range(0, len(pmids), 200):
        r = _get(f"{EUTILS}/efetch.fcgi", _params({"db": "pubmed", "id": ",".join(pmids[i:i + 200]), "retmode": "xml"}))
        _sleep()
        root = ET.fromstring(r.content)
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
            parts = []
            for ab in art.iter("AbstractText"):
                txt = "".join(ab.itertext()).strip()
                if txt:
                    parts.append(txt)
            mesh = [m.findtext("DescriptorName") for m in art.iter("MeshHeading")
                    if m.find("DescriptorName") is not None and m.find("DescriptorName").get("MajorTopicYN") == "Y"]
            year = art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or ""
            if parts:
                out.append({"pmid": pmid, "title": title.strip(), "abstract": " ".join(parts), "mesh_major": mesh,
                            "year": year[:4]})
    return out


def download(per_topic: int = 500, out: str | None = None) -> str:
    out = out or str(DATA_DIR / "raw" / "pubmed.jsonl")
    if os.path.exists(out):
        log.info(f"{out} exists; skipping download")
        return out
    rows, seen = [], set()
    for topic, query in TOPICS.items():
        ids = [i for i in esearch(query, per_topic * 2) if i not in seen][: per_topic]
        recs = efetch(ids)
        for r in recs:
            seen.add(r["pmid"])
            rows.append({"doc_id": f"pm{r['pmid']}", "source": "pubmed", "label": topic, **r})
        log.info(f"pubmed {topic}: {len(recs)} abstracts")
    write_jsonl(rows, out)
    return out


def fetch_by_pmid(pmids: list[str]) -> dict[str, dict]:
    return {r["pmid"]: r for r in efetch(list(dict.fromkeys(pmids)))}


def load(path: str | None = None) -> list[dict]:
    return read_jsonl(path or DATA_DIR / "raw" / "pubmed.jsonl")
