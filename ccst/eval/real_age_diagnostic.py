"""Diagnostic for item 4: how often does the duration parser fire on an *age* phrase in real abstracts?
Counts durations whose matched text sits next to an age cue (aged / age of / ≥ / older than / or older / of age)."""
from __future__ import annotations

import re

from ..extraction.context import _TIME_RE, parse_time

AGE_CUE = re.compile(r"(?:aged|age[ds]?\s+(?:of\s+)?|mean age(?:\s+was)?|[≥>]=?|older than|over|between)\s*$", re.I)
AGE_TAIL = re.compile(r"^\s*(?:or older|or over|and older|and over|of age|old|or more of age)", re.I)


def count(texts: list[str]) -> dict:
    fired = misread = 0
    for t in texts:
        for m in _TIME_RE.finditer(t):
            if not parse_time(t[m.start():m.end()]):
                continue
            fired += 1
            before, after = t[max(0, m.start() - 25):m.start()], t[m.end():m.end() + 20]
            misread += bool(AGE_CUE.search(before) or AGE_TAIL.search(after))
    return {"duration_mentions": fired, "age_cued_mentions": misread}


def first_duration_is_age(texts: list[str]) -> float:
    """Share of abstracts whose *document-level* time value comes from an age phrase."""
    bad = tot = 0
    for t in texts:
        m = _TIME_RE.search(t)
        if not m:
            continue
        tm = parse_time(t)
        if tm is None:
            continue
        tot += 1
        before, after = t[max(0, m.start() - 25):m.start()], t[m.end():m.end() + 20]
        bad += bool(AGE_CUE.search(before) or AGE_TAIL.search(after))
    return bad / max(tot, 1)
