"""Held-out extraction check (item 4, docs/parsing_fixes_protocol.md).

Written before the parsing fixes were implemented. Vocabulary and sentence forms are deliberately absent from
both benchmark generators, so improvements here are not template fitting. Gold annotations are by
construction (each template states its polarity; each context sentence lists its gold fields).
"""
from __future__ import annotations

import itertools

import numpy as np

SUBJECTS = ["ketogenic diet", "sodium bicarbonate", "nicotinamide riboside", "cognitive behavioural therapy",
            "low-dose naltrexone", "whole-body vibration", "dual antiplatelet therapy", "intermittent hypoxia",
            "probiotic supplementation", "resistance training"]
OBJECTS = ["serum uric acid", "fasting insulin", "carotid intima thickness", "hepatic fat fraction",
           "depressive symptoms", "bone turnover markers", "plasma homocysteine", "renal clearance", "grip strength",
           "c-reactive protein"]

# (template, gold polarity): +1 increases, -1 decreases, 0 no effect
CLAIM_TEMPLATES = [
    ("{O} rose by {p}% after {S}.", 1),
    ("{S} raised {O} by {p}%.", 1),
    ("Treatment with {S} was followed by a {p}% rise in {O}.", 1),
    ("{O} increased significantly with {S} (p < 0.05).", 1),
    ("Exposure to {S} resulted in higher {O}.", 1),
    ("{S} was linked to elevated {O}.", 1),
    ("{S} administration led to an increase in {O}.", 1),
    ("{O} was {p}% higher among those receiving {S}.", 1),
    ("{O} fell by {p}% after {S}.", -1),
    ("{S} lowered {O} by {p}%.", -1),
    ("{S} produced a marked decline in {O}.", -1),
    ("A significant drop in {O} followed {S}.", -1),
    ("{O} decreased with {S} (p = 0.02).", -1),
    ("{S} was linked to reduced {O}.", -1),
    ("{S} administration led to a reduction in {O}.", -1),
    ("{O} was {p}% lower among those receiving {S}.", -1),
    ("Compared with placebo, {S} reduced {O}.", -1),
    ("{S} attenuated {O}.", -1),
    ("{S} did not alter {O}.", 0),
    ("No significant change in {O} was observed with {S}.", 0),
    ("{O} was unaffected by {S}.", 0),
    ("{S} had no measurable impact on {O}.", 0),
]

# (sentence, gold: age group / duration in days / dose number and unit dimension); None = not stated
CONTEXT_SENTENCES = [
    ("Participants were 65 years or older.", {"age": "elderly", "days": None}),
    ("The mean age was 72 years.", {"age": "elderly", "days": None}),
    ("We enrolled adults aged 40 to 60 years.", {"age": "adult", "days": None}),
    ("Children between 6 and 11 years of age were included.", {"age": "child", "days": None}),
    ("Patients older than 70 years were randomized.", {"age": "elderly", "days": None}),
    ("Women aged ≥70 years were eligible.", {"age": "elderly", "days": None}),
    ("The intervention lasted 12 weeks.", {"age": None, "days": 84}),
    ("Follow-up was 5 years.", {"age": None, "days": 1825}),
    ("Outcomes were assessed during a 6-month follow-up.", {"age": None, "days": 182.4}),
    ("Data were collected over a 2-year period.", {"age": None, "days": 730}),
    ("The drug was given for 10 days.", {"age": None, "days": 10}),
    ("Adults aged 50 years received the drug for 8 weeks.", {"age": "adult", "days": 56}),
    ("In a 3-year trial of patients aged 75 years and over, outcomes improved.", {"age": "elderly", "days": 1095}),
    ("Infants under 1 year of age were followed for 6 months.", {"age": "infant", "days": 182.4}),
    ("Participants took the supplement at a dose of 40 mg/day.", {"dose": 40.0, "unit": "mass"}),
    ("Patients received 2 g per day.", {"dose": 2000.0, "unit": "mass"}),
    ("The regimen was 500 mg twice daily.", {"dose": 500.0, "unit": "mass"}),
    ("A daily dose of 1,000 IU was used.", {"dose": 1000.0, "unit": "iu"}),
    ("Cells were treated with 2,500 µg of the compound.", {"dose": 2.5, "unit": "mass"}),
]


def claim_items(per_template: int = 5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    pairs = list(itertools.product(SUBJECTS, OBJECTS))
    out = []
    for t, pol in CLAIM_TEMPLATES:
        for k in rng.choice(len(pairs), size=per_template, replace=False):
            s, o = pairs[k]
            text = t.format(S=s, O=o, p=int(rng.choice([12, 18, 27, 35])))
            out.append({"text": text[0].upper() + text[1:], "subj": s, "obj": o, "polarity": pol, "template": t})
    return out


def evaluate(extractor=None) -> dict:
    """Claim recall / sign accuracy and context accuracy of the current extractor and context parser."""
    from ..extraction.context import parse_dose, parse_population, parse_time
    from ..extraction.svo import ClaimExtractor, _norm

    if extractor is None:
        from ..embed import get_embedder

        extractor = ClaimExtractor(embedder=get_embedder())
    items = claim_items()
    claims = extractor.extract_docs([(f"h{i}", it["text"]) for i, it in enumerate(items)])
    found = sign_ok = 0
    per_template: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        hit = [c for c in claims[f"h{i}"] if _norm(it["subj"]) in c.subj and _norm(it["obj"]) in c.obj]
        ok = bool(hit)
        found += ok
        pol = 0 if (hit and hit[0].null_effect) else (hit[0].polarity if hit else None)
        sign_ok += ok and pol == it["polarity"]
        per_template.setdefault(it["template"], []).append(int(ok and pol == it["polarity"]))
    ctx_rows = []
    age_ok = age_n = days_ok = days_n = dose_ok = dose_n = false_time = 0
    for sent, gold in CONTEXT_SENTENCES:
        pop, tm, dose = parse_population(sent), parse_time(sent), parse_dose(sent)
        got_age = (pop or {}).get("quals", {}).get("age")
        got_days = (tm or {}).get("days")
        if "age" in gold:
            age_n += gold["age"] is not None
            age_ok += gold["age"] is not None and got_age == gold["age"]
            if gold["days"] is None:
                false_time += got_days is not None
            else:
                days_n += 1
                days_ok += got_days is not None and abs(got_days - gold["days"]) / gold["days"] < 0.05
        if "dose" in gold:
            dose_n += 1
            dose_ok += bool(dose) and "num" in dose and abs(dose["num"] - gold["dose"]) < 1e-6 and dose["unit"] == gold["unit"]
        ctx_rows.append({"sentence": sent, "age": got_age, "days": got_days, "dose": (dose or {}).get("num")})
    n = len(items)
    return {"claim_recall": found / n, "claim_sign_accuracy": sign_ok / max(found, 1),
            "claim_correct": sign_ok / n, "n_claims": n,
            "age_accuracy": age_ok / max(age_n, 1), "duration_accuracy": days_ok / max(days_n, 1),
            "age_misread_as_duration": false_time, "dose_accuracy": dose_ok / max(dose_n, 1),
            "per_template_correct": {t: float(np.mean(v)) for t, v in per_template.items()}, "context_rows": ctx_rows}
