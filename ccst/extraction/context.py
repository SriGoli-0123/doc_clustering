"""Rule-based scope-condition and epistemic-status extraction (proposal Sec. 4.1-4.2).

Each context dimension is parsed into a small typed value so that *value compatibility*
(equivalent / nested / disjoint / unknown) can be computed without treating, e.g., "mice" and
"adult patients" as interchangeable. Missing fields stay ``None`` (=> unknown, never a match).
"""
from __future__ import annotations

import math
import re

WORDNUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "twenty-four": 24,
    "thirty": 30, "forty": 40, "forty-eight": 48, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
    "a": 1, "an": 1, "single": 1, "several": 3,
}
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?|" + "|".join(sorted((k for k in WORDNUM if len(k) > 2), key=len, reverse=True)) + r")"

# ---------------------------------------------------------------- population / system
POP_KIND = {
    "human": "patients adults adult children child infants infant neonates newborns women woman men man participants "
    "subjects volunteers individuals people persons smokers athletes students veterans residents elderly adolescents "
    "teenagers pregnant mothers girls boys survivors outpatients inpatients workers employees population populations cohorts",
    "animal": "mice mouse rats rat rodents animals dogs pigs piglets zebrafish primates monkeys rabbits sheep cattle cows "
    "hamsters",
    "cell": "cells cell lines cultures organoids tissues tissue fibroblasts neurons hepatocytes myocytes macrophages",
    "system": "networks network hosts servers devices routers nodes systems clusters endpoints machines vehicles robots "
    "drones reactors grids turbines microgrids controllers datacenters data centers deployments platforms "
    "applications services infrastructures plants processes pipelines satellites quadrotors manipulators",
    "econ": "firms firm banks households markets countries economies companies investors consumers economy regions "
    "municipalities borrowers lenders traders exporters enterprises",
    "eco": "forests forest lakes lake species colonies fish birds insects crops soils ecosystems reefs wetlands "
    "grasslands rivers streams herbivores predators prey pollinators plankton communities meadows",
}
HEAD2KIND = {w: k for k, ws in POP_KIND.items() for w in ws.split()}
# Mutually exclusive qualifier groups: conflicting values => materially incompatible populations.
QUALIFIER_GROUPS = {
    "age": {"infant": "infant infants neonates neonatal newborn newborns", "child": "child children pediatric paediatric "
            "boys girls school-age", "adolescent": "adolescent adolescents teenagers teenage youth",
            "adult": "adult adults middle-aged", "elderly": "elderly older aged geriatric senior seniors"},
    "sex": {"female": "women woman female females girls mothers pregnant", "male": "men man male males boys"},
    "health": {"healthy": "healthy", "diseased": "diabetic obese hypertensive critically ill septic cancer "
               "cardiac infected chronic"},
    "scale": {"small": "small small-scale micro tiny", "large": "large large-scale massive big"},
    "sector": {"enterprise": "enterprise corporate", "home": "home residential household",
               "industrial": "industrial manufacturing", "public": "public municipal government"},
    "income": {"low": "low-income poor developing", "high": "high-income wealthy advanced developed"},
    "climate": {"tropical": "tropical", "temperate": "temperate", "arctic": "arctic polar boreal",
                "arid": "arid desert dryland"},
    "species": {"mouse": "mice mouse murine", "rat": "rats rat", "human": "human humans patients"},
}
QUAL2GROUP = {}
for _g, _vals in QUALIFIER_GROUPS.items():
    for _v, _words in _vals.items():
        for _w in _words.split():
            QUAL2GROUP.setdefault(_w, []).append((_g, _v))

_POP_RE = re.compile(
    r"\b(?:in|among|of|on|for|from|across|using)\s+((?:[\w\-]+\s+){0,4}?(?:"
    + "|".join(sorted(map(re.escape, HEAD2KIND), key=len, reverse=True))
    + r"))\b",
    re.I,
)
_AGE_RE = re.compile(r"\baged\s+(\d+)|(\d+)\s*(?:years? old|years of age|-year-old)", re.I)
# Item 4: age expressions that the duration parser used to steal ("65 years or older", "mean age 72 years").
_AGE_PATTERNS = [
    (re.compile(r"\b(?:aged|ages?)\s+(\d+)\s*(?:to|-|–|and)\s*(\d+)", re.I), "range"),
    (re.compile(r"\bbetween\s+(\d+)\s+and\s+(\d+)\s+years?\s+of\s+age", re.I), "range"),
    (re.compile(r"(?:\baged\s*|\b(?:patients|participants|adults|individuals|people|women|men|subjects|persons)\s+)?"
                r"(?:≥|>=|>|\bolder than|(?<=aged )over|(?<=aged)over)\s*(\d+)\s*(?:years?|y)\b", re.I), "above"),
    (re.compile(r"\b(?:aged|patients|participants|adults|individuals|people|women|men|subjects|persons)\s+(?:over|above)\s+"
                r"(\d+)\s*(?:years?|y)\b", re.I), "above"),
    (re.compile(r"\b(\d+)\s*years?\s*(?:or older|or over|and older|and over|or more)", re.I), "above"),
    (re.compile(r"\bmean age\s*(?:was|of|:|=)?\s*(\d+(?:\.\d+)?)", re.I), "point"),
    (re.compile(r"\b(?:under|below)\s*(\d+)\s*years?\s*(?:of age|old)|\b(?:younger than|<)\s*(\d+)\s*years?", re.I),
     "below"),
    (re.compile(r"\baged\s+(\d+)", re.I), "point"),
    (re.compile(r"\b(\d+)\s*(?:years? old|years? of age|-year-old)", re.I), "point"),
]
AGE_CONTEXT = re.compile(r"(?:\baged(?:\s+(?:over|under|above))?|\bages?|\bage of|\bmean age(?:\s+was)?|≥|>=|>|"
                         r"\bolder than|\byounger than|\bbetween|<|\b(?:patients|participants|adults|individuals|people|"
                         r"women|men|subjects|persons)\s+(?:over|under|above))\s*$", re.I)
AGE_TAIL = re.compile(r"^\s*(?:or older|or over|and older|and over|or more|of age|old\b|-year-old)", re.I)


def _age_value(text: str) -> tuple[float, str] | None:
    for rx, kind in _AGE_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        nums = [float(g) for g in m.groups() if g]
        if kind == "range":
            return (nums[0] + nums[1]) / 2, m.group(0)
        if kind == "above":
            return nums[0] + 5, m.group(0)
        if kind == "below":
            return nums[0] / 2, m.group(0)
        return nums[0], m.group(0)
    return None


def _age_group(age: float) -> str:
    if age < 2:
        return "infant"
    if age < 12:
        return "child"
    if age < 18:
        return "adolescent"
    if age < 65:
        return "adult"
    return "elderly"


def parse_population(text: str) -> dict | None:
    m = _POP_RE.search(text)
    age = _age_value(text)
    if not m and not age:
        return None
    phrase = m.group(1).lower() if m else ""
    words = re.findall(r"[\w\-]+", phrase)
    head = words[-1] if words else ""
    kind = HEAD2KIND.get(head)
    if kind is None and len(words) >= 2:
        kind = HEAD2KIND.get(" ".join(words[-2:]))
    quals: dict[str, str] = {}
    for w in words:
        for g, v in QUAL2GROUP.get(w, []):
            quals.setdefault(g, v)
    if age:
        quals["age"] = _age_group(age[0])
    stop = {"the", "a", "an", "of", "with", "and", "in"}
    return {"text": phrase or age[1], "kind": kind or "human", "quals": quals,
            "tokens": sorted(set(words) - stop)}


# ---------------------------------------------------------------- dose / intensity
UNIT_CANON = {
    "mg": ("mass", 1.0), "milligrams": ("mass", 1.0), "g": ("mass", 1000.0), "grams": ("mass", 1000.0),
    "µg": ("mass", 1e-3), "ug": ("mass", 1e-3), "mcg": ("mass", 1e-3), "micrograms": ("mass", 1e-3),
    "mg/kg": ("mass_per_kg", 1.0), "iu": ("iu", 1.0), "units": ("iu", 1.0), "ml": ("volume", 1.0),
    "mmol": ("mmol", 1.0), "%": ("percent", 1.0), "percent": ("percent", 1.0), "hz": ("freq", 1.0),
    "khz": ("freq", 1e3), "mhz": ("freq", 1e6), "gbps": ("rate", 1e3), "mbps": ("rate", 1.0),
    "kw": ("power", 1.0), "mw": ("power", 1e3), "w": ("power", 1e-3), "ppm": ("ppm", 1.0),
    "usd": ("money", 1.0), "dollars": ("money", 1.0), "basis points": ("bp", 1.0), "bps": ("bp", 1.0),
    "requests per second": ("rps", 1.0), "packets per second": ("rps", 1.0), "kg/ha": ("kg_ha", 1.0),
    "°c": ("temp", 1.0), "degrees": ("temp", 1.0), "mg/l": ("conc", 1.0),
    "nmol/l": ("molar", 1e-3), "µmol/l": ("molar", 1.0), "umol/l": ("molar", 1.0), "mmol/l": ("molar", 1e3),
}
_DOSE_RE = re.compile(
    _NUM + r"\s*(" + "|".join(sorted(map(re.escape, UNIT_CANON), key=len, reverse=True))
    + r")(?:/(?:day|d|week|wk|h|dose))?(?![\w/])", re.I
)
_LEVEL_RE = re.compile(r"\b(low|moderate|medium|high|heavy|light|mild|intense)[\s\-]+"
                       r"(dose|dosage|intensity|load|rate|concentration|exposure|level|frequency|volume|traffic)", re.I)
LEVELS = {"low": 0, "light": 0, "mild": 0, "moderate": 1, "medium": 1, "high": 2, "heavy": 2, "intense": 2}


def _canon_unit(u: str):
    """Unit lookup robust to the two Unicode 'micro' characters (U+00B5 vs U+03BC) and to 'u' spellings."""
    u = u.lower().replace("\u03bc", "\u00b5")
    return UNIT_CANON.get(u) or UNIT_CANON.get(u.replace("\u00b5", "u"))


def _num(s: str) -> float:
    s = s.lower()
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):  # thousands separator: 1,000 -> 1000
        s = s.replace(",", "")
    s = s.replace(",", ".")
    return float(WORDNUM.get(s, s)) if s in WORDNUM else float(s)


def parse_dose(text: str) -> dict | None:
    m = _DOSE_RE.search(text)
    lm = _LEVEL_RE.search(text)
    if not m and not lm:
        return None
    out: dict = {"text": (m.group(0) if m else lm.group(0)).lower()}
    if m:
        unit = _canon_unit(m.group(2))
        if unit is not None:
            dim, scale = unit
            out.update(num=_num(m.group(1)) * scale, unit=dim)
    if lm:
        out["level"] = LEVELS[lm.group(1).lower()]
    return out


# ---------------------------------------------------------------- time horizon
UNIT_DAYS = {"second": 1 / 86400, "minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30.4,
             "year": 365.0, "decade": 3650.0, "cycle": 1.0, "epoch": 1.0, "quarter": 91.0}
_TIME_RE = re.compile(r"(?<!aged )(?<!age )" + _NUM + r"[\s\-]*(second|minute|hour|day|week|month|year|decade|quarter)s?\b"
                      r"(?!\s*(?:old|of age))", re.I)
_TIME_Q_RE = re.compile(r"\b(short[\s\-]term|acute|immediate|long[\s\-]term|chronic|prolonged|sustained|lifelong)\b", re.I)
TIME_Q = {"short": 0, "acute": 0, "immediate": 0, "long": 1, "chronic": 1, "prolonged": 1, "sustained": 1,
          "lifelong": 1}


def _is_age(text: str, m) -> bool:
    return bool(AGE_CONTEXT.search(text[max(0, m.start() - 25):m.start()]) or AGE_TAIL.search(text[m.end():m.end() + 20]))


def parse_time(text: str) -> dict | None:
    m = next((x for x in _TIME_RE.finditer(text) if not _is_age(text, x)), None)
    qm = _TIME_Q_RE.search(text)
    if not m and not qm:
        return None
    out: dict = {"text": (m.group(0) if m else qm.group(0)).lower()}
    if m:
        out["days"] = _num(m.group(1)) * UNIT_DAYS[m.group(2).lower()]
    if qm:
        out["horizon"] = TIME_Q[re.split(r"[\s\-]", qm.group(1).lower())[0]]
    elif m:
        out["horizon"] = 0 if out["days"] < 28 else 1
    return out


# ---------------------------------------------------------------- environment / setting
ENV_PATTERNS = [
    ("in_vitro", r"\bin vitro\b|\bcell cultures?\b|\bpetri\b"),
    ("in_silico", r"\bin silico\b|\bsimulat(?:ed|ion) environment\b|\bvirtual environment\b"),
    ("in_vivo", r"\bin vivo\b|\banimal model\b"),
    ("clinical", r"\bclinical setting\b|\bhospitals?\b|\bintensive care\b|\bicu\b|\boutpatient\b|\binpatient\b|\bclinics?\b"),
    ("community", r"\bcommunity[\s\-]dwelling\b|\bcommunity setting\b|\bprimary care\b|\bat home\b"),
    ("laboratory", r"\blaborator(?:y|ies)\b|\blab conditions\b|\bcontrolled environment\b|\btest ?bed\b|\bgreenhouse\b|\bwind tunnel\b|\bbench(?:top)?\b"),
    ("field", r"\bfield (?:trial|study|site|conditions|deployment|experiment)s?\b|\bin the field\b|\bin the wild\b|\bnatural (?:habitat|conditions)\b"),
    ("production", r"\bproduction (?:deployment|environment|network|system)s?\b|\breal[\s\-]world deployment\b|\blive traffic\b"),
    ("cloud", r"\bcloud\b"),
    ("urban", r"\burban\b|\bcities\b|\bmetropolitan\b"),
    ("rural", r"\brural\b|\bvillages?\b"),
    ("marine", r"\bmarine\b|\bocean\b|\bcoastal\b"),
    ("freshwater", r"\bfreshwater\b|\blakes?\b|\brivers?\b"),
    ("recession", r"\brecession\b|\bdownturn\b|\bcrisis\b"),
    ("expansion", r"\bexpansion(?:ary)?\b|\bboom\b"),
]
_ENV_RES = [(k, re.compile(p, re.I)) for k, p in ENV_PATTERNS]


def parse_environment(text: str) -> dict | None:
    for k, rx in _ENV_RES:
        m = rx.search(text)
        if m:
            return {"text": m.group(0).lower(), "cat": k}
    return None


# ---------------------------------------------------------------- study design
DESIGN_PATTERNS = [
    ("meta", r"\bmeta[\s\-]analys[ie]s\b|\bsystematic review\b|\bpooled analysis\b"),
    ("simulation", r"\bsimulation(?:s| study| experiments?)?\b|\bsimulated\b|\bin silico\b|\bagent[\s\-]based model\b|"
                   r"\bnumerical model(?:ling|ing)?\b|\bcomputational model\b"),
    ("rct", r"\brandomi[sz]ed\b|\bplacebo[\s\-]controlled\b|\b(?:rct|rcts)\b|\bclinical trial\b"),
    ("cohort", r"\bcohort\b|\bprospective\b|\blongitudinal\b|\bpanel data\b"),
    ("case_control", r"\bcase[\s\-]control\b"),
    ("cross_sectional", r"\bcross[\s\-]sectional\b|\bsurvey\b"),
    ("experiment", r"\bcontrolled experiments?\b|\bexperiment(?:s|al study)?\b|\bfield experiment\b|\bablation\b"),
    ("observational", r"\bobservational\b|\bfield study\b|\bretrospective\b|\bcase study\b|\bmeasurement study\b"),
]
_DESIGN_RES = [(k, re.compile(p, re.I)) for k, p in DESIGN_PATTERNS]


def parse_design(text: str) -> dict | None:
    for k, rx in _DESIGN_RES:
        m = rx.search(text)
        if m:
            return {"text": m.group(0).lower(), "cat": k}
    return None


# Item 4 / proposal Sec. 9: conditions outside the five typed fields ("patients with chronic kidney disease",
# "under drought conditions"). Compared by content-word overlap.
_OTHER_RE = re.compile(
    r"\b(?:patients|participants|adults|children|individuals|subjects|people|women|men|those|cases|mice|rats)\s+"
    r"(?:with|who had|suffering from|diagnosed with)\s+((?:[a-z0-9][\w\-]*\s?){1,5}?)(?=\s*[,.;:)]|\s*\b(?:and|who|were|was|"
    r"received|receiving|treated|undergoing|in|at|for|from|aged)\b|\s*$)|\bunder\s+((?:[a-z][\w\-]*\s?){1,3}?)\s+conditions\b",
    re.I)
_OTHER_STOP = {"a", "an", "the", "mean", "median", "age", "aged", "history", "of", "or", "and", "without", "no",
               "at", "least", "one", "two", "both", "any", "known", "high", "low", "risk"}


def parse_other(text: str) -> dict | None:
    m = _OTHER_RE.search(text)
    if not m:
        return None
    phrase = (m.group(1) or m.group(2) or "").strip().lower()
    toks = sorted(w for w in re.findall(r"[a-z][\w\-]+", phrase) if w not in _OTHER_STOP)
    return {"text": phrase, "tokens": toks} if toks else None


PARSERS = {"population": parse_population, "dose": parse_dose, "time": parse_time,
           "environment": parse_environment, "design": parse_design, "other": parse_other}


def parse_context(text: str) -> dict:
    return {dim: fn(text) for dim, fn in PARSERS.items()}


# ---------------------------------------------------------------- epistemic status
_QUOTED = re.compile(
    r"^(?:previous|prior|earlier|past|recent|several|many|some|other)\s+(?:studies|work|research|reports|literature|"
    r"authors|investigators|trials)|\bit (?:has|had) (?:been|long been) (?:reported|suggested|proposed|hypothesi[sz]ed|"
    r"claimed|argued|assumed|believed|thought)|\bit is (?:widely |commonly |generally )?(?:believed|thought|assumed|"
    r"held)|\bhas been (?:reported|suggested|proposed|claimed|argued)\b|\bwas previously\b|\bconventional wisdom\b|"
    r"\b(?:is|are) (?:widely |commonly )?(?:believed|thought) to\b|\baccording to\b",
    re.I,
)
_HYPO = re.compile(r"\b(?:may|might|could|possibly|potentially|hypothesi[sz]e[sd]?|speculate|we propose that|"
                   r"is likely to|are likely to|putative)\b", re.I)
_SIM = re.compile(r"\bsimulat\w*\b|\bmodel (?:predicts|predicted|suggests)\b|\bin silico\b", re.I)
_EST = re.compile(r"\bestimat\w*\b|\bprojected\b|\bforecast\w*\b", re.I)


def epistemic_status(sentence: str) -> str:
    s = sentence.strip()
    if _QUOTED.search(s):
        return "quoted"
    if _SIM.search(s):
        return "simulated"
    if _HYPO.search(s):
        return "hypothesized"
    if _EST.search(s):
        return "estimated"
    return "observed"


def log_ratio(a: float, b: float) -> float:
    a, b = max(a, 1e-9), max(b, 1e-9)
    return abs(math.log(a / b))
