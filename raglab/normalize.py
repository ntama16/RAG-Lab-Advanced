"""Query normalisation.

Meets the corpus in its own words without losing the user's: every rule appends rather than substitutes,
identifiers are protected, and every change is recorded as a step for the trace.
Rules are data (editable in the Normalisation lab); an LLM can propose rules from the corpus itself.
"""
from __future__ import annotations
import json, re
from .bm25 import tokenize

GENERIC_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                   "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "twenty": 20, "thirty": 30, "sixty": 60, "ninety": 90}
UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "fortnight": 14, "fortnights": 14, "month": 30, "months": 30,
             "year": 365, "years": 365}
EMPTY_RULES = {"abbreviations": {}, "aliases": {}, "terminology": {}, "dates": True, "spelling": True}

# Hand-written rules for the bundled Halcyon Trailworks corpus.
HALCYON_RULES = {
    "abbreviations": {"rma": "return merchandise authorization RMA", "doa": "dead-on-arrival DOA",
                      "eta": "estimated time of arrival ETA delivery", "sla": "service level SLA response time",
                      "po": "purchase order", "fw": "firmware", "hq": "headquarters", "roi": "return on investment"},
    "aliases": {"ridgeline": "ZX-481", "the 481": "ZX-481", "481": "ZX-481", "cairn": "ZX-470", "470": "ZX-470",
                "trailpulse": "TrailPulse watch", "power bank": "Ember 20 power bank", "meridian": "Meridian 45 pack"},
    "terminology": {"send back": "return", "sent back": "returned", "money back": "refund", "give back": "return",
                    "broken on arrival": "dead-on-arrival DOA", "arrived broken": "dead-on-arrival DOA defect",
                    "won't switch on": "won't turn on power on", "won't turn on": "power on",
                    "find satellites": "GPS fix acquisition", "dropped in a lake": "submersion water resistance IPX",
                    "dropped in water": "submersion water resistance IPX", "in the water": "submersion water resistance",
                    "how long do i have": "return window days", "take ages": "slow", "takes ages": "slow",
                    "cash back": "refund", "guarantee": "warranty guarantee", "return rate": "return rate quality"},
    "dates": True, "spelling": True,
}


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def canonicalize_dates(q: str, steps: list) -> str:
    """'three weeks' -> '21 days', '2 months' -> '60 days'."""
    pat = re.compile(rf"\b((?:{'|'.join(GENERIC_NUMBERS)})|\d+|a|an)\s+({'|'.join(UNIT_DAYS)})\b", re.I)

    def sub(m):
        num, unit = m.group(1).lower(), m.group(2).lower()
        n = int(num) if num.isdigit() else 1 if num in ("a", "an") else GENERIC_NUMBERS.get(num)
        if n is None or (unit.startswith("day") and num.isdigit()):
            return m.group(0)
        to = f"{n * UNIT_DAYS[unit]} days"
        steps.append({"rule": "date", "from": m.group(0), "to": f"{m.group(0)} {to}",
                      "note": f'Durations are canonicalised to days so "{m.group(0)}" meets text written as "{to}".'})
        return f"{m.group(0)} {to}"
    return pat.sub(sub, q)


def correct_spelling(q: str, vocab: set[str], protected: set[str], steps: list) -> str:
    """Edit distance 1-2 against the corpus vocabulary, words of five letters or more, first letter must match."""
    by_letter: dict[str, list[str]] = {}
    for v in vocab:
        by_letter.setdefault(v[0], []).append(v)

    def sub(m):
        word = m.group(0)
        low = word.lower()
        if low in vocab or low in protected:
            return word
        best, best_d = None, 99
        for v in by_letter.get(low[0], []):
            if abs(len(v) - len(low)) > 2:
                continue
            d = levenshtein(low, v)
            if d < best_d:
                best, best_d = v, d
        if best and best_d <= (2 if len(low) >= 8 else 1):
            steps.append({"rule": "spelling", "from": word, "to": best,
                          "note": f'"{word}" is not in the corpus vocabulary; nearest term "{best}" (edit distance {best_d}).'})
            return best
        return word
    return re.sub(r"[A-Za-z]{5,}", sub, q)


def normalize(query: str, rules: dict, vocab: set[str]) -> dict:
    rules = {**EMPTY_RULES, **(rules or {})}
    steps: list[dict] = []
    q = query.strip()
    protected = [m for m in re.findall(r"\b[A-Z]{1,4}-\d{2,5}\b", q)]
    for alias, canonical in rules["aliases"].items():
        pat = re.compile(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", re.I)
        if pat.search(q) and canonical.lower() not in q.lower():
            q = pat.sub(lambda m: f"{m.group(0)} {canonical}", q, count=1)
            protected.append(canonical)
            steps.append({"rule": "alias", "from": alias, "to": f"{alias} {canonical}",
                          "note": f'"{alias}" is a known alias; the canonical identifier "{canonical}" is appended and protected from rewriting.'})
    for abbr, expansion in rules["abbreviations"].items():
        pat = re.compile(rf"\b{re.escape(abbr)}\b", re.I)
        if pat.search(q):
            q = pat.sub(expansion, q, count=1)
            steps.append({"rule": "acronym", "from": abbr.upper(), "to": expansion,
                          "note": "Acronym expanded so both the short and the long form can match."})
    for phrase, term in rules["terminology"].items():
        pat = re.compile(rf"\b{re.escape(phrase)}\b", re.I)
        if pat.search(q):
            q = pat.sub(lambda m: f"{m.group(0)} {term}", q, count=1)
            steps.append({"rule": "terminology", "from": phrase, "to": f"{phrase} {term}",
                          "note": "User wording mapped to corpus terminology; the original phrase is kept so intent is not lost."})
    if rules["dates"]:
        q = canonicalize_dates(q, steps)
    if rules["spelling"] and vocab:
        prot = {w.lower() for t in protected for w in re.findall(r"[A-Za-z]+", t)}
        q = correct_spelling(q, vocab, prot, steps)
    return {"original": query, "normalized": re.sub(r"\s+", " ", q).strip(), "steps": steps,
            "protected": list(dict.fromkeys(protected))}


def vocabulary(texts: list[str]) -> set[str]:
    return {t for text in texts for t in tokenize(text) if t.isalpha()}


SUGGEST = """You maintain the query-normalisation rules for a document search system.
From the corpus sample below, propose rules that help a user's everyday wording meet the corpus's wording.

- abbreviations: acronyms used in the corpus -> "long form ACRONYM" (keep the acronym in the value).
- aliases: informal or short product/system names -> the canonical identifier used in the corpus.
- terminology: phrases a lay user would type -> the term the corpus actually uses.

Only propose rules supported by the text. Keys are lower case. At most 12 per group.
Reply with JSON: {{"abbreviations": {{}}, "aliases": {{}}, "terminology": {{}}}}

Corpus sample:
{sample}"""


def clean_suggestion(out) -> dict:
    res = {"abbreviations": {}, "aliases": {}, "terminology": {}}
    if isinstance(out, dict):
        for g in res:
            if isinstance(out.get(g), dict):
                res[g] = {str(k).lower().strip(): str(v).strip() for k, v in out[g].items() if str(k).strip() and str(v).strip()}
    return res


def dumps(rules: dict) -> str:
    return json.dumps({**EMPTY_RULES, **rules}, indent=2, ensure_ascii=False)


def rules_key(index_id: str | None) -> str:
    """Storage key for a rule set. Rules were one global row while the UI asked you to pick an
    index, so tuning them for one corpus silently retuned every other."""
    return f"normalization:{index_id}" if index_id else "normalization"
