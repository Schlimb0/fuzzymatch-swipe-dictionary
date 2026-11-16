#!/usr/bin/env python3
"""
swipe_dedupe — deduplicate a product catalogue, with a Tinder-style CLI for
building the abbreviation dictionary and reviewing the grey-zone matches.

Pipeline
  1. normalise   lowercase, strip punctuation, standardise units (0.1kg -> 100g),
                 expand approved abbreviations (chd -> cheddar)
  2. exact       identical normalised records collapse immediately
  3. candidates  TF-IDF (char 3-5 grams) nearest neighbours, so blank brand/weight
                 fields can't hide a match
  4. compare     fuzzy token score + token-difference rules:
                   - numbers / sizes differ ............ never a match
                   - short codes differ (IPA vs APA) ... never a match
                   - pair you swiped "different" ....... never a match
                   - long-word typo (chedar/cheddar) ... ignored
                   - blank field ....................... unknown, not a mismatch
  5. cluster     merge, but refuse to chain incompatible records together
  6. survive     keep the most complete record, fill its blanks from the others

Workflow
  python swipe_dedupe.py demo                          # optional: sample data
  python swipe_dedupe.py mine products.csv             # find abbreviation candidates
  python swipe_dedupe.py swipe-abbr                    # swipe the dictionary
  python swipe_dedupe.py dedupe products.csv           # run the pipeline
  python swipe_dedupe.py mine products.csv             # again: now also "does this word matter?" cards
  python swipe_dedupe.py swipe-abbr
  python swipe_dedupe.py dedupe products.csv
  python swipe_dedupe.py swipe-matches                 # swipe what's left of the grey zone
  python swipe_dedupe.py dedupe products.csv           # final run with your decisions

Keys (swipe screens)
  → / l / d   same          ← / h / a   different        ↓ / j / space   skip
  u           undo          f           flip direction    k           same + learn abbreviation (matches)
  ?           help          q / Esc     save & quit

Requires: Python 3.9+, scikit-learn, numpy.  Recommended: rapidfuzz (much faster).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# --------------------------------------------------------------------------- fuzzy

try:
    from rapidfuzz import fuzz as _rf

    def ratio(a: str, b: str) -> float:
        return _rf.ratio(a, b)

    def token_sort(a: str, b: str) -> float:
        return _rf.token_sort_ratio(a, b)

    def token_set(a: str, b: str) -> float:
        return _rf.token_set_ratio(a, b)

    FUZZ_BACKEND = "rapidfuzz"
except ImportError:  # pure-python fallback (slower, same behaviour)
    from difflib import SequenceMatcher

    def ratio(a: str, b: str) -> float:
        if not a and not b:
            return 100.0
        return 100.0 * SequenceMatcher(None, a, b).ratio()

    def token_sort(a: str, b: str) -> float:
        return ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))

    def token_set(a: str, b: str) -> float:
        A, B = set(a.split()), set(b.split())
        inter = " ".join(sorted(A & B))
        da, db = " ".join(sorted(A - B)), " ".join(sorted(B - A))
        if inter and (not da or not db):
            return 100.0
        s1, s2 = f"{inter} {da}".strip(), f"{inter} {db}".strip()
        return max(ratio(inter, s1), ratio(inter, s2), ratio(s1, s2))

    FUZZ_BACKEND = "difflib (pip install rapidfuzz for ~20x speed)"

# ------------------------------------------------------------------ normalisation

BLANKS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "?", "unknown"}

UNIT_RE = re.compile(
    r"(?<![a-z0-9.])(\d+(?:[.,]\d+)?)\s*"
    r"(kilograms?|kgs?|grams?|gr|g|mg|litres?|liters?|ltrs?|l|ml|cl|oz|lbs?|pk|packs?)"
    r"(?![a-z])"
)
QTY_RE = re.compile(r"^\d+(?:\.\d+)?(?:g|ml|pk|oz|lb)$")


def _unit_sub(m: re.Match) -> str:
    v = float(m.group(1).replace(",", "."))
    u = m.group(2)
    if u.startswith("k"):
        v, b = v * 1000, "g"
    elif u == "mg":
        v, b = v / 1000, "g"
    elif u.startswith("g"):
        b = "g"
    elif u == "ml":
        b = "ml"
    elif u == "cl":
        v, b = v * 10, "ml"
    elif u.startswith("lb"):
        b = "lb"
    elif u.startswith("l"):
        v, b = v * 1000, "ml"
    elif u == "oz":
        b = "oz"
    else:
        b = "pk"
    return f" {v:g}{b} "


def base_norm(s) -> str:
    """Lowercase, ascii-fold, standardise quantities, drop punctuation."""
    if s is None:
        return ""
    s = str(s).replace("×", "x")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()
    if s in BLANKS:
        return ""
    s = s.replace("&", " and ")
    s = re.sub(r"(\d)\s*[x*]\s*(?=\d)", r"\1pk ", s)            # 4x330ml -> 4pk 330ml
    s = UNIT_RE.sub(_unit_sub, s)                                # 0.1kg -> 100g
    s = re.sub(r"(?<![a-z0-9])(\d+)\s*x(?![a-z0-9])", r" \1pk ", s)   # 6 x -> 6pk
    s = re.sub(r"(?<![a-z0-9])x\s*(\d+)(?![a-z0-9.])", r" \1pk ", s)  # x6 -> 6pk
    s = re.sub(r"[^a-z0-9.]+", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    return " ".join(s.split())


def has_digit(t: str) -> bool:
    return any(ch.isdigit() for ch in t)


def is_qty(t: str) -> bool:
    return bool(QTY_RE.match(t))


def looks_like_abbrev(a: str, b: str) -> bool:
    """Is the shorter string an in-order subsequence of the longer, same first letter?"""
    s, l = sorted((a, b), key=len)
    if not s or len(s) >= len(l) or s[0] != l[0]:
        return False
    it = iter(l.replace(" ", ""))
    return all(ch in it for ch in s.replace(" ", ""))


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Optimal-string-alignment distance (insert/delete/substitute/swap), capped."""
    if abs(len(a) - len(b)) >= cap:
        return cap
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if prev2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) >= cap:
            return cap
        prev2, prev = prev, cur
    return min(prev[-1], cap)


def close_typo(x: str, y: str) -> bool:
    """One edit apart (two if 10+ letters)."""
    return edit_distance(x, y) <= (2 if min(len(x), len(y)) >= 10 else 1)


TYPO_MAX_FREQ = 3


def is_typo(x: str, y: str, freq=None) -> bool:
    """chedar~cheddar yes; salted~unsalted no (one contains the other); codes no.
    With freq: only if the rarer spelling is rare — two common words one letter
    apart (Sagal / Kagal) are two real things, not a typo."""
    if len(x) < 5 or len(y) < 5:
        return False
    if freq is not None and min(freq.get(x, 0), freq.get(y, 0)) > TYPO_MAX_FREQ:
        return False
    if x in y or y in x or has_digit(x) or has_digit(y):
        return False
    return close_typo(x, y)


class Normalizer:
    def __init__(self, abbr: dict):
        self.tok: dict[str, str] = {}
        self.col: dict[str, dict[str, str]] = defaultdict(dict)
        for e in abbr.get("accepted", []):
            if e["scope"] == "token":
                self.tok[e["from"]] = e["to"]
            elif e["scope"] == "ignore":
                self.tok[e["from"]] = ""
            else:
                self.col[e["scope"].split(":", 1)[1]][e["from"]] = e["to"]
        for k in list(self.tok):  # resolve chains a->b->c
            seen, v = {k}, self.tok[k]
            while v in self.tok and v not in seen:
                seen.add(v)
                v = self.tok[v]
            self.tok[k] = v
        self.rejected = {frozenset((r["a"], r["b"])) for r in abbr.get("rejected", []) if r["scope"] != "ignore"}
        self.matters = {r["a"] for r in abbr.get("rejected", []) if r["scope"] == "ignore"}

    def text(self, s) -> str:
        s = base_norm(s)
        if self.tok and s:
            s = " ".join(x for x in (self.tok.get(t, t) for t in s.split()) if x)
        return s

    def attr(self, col: str, v) -> str:
        s = self.text(v)
        cmap = self.col.get(col)
        if cmap:
            s = cmap.get(s, s)
        return s


def attr_equal(a: str, b: str, freq=None) -> bool:
    if a == b:
        return True
    na = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]*)", a)
    nb = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]*)", b)
    if na and nb:
        if not na.group(2) or not nb.group(2):
            return float(na.group(1)) == float(nb.group(1))
        return False
    if has_digit(a) or has_digit(b):
        return False
    if freq is not None and min(freq.get(a, 0), freq.get(b, 0)) > TYPO_MAX_FREQ:
        return False  # both common values: genuinely different
    return close_typo(a, b) or token_set(a, b) >= 100


# ------------------------------------------------------------------------ files


def load_json(p: Path, default):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)


def load_csv(path: str):
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rd = csv.DictReader(f, dialect=dialect)
        rows = [dict(r) for r in rd]
        return list(rd.fieldnames or []), rows


def write_csv(path: Path, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


HARD_HINTS = ("brand", "weight", "size", "volume", "qty", "quantity", "pack", "net")


def resolve_cols(header, args) -> dict:
    low = {h.lower().strip(): h for h in header}

    def pick(name, fallbacks):
        if name:
            if name not in header:
                sys.exit(f"Column '{name}' not found. Columns: {', '.join(header)}")
            return name
        for fb in fallbacks:
            if fb in low:
                return low[fb]
        return None

    split = lambda s: [c.strip() for c in s.split(",") if c.strip()] if s else []
    id_col = pick(args.id_col, ["id", "item_id", "product_id", "sku", "code"])
    desc_col = pick(args.desc_col, ["description", "desc", "name", "product_name", "title"])
    if not desc_col:
        sys.exit("Couldn't find a description column; pass --desc-col NAME")
    ignore = set(split(args.ignore))
    for c in ignore:
        if c not in header:
            sys.exit(f"--ignore column '{c}' not found")
    rest = [h for h in header if h not in (id_col, desc_col) and h not in ignore]
    hard = split(args.hard) if args.hard else [h for h in rest if any(k in h.lower() for k in HARD_HINTS)]
    soft = split(args.soft) if args.soft else [h for h in rest if h not in hard]
    for c in hard + soft:
        if c not in header:
            sys.exit(f"Column '{c}' not found")
    return {"id": id_col, "desc": desc_col, "hard": hard, "soft": soft, "ignore": sorted(ignore)}


def add_col_args(p):
    p.add_argument("--id-col", help="ID column (default: id/sku/... or row number)")
    p.add_argument("--desc-col", help="description column (default: description/name/...)")
    p.add_argument("--hard", help="comma list: fields that must agree when both filled (default: brand/weight/size-like)")
    p.add_argument("--soft", help="comma list: fields compared loosely (default: everything else)")
    p.add_argument("--ignore", help="comma list of columns to ignore entirely")


# ------------------------------------------------------------------------ mining


def cmd_mine(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    header, rows = load_csv(args.input)
    cfg = resolve_cols(header, args)
    abbr = load_json(work / "abbreviations.json", {"accepted": [], "rejected": []})
    norm = Normalizer(abbr)
    decided = {(e["scope"], frozenset((e["from"], e["to"]))) for e in abbr["accepted"]}
    decided |= {(e["scope"], frozenset((e["a"], e["b"]))) for e in abbr["rejected"]}

    tok_freq = Counter()
    slots: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in rows:
        raw = r[cfg["desc"]]
        toks = norm.text(raw).split()
        tok_freq.update(toks)
        key = " ".join(toks)
        if key in seen or len(toks) < 2:
            continue
        seen.add(key)
        for i, t in enumerate(toks):
            if has_digit(t):
                continue
            slots[" ".join(toks[:i] + ["*"] + toks[i + 1:])][t].append(raw)

    pairs: dict[tuple, dict] = {}

    def add(scope, a, b, ex_a, ex_b, note=""):
        if (scope, frozenset((a, b))) in decided:
            return
        e = pairs.setdefault((scope, a, b), {"count": 0, "examples": []})
        e["count"] += 1
        if len(e["examples"]) < 2:
            e["examples"].append([ex_a, ex_b, note])

    for toks in slots.values():
        if 2 <= len(toks) <= args.max_slot:
            for a, b in combinations(sorted(toks), 2):
                add("token", a, b, toks[a][0], toks[b][0])

    # column values that differ between records with the same description
    by_desc = defaultdict(list)
    for r in rows:
        d = norm.text(r[cfg["desc"]])
        if d:
            by_desc[d].append(r)
    for d, grp in by_desc.items():
        if len(grp) < 2:
            continue
        for c in cfg["hard"] + cfg["soft"]:
            vals = {}
            for r in grp:
                v = norm.attr(c, r[c])
                if v and v not in vals:
                    vals[v] = r
            if 2 <= len(vals) <= 6:
                for a, b in combinations(sorted(vals), 2):
                    if has_digit(a) and has_digit(b):
                        continue
                    add(f"col:{c}", a, b, vals[a][cfg["desc"]], vals[b][cfg["desc"]],
                        f"{c}: {vals[a][c]!s}  vs  {vals[b][c]!s}")

    # one-sided words from the last dedupe's grey zone: do they change the product?
    extra = Counter()
    ex_for = {}
    for q in load_json(work / "review_queue.json", []):
        for t in set(q.get("extra", [])):
            if has_digit(t) or ("ignore", frozenset((t, ""))) in decided:
                continue
            extra[t] += 1
            ex_for.setdefault(t, [q["a"][q["desc_col"]], q["b"][q["desc_col"]], ""])
    ignore_cards = [{"scope": "ignore", "from": t, "to": "", "count": n, "likely": False, "examples": [ex_for[t]]}
                    for t, n in extra.most_common() if n >= args.min_count]

    out = []
    for (scope, a, b), e in pairs.items():
        if is_typo(a, b, tok_freq):
            continue  # typo — the matcher already tolerates these
        likely = looks_like_abbrev(a, b)
        if not likely and (e["count"] < args.min_count or
                           (scope == "token" and min(tok_freq[a], tok_freq[b]) < 3)):
            continue
        if likely:
            frm, to = sorted((a, b), key=len)
        else:  # synonym: map rarer form onto commoner one
            frm, to = sorted((a, b), key=lambda t: (tok_freq[t], len(t)))
        out.append({"scope": scope, "from": frm, "to": to, "count": e["count"],
                    "likely": likely, "examples": e["examples"]})
    out.sort(key=lambda e: (not e["likely"], -e["count"], e["from"]))
    out[sum(e["likely"] for e in out):sum(e["likely"] for e in out)] = ignore_cards
    save_json(work / "abbr_candidates.json", out)
    save_json(work / "config.json", cfg)
    n_likely = sum(e["likely"] for e in out)
    print(f"Mined {len(out)} cards from {len(rows):,} rows: {n_likely} likely abbreviations, "
          f"{len(ignore_cards)} 'does this word matter?' words (from the last dedupe's grey zone), "
          f"{len(out)-n_likely-len(ignore_cards)} possible synonyms -> {work/'abbr_candidates.json'}")
    print(f"Next: python {Path(sys.argv[0]).name} swipe-abbr" + (f" --work {args.work}" if args.work != "dedupe_work" else ""))


# ----------------------------------------------------------------------- compare


class Engine:
    def __init__(self, cfg, norm: Normalizer, review_min: float, rows=()):
        self.cfg, self.norm, self.review_min = cfg, norm, review_min
        # known text values of hard fields (e.g. brands), so a blank brand can be
        # inferred from the description: "Acme Dairy Cheddar..." -> brand acme dairy
        self.known = {}
        self.tok_freq = Counter()
        for r in rows:
            self.tok_freq.update(self.norm.text(r[cfg["desc"]]).split())
        self.val_freq = {}
        for k in cfg["hard"]:
            vals = Counter(self.norm.attr(k, r[k]) for r in rows)
            vals.pop("", None)
            self.val_freq[k] = vals
            if vals and not any(has_digit(v) for v in vals):
                pats = sorted((v for v, n in vals.items() if n >= 2), key=len, reverse=True)
                if pats:
                    self.known[k] = re.compile(r"(?<!\S)(" + "|".join(map(re.escape, pats)) + r")(?!\S)")

    def prep(self, row: dict) -> dict:
        c = self.cfg
        p = {
            "desc": self.norm.text(row[c["desc"]]),
            "hard": {k: self.norm.attr(k, row[k]) for k in c["hard"]},
            "soft": {k: self.norm.attr(k, row[k]) for k in c["soft"]},
        }
        for k, rx in self.known.items():
            if not p["hard"][k]:
                m = rx.search(p["desc"])
                if m:
                    p["hard"][k] = m.group(1)
        return p

    def compare(self, A: dict, B: dict) -> dict:
        da, db = A["desc"], B["desc"]
        score = max(token_sort(da, db), token_set(da, db))
        res = {"score": round(score, 1), "decision": "low", "reasons": [], "only_a": [], "only_b": [], "pair": None, "extra": []}

        def veto(msg):
            res["decision"] = "veto"
            res["veto"] = msg
            res["reasons"].append(msg)
            return res

        for k in self.cfg["hard"]:
            a, b = A["hard"][k], B["hard"][k]
            if a and b and not attr_equal(a, b, self.val_freq.get(k) or None):
                return veto(f"{k} differs ({a} / {b})")

        ca, cb = Counter(da.split()), Counter(db.split())
        only_a, only_b = list((ca - cb).elements()), list((cb - ca).elements())
        res["only_a"], res["only_b"] = only_a, only_b
        for t in only_a + only_b:
            if t in self.norm.matters:
                return veto(f"“{t}” matters (only one side)")
        for x in set(only_a):
            for y in set(only_b):
                if frozenset((x, y)) in self.norm.rejected:
                    return veto(f"you said {x} ≠ {y}")
        rem_b, paired = only_b[:], []
        for x in only_a:
            if rem_b:
                y = max(rem_b, key=lambda y: ratio(x, y))
                rem_b.remove(y)
                paired.append((x, y))
            else:
                paired.append((x, None))
        paired += [(None, y) for y in rem_b]

        real = []
        for x, y in paired:
            if x and y:
                if has_digit(x) and has_digit(y):
                    return veto(f"size/number differs ({x} / {y})")
                if is_typo(x, y, self.tok_freq or None):
                    res["reasons"].append(f"typo {x}~{y}")
                    continue
                if len(x) <= 4 and len(y) <= 4 and x.isalpha() and y.isalpha():
                    return veto(f"codes differ ({x} / {y})")
                real.append(f"{x}/{y}")
            else:
                t = x or y
                other = B["desc"] if x else A["desc"]
                if is_qty(t) and any(is_qty(o) for o in other.split()):
                    return veto(f"size/number differs ({t} on one side only)")
                real.append(f"+{t}")
                res["extra"].append(x or y)
        if len(paired) == 1 and paired[0][0] and paired[0][1]:
            res["pair"] = list(paired[0])

        soft_conf = []
        for k in self.cfg["soft"]:
            a, b = A["soft"][k], B["soft"][k]
            if a and b and a != b and ratio(a, b) < 85 and token_set(a, b) < 100:
                soft_conf.append(f"{k}: {a} / {b}")

        if real:
            res["reasons"].append("words differ: " + ", ".join(real))
        res["n_real"] = len(real) + len(soft_conf)
        res["reasons"] += soft_conf
        if not real and not soft_conf:
            res["decision"] = "auto"
        elif score >= self.review_min:
            res["decision"] = "review"
        return res


# ------------------------------------------------------------------------ dedupe


def cmd_dedupe(args):
    try:
        import numpy as np  # noqa: F401
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        sys.exit("dedupe needs scikit-learn:  pip install scikit-learn rapidfuzz")

    global TYPO_MAX_FREQ
    TYPO_MAX_FREQ = args.typo_max_freq
    t0 = time.time()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    header, rows = load_csv(args.input)
    cfg = resolve_cols(header, args)
    save_json(work / "config.json", cfg)
    abbr = load_json(work / "abbreviations.json", {"accepted": [], "rejected": []})
    norm = Normalizer(abbr)
    eng = Engine(cfg, norm, args.review_min, rows)
    md = load_json(work / "match_decisions.json", {"same": [], "different": []})

    ids = [str(r[cfg["id"]]) if cfg["id"] else str(i + 1) for i, r in enumerate(rows)]
    if len(set(ids)) != len(ids):
        print("! IDs are not unique; using row numbers instead")
        ids = [str(i + 1) for i in range(len(rows))]
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}")
    log(f"{len(rows):,} rows · fuzzy backend: {FUZZ_BACKEND}")
    log(f"hard fields: {cfg['hard'] or '-'} · soft fields: {cfg['soft'] or '-'} · "
        f"{len(norm.tok)} abbreviations, {len(norm.rejected)} 'different' pairs")

    # 1-2. normalise + exact units
    prepped = [eng.prep(r) for r in rows]
    unit_of_key, units = {}, []
    row_unit = []
    for i, p in enumerate(prepped):
        key = json.dumps(p, sort_keys=True)
        u = unit_of_key.setdefault(key, len(units))
        if u == len(units):
            units.append([])
        units[u].append(i)
        row_unit.append(u)
    rep = [u[0] for u in units]
    U = len(units)
    log(f"exact duplicates collapsed: {len(rows)-U:,} rows → {U:,} unique records")

    # 3. candidates
    desc_groups = defaultdict(list)
    for u in range(U):
        d = prepped[rep[u]]["desc"]
        if d:
            desc_groups[d].append(u)
    cand = set()
    for g in desc_groups.values():
        for a, b in combinations(g[:200], 2):
            cand.add((a, b))
    uniq = list(desc_groups)
    if len(uniq) > 1:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        X = vec.fit_transform(uniq)
        k = min(args.k + 1, len(uniq))
        dist, idx = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(X).kneighbors(X)
        for i in range(len(uniq)):
            for d, j in zip(dist[i], idx[i]):
                if j == i or 1 - d < args.min_cos:
                    continue
                for a in desc_groups[uniq[i]][:25]:
                    for b in desc_groups[uniq[j]][:25]:
                        cand.add((min(a, b), max(a, b)))
    log(f"TF-IDF candidate pairs: {len(cand):,}")

    # 4. compare
    cache = {}

    def cmp(a, b):
        key = (a, b) if a < b else (b, a)
        r = cache.get(key)
        if r is None:
            r = cache[key] = eng.compare(prepped[rep[key[0]]], prepped[rep[key[1]]])
        return r

    stats = Counter()
    veto_reasons = Counter()
    for a, b in cand:
        r = cmp(a, b)
        stats[r["decision"]] += 1
        if r["decision"] == "veto":
            veto_reasons[re.sub(r" \(.*", "", re.sub(r"said (\S+) ≠ (\S+)", lambda m: "said " + " ≠ ".join(sorted(m.groups())), r["veto"]))] += 1
    log("pairs → " + ", ".join(f"{k}: {v:,}" for k, v in sorted(stats.items())))

    # 5. cluster
    id_unit = {ids[i]: row_unit[i] for i in range(len(rows))}
    to_units = lambda pr: (id_unit.get(pr[0]), id_unit.get(pr[1]))
    cannot = set()
    for pr in md["different"]:
        a, b = to_units(pr)
        if a is not None and b is not None:
            cannot.add(frozenset((a, b)))
    parent = list(range(U))
    members = {u: [u] for u in range(U)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    blocked = 0

    def union(a, b, forced=False):
        nonlocal blocked
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        for x in members[ra]:
            for y in members[rb]:
                if frozenset((x, y)) in cannot or (not forced and cmp(x, y)["decision"] == "veto"):
                    blocked += 1
                    return
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra] += members.pop(rb)

    for pr in md["same"]:
        a, b = to_units(pr)
        if a is not None and b is not None:
            union(a, b, forced=True)
    for (a, b) in sorted((p for p in cand if cache[p]["decision"] == "auto"), key=lambda p: -cache[p]["score"]):
        union(a, b)
    if blocked:
        log(f"chaining guard blocked {blocked:,} merges that would join incompatible records")

    # review queue
    decided = {frozenset(p) for p in md["same"] + md["different"]}
    queue = []
    for (a, b) in cand:
        r = cache[(a, b)]
        if r["decision"] != "review" or find(a) == find(b):
            continue
        ia, ib = rep[a], rep[b]
        if frozenset((ids[ia], ids[ib])) in decided:
            continue
        cols = [cfg["desc"]] + cfg["hard"] + cfg["soft"]
        queue.append({
            "ids": [ids[ia], ids[ib]], "score": r["score"], "reasons": r["reasons"], "n_real": r.get("n_real", 1),
            "only_a": r["only_a"], "only_b": r["only_b"], "pair": r["pair"], "extra": r["extra"],
            "cols": cols, "desc_col": cfg["desc"],
            "a": {c: rows[ia][c] for c in cols}, "b": {c: rows[ib][c] for c in cols},
        })
    queue.sort(key=lambda q: (q["n_real"], -q["score"]))  # fewest differences first
    save_json(work / "review_queue.json", queue)

    # 6. survivorship
    clusters = defaultdict(list)
    for u in range(U):
        clusters[find(u)].extend(units[u])
    all_cols = [c for c in header]
    nonblank = lambda r: sum(1 for c in all_cols if base_norm(r.get(c, "")))
    out_rows, mapping = [], []
    for members_rows in clusters.values():
        members_rows.sort()
        canon = max(members_rows, key=lambda i: (nonblank(rows[i]), len(rows[i][cfg["desc"]] or ""), -i))
        merged = dict(rows[canon])
        filled = []
        for c in all_cols:
            if not base_norm(merged.get(c, "")):
                vals = Counter(rows[i][c] for i in members_rows if base_norm(rows[i].get(c, "")))
                if vals:
                    merged[c] = vals.most_common(1)[0][0]
                    filled.append(c)
        merged["_merged_count"] = len(members_rows)
        merged["_merged_ids"] = ";".join(ids[i] for i in members_rows) if len(members_rows) > 1 else ""
        merged["_filled_from_duplicates"] = ",".join(filled)
        merged["_order"] = canon
        out_rows.append(merged)
        for i in members_rows:
            how = "canonical" if i == canon else ("exact" if row_unit[i] == row_unit[canon] else "fuzzy")
            mapping.append({"id": ids[i], "canonical_id": ids[canon], "how": how})
    out_rows.sort(key=lambda r: r["_order"])
    write_csv(work / "deduped.csv", all_cols + ["_merged_count", "_merged_ids", "_filled_from_duplicates"], out_rows)
    mapping.sort(key=lambda m: m["id"])
    write_csv(work / "id_mapping.csv", ["id", "canonical_id", "how"], mapping)

    multi = sum(1 for c in clusters.values() if len(c) > 1)
    lines = [
        f"rows in:            {len(rows):,}",
        f"rows out:           {len(out_rows):,}  (removed {len(rows)-len(out_rows):,})",
        f"duplicate groups:   {multi:,}",
        f"grey-zone pairs:    {len(queue):,}  → swipe-matches",
        f"user decisions:     {len(md['same'])} same, {len(md['different'])} different",
        "vetoes (pairs kept apart):",
    ] + [f"  {v:>7,}  {k}" for k, v in veto_reasons.most_common()]
    (work / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines))
    print(f"\nWrote {work/'deduped.csv'}, {work/'id_mapping.csv'}, {work/'review_queue.json'}")
    if queue:
        print(f"Next: python {Path(sys.argv[0]).name} swipe-matches" + (f" --work {args.work}" if args.work != "dedupe_work" else ""))


# ---------------------------------------------------------------------- terminal

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def sty(s, *codes) -> str:
    if not COLOR or not codes:
        return str(s)
    return f"\x1b[{';'.join(str(c) for c in codes)}m{s}\x1b[0m"


def vlen(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def padr(s: str, w: int) -> str:
    return s + " " * max(0, w - vlen(s))


def center(s: str, w: int) -> str:
    gap = max(0, w - vlen(s))
    return " " * (gap // 2) + s + " " * (gap - gap // 2)


def wrap_items(items, width):
    """items: list of (plain, styled). Returns wrapped styled lines."""
    lines, cur, cur_len = [], [], 0
    for plain, styled in items:
        while len(plain) > width:  # hard-split very long words
            if cur:
                lines.append(" ".join(cur))
                cur, cur_len = [], 0
            lines.append(plain[:width])
            plain = styled = plain[width:]
        add = len(plain) + (1 if cur else 0)
        if cur and cur_len + add > width:
            lines.append(" ".join(cur))
            cur, cur_len = [], 0
            add = len(plain)
        cur.append(styled)
        cur_len += add
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def box(lines, w, color=None, title=""):
    inner = w - 4
    top = "╭" + "─" * (w - 2) + "╮"
    if title:
        t = f" {title} "
        top = "╭─" + t + "─" * max(0, w - 3 - len(t)) + "╮"
    out = [sty(top, color)]
    for l in lines:
        out.append(sty("│", color) + " " + padr(l, inner) + " " + sty("│", color))
    out.append(sty("╰" + "─" * (w - 2) + "╯", color))
    return out


class Term:
    def __init__(self, anim=True):
        self.tty = sys.stdin.isatty() and sys.stdout.isatty()
        self.anim = anim and self.tty
        self.win = os.name == "nt"
        self._old = None
        self._pending = ""

    def __enter__(self):
        if self.tty:
            if self.win:
                os.system("")  # enable ANSI on Windows 10+
            else:
                import termios
                import tty
                self._old = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
            sys.stdout.write("\x1b[?1049h\x1b[?25l")
            sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        if self.tty:
            sys.stdout.write("\x1b[?25h\x1b[?1049l")
            sys.stdout.flush()
            if self._old is not None:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)

    def size(self):
        s = shutil.get_terminal_size((96, 32))
        return s.columns, s.lines

    def draw(self, lines):
        if self.tty:
            sys.stdout.write("\x1b[H\x1b[2J" + "\n".join(lines))
        else:
            sys.stdout.write("\n".join(ANSI_RE.sub("", l) for l in lines) + "\n" + "·" * 40 + "\n")
        sys.stdout.flush()

    def key(self) -> str:
        if self.win and self.tty:
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                return {"K": "left", "M": "right", "P": "down", "H": "up"}.get(msvcrt.getwch(), "")
            return {"\r": "enter", "\x1b": "esc", "\x08": "backspace", " ": "space"}.get(ch, ch)
        if not self.tty:
            ch = sys.stdin.read(1)
            if ch == "":
                return "q"  # EOF when piped
            if ch in ("\n", "\r"):
                return self.key()
            return {" ": "space"}.get(ch, ch)
        # raw fd reads: sys.stdin's buffer would swallow the rest of an arrow-key sequence
        import select
        fd = sys.stdin.fileno()
        if not self._pending:
            self._pending = os.read(fd, 64).decode("utf-8", "ignore")
        if self._pending.startswith("\x1b"):
            if len(self._pending) == 1 and select.select([fd], [], [], 0.05)[0]:
                self._pending += os.read(fd, 64).decode("utf-8", "ignore")
            for seq, name in (("\x1b[C", "right"), ("\x1b[D", "left"), ("\x1b[B", "down"), ("\x1b[A", "up"),
                              ("\x1bOC", "right"), ("\x1bOD", "left"), ("\x1bOB", "down"), ("\x1bOA", "up")):
                if self._pending.startswith(seq):
                    self._pending = self._pending[len(seq):]
                    return name
            m = re.match(r"\x1b\[[0-9;]*[~A-Za-z]", self._pending)  # other escape sequence: ignore
            if m:
                self._pending = self._pending[m.end():]
                return ""
            self._pending = self._pending[1:]
            return "esc"
        ch, self._pending = self._pending[0], self._pending[1:]
        if ch in ("\n", "\r"):
            return "enter"
        return {"\x7f": "backspace", "\x08": "backspace", " ": "space"}.get(ch, ch)


KEYMAP = {
    "right": "yes", "l": "yes", "d": "yes", "y": "yes",
    "left": "no", "h": "no", "a": "no", "n": "no",
    "down": "skip", "up": "skip", "j": "skip", "space": "skip", "s": "skip",
    "u": "undo", "backspace": "undo", "z": "undo",
    "f": "flip", "k": "learn", "?": "help", "q": "quit", "esc": "quit",
}

MILESTONES = {10: "🔥 Warming up", 25: "⚡ 25 down!", 50: "🏆 Half a century", 100: "👑 Centurion",
              200: "🧀 Big cheese", 500: "🚀 Unstoppable", 1000: "🐐 Dedupe legend"}


# ----------------------------------------------------------------- swipe engine


class Swiper:
    """Generic Tinder loop. Subclasses supply render_card/apply/revert."""

    title = ""
    yes_label, no_label = "same", "different"
    extra_help: list[str] = []

    def __init__(self, cards, term: Term):
        self.cards, self.term = cards, term
        self.i, self.history = 0, []
        self.n_yes = self.n_no = self.n_skip = 0
        self.streak = self.best_streak = 0
        self.toast, self.toast_until = "", 0.0
        self.times: list[float] = []
        self.t_card = time.time()

    # -- to override
    def render_card(self, card, w) -> list[str]: ...
    def apply(self, card, decision): ...
    def revert(self, card, decision): ...
    def flip(self, card): ...

    # -- drawing
    def frame(self, card_lines, offset=0, stamp=None, color=None):
        cols, rows = self.term.size()
        total = len(self.cards)
        done = min(self.i, total)
        bar_w = max(10, min(cols - 30, 60))
        fill = int(bar_w * done / max(1, total))
        bar = sty("█" * fill, 36) + sty("░" * (bar_w - fill), 2)
        head = sty(" SWIPE·DEDUPE ", 1, 7) + "  " + sty(self.title, 1)
        lines = ["", head, f" {bar} {sty(f'{done}/{total}', 1)}", ""]
        w = vlen(card_lines[0]) if card_lines else 60
        margin = max(0, (cols - w) // 2 + offset)
        if stamp:
            lines.append(" " * max(0, (cols - vlen(stamp)) // 2 + offset) + stamp)
        else:
            lines.append("")
        lines += [" " * margin + l for l in card_lines]
        lines.append("")
        no_l, yes_l = self.labels(self.cards[self.i]) if self.i < len(self.cards) else (self.no_label, self.yes_label)
        nl = sty(f"✗ ← {no_l}", 31, 1)
        yl = sty(f"{yes_l} → ♥", 32, 1)
        sk = sty("↓ skip", 2)
        lines.append(center(f"{nl}        {sk}        {yl}", cols))
        hint = "u undo · " + " · ".join(self.extra_help + ["? help", "q save & quit"])
        lines.append(center(sty(hint, 2), cols))
        lines.append("")
        avg = (sum(self.times[-20:]) / len(self.times[-20:])) if self.times else 0
        left = total - done
        eta = f"{avg:.1f}s/card · ~{max(1, round(avg*left/60))} min left" if self.times else "go!"
        stats = (f"{sty('♥', 32)} {self.n_yes}   {sty('✗', 31)} {self.n_no}   "
                 f"{sty('↷', 2)} {self.n_skip}   🔥 streak {self.streak}   {sty(eta, 2)}")
        lines.append(center(stats, cols))
        if self.toast and time.time() < self.toast_until:
            lines.append("")
            lines.append(center(sty(f" {self.toast} ", 1, 30, 43), cols))
        return lines

    def animate(self, card_lines, decision):
        if not self.term.anim:
            return
        cols, _ = self.term.size()
        if decision in ("yes", "learn"):
            stamp, color, d = sty(" ♥ SAME ", 1, 30, 42), 32, 1
        elif decision == "no":
            stamp, color, d = sty(" ✗ DIFFERENT ", 1, 37, 41), 31, -1
        else:
            stamp, color, d = sty(" ↷ SKIP ", 1, 30, 47), 2, 0
        w = vlen(card_lines[0])
        maxoff = max(4, (cols - w) // 2)
        for k in range(1, 7):
            tinted = [sty(ANSI_RE.sub("", l), color) for l in card_lines]
            off = int(d * maxoff * k / 6) if d else 0
            self.term.draw(self.frame(tinted, offset=off, stamp=stamp))
            time.sleep(0.03)

    def show_help(self):
        cols, _ = self.term.size()
        lines = ["", sty(" HOW TO PLAY ", 1, 7), "",
                 f"  → / l / d     {self.yes_label}",
                 f"  ← / h / a     {self.no_label}",
                 "  ↓ / j / space skip (comes back next session)",
                 "  u / z         undo last swipe",
                 *[f"  {h}" for h in self.extra_help],
                 "  q / Esc       save & quit (everything is saved after every swipe)",
                 "", sty("  press any key", 2)]
        self.term.draw(lines)
        self.term.key()

    def run(self):
        while True:
            cols, _ = self.term.size()
            w = max(50, min(cols - 4, 92))
            if self.i >= len(self.cards):
                self.finish()
                return
            card = self.cards[self.i]
            card_lines = self.render_card(card, w)
            self.term.draw(self.frame(card_lines))
            action = KEYMAP.get(self.term.key())
            if action is None:
                continue
            if action == "quit":
                return
            if action == "help":
                self.show_help()
                continue
            if action == "flip":
                self.flip(card)
                continue
            if action == "undo":
                if self.history:
                    idx, dec = self.history.pop()
                    if dec != "skip":
                        self.revert(self.cards[idx], dec)
                    if dec in ("yes", "learn"):
                        self.n_yes -= 1
                    elif dec == "no":
                        self.n_no -= 1
                    else:
                        self.n_skip -= 1
                    self.streak = max(0, self.streak - 1)
                    self.i = idx
                    self.say("↶ undone")
                continue
            if action == "learn" and not self.can_learn(card):
                self.say("nothing to learn on this card — use → instead")
                continue
            self.animate(card_lines, action)
            if action != "skip":
                self.apply(card, action)
            self.history.append((self.i, action))
            self.times.append(min(30.0, time.time() - self.t_card))
            self.t_card = time.time()
            if action in ("yes", "learn"):
                self.n_yes += 1
            elif action == "no":
                self.n_no += 1
            else:
                self.n_skip += 1
            if action == "skip":
                self.streak = 0
            else:
                self.streak += 1
                self.best_streak = max(self.best_streak, self.streak)
            decided = self.n_yes + self.n_no
            if action != "skip" and decided in MILESTONES:
                self.say(MILESTONES[decided])
            elif action != "skip" and self.streak and self.streak % 15 == 0:
                self.say(random.choice(["💪 Machine!", "🎯 Locked in", "🔥 Streak on fire", "🧹 So clean"]))
            self.i += 1

    def can_learn(self, card) -> bool:
        return False

    def labels(self, card):
        return self.no_label, self.yes_label

    def say(self, msg, secs=2.5):
        self.toast, self.toast_until = msg, time.time() + secs

    def finish(self):
        cols, _ = self.term.size()
        lines = ["", sty(" ALL DONE 🎉 ", 1, 30, 42), "",
                 f"  {sty('♥', 32)} {self.n_yes} same   {sty('✗', 31)} {self.n_no} different   "
                 f"{sty('↷', 2)} {self.n_skip} skipped   best streak {self.best_streak}", ""]
        lines += self.finish_lines()
        lines += ["", sty("  press any key to exit", 2)]
        self.term.draw(lines)
        if self.term.tty:
            self.term.key()

    def finish_lines(self):
        return []


def highlight_raw(raw: str, tokens: set, norm_fn, code=(1, 30, 43)) -> list:
    items = []
    for word in str(raw).split():
        n = norm_fn(word)
        hit = n and any(t in tokens for t in n.split())
        items.append((word, sty(word, *code) if hit else word))
    return items


class AbbrSwiper(Swiper):
    title = "abbreviation dictionary"
    yes_label, no_label = "same meaning", "different"
    extra_help = ["f flip direction"]

    def __init__(self, cards, term, store_path: Path):
        super().__init__(cards, term)
        self.path = store_path
        self.store = load_json(store_path, {"accepted": [], "rejected": []})

    def render_card(self, c, w):
        if c["scope"] == "ignore":
            return self.render_word_card(c, w)
        inner = w - 4
        frm, to = c["from"].upper(), c["to"].upper()
        big = sty(frm, 1, 36) + sty("   ⟶   ", 2) + sty(to, 1, 35)
        lines = ["", center(big, inner), ""]
        where = "descriptions" if c["scope"] == "token" else f"column “{c['scope'][4:]}”"
        tag = sty("looks like an abbreviation or typo", 32) if c["likely"] else sty("possible synonym / variant", 33)
        lines.append(center(f"seen in {sty(c['count'], 1)} context{'s' * (c['count'] != 1)} · {tag}", inner))
        lines.append(center(sty(f"scope: {where}", 2), inner))
        lines.append("")
        toks = {c["from"], c["to"]}
        for ex_a, ex_b, note in c["examples"]:
            for lbl, ex in (("A", ex_a), ("B", ex_b)):
                wrapped = wrap_items(highlight_raw(ex, toks, base_norm), inner - 4)
                lines.append(sty(lbl, 2) + "   " + wrapped[0])
                lines += ["    " + l for l in wrapped[1:]]
            if note:
                lines.append("    " + sty(note, 2))
            lines.append("")
        q = f"Same thing?  → yes: {c['from']} will be read as {c['to']}"
        lines.append(sty(q[:inner], 2))
        return box(lines, w, color=36, title=f"#{self.i+1}")

    def render_word_card(self, c, w):
        inner = w - 4
        lines = ["", center(sty("does  ", 2) + sty(f"“{c['from'].upper()}”", 1, 33) + sty("  change the product?", 2), inner), ""]
        lines.append(center(f"unmatched word in {sty(c['count'], 1)} grey-zone pair{'s' * (c['count'] != 1)}", inner))
        lines.append("")
        for ex_a, ex_b, _ in c["examples"]:
            for lbl, ex in (("A", ex_a), ("B", ex_b)):
                wrapped = wrap_items(highlight_raw(ex, {c["from"]}, base_norm), inner - 4)
                lines.append(sty(lbl, 2) + "   " + wrapped[0])
                lines += ["    " + l for l in wrapped[1:]]
            lines.append("")
        lines.append(sty(f"→ noise: ignore “{c['from']}” when matching    ← it matters: never merge if only one side has it"[:inner], 2))
        return box(lines, w, color=33, title=f"#{self.i+1} · word check")

    def labels(self, c):
        return ("it matters", "just noise") if c["scope"] == "ignore" else (self.no_label, self.yes_label)

    def flip(self, c):
        if c["scope"] == "ignore":
            return
        c["from"], c["to"] = c["to"], c["from"]
        self.say("⇄ flipped direction")

    def apply(self, c, d):
        if d == "yes":
            self.store["accepted"].append({"from": c["from"], "to": c["to"], "scope": c["scope"], "count": c["count"]})
        else:
            self.store["rejected"].append({"a": c["from"], "b": c["to"], "scope": c["scope"]})
        save_json(self.path, self.store)

    def revert(self, c, d):
        key = "accepted" if d == "yes" else "rejected"
        if self.store[key]:
            self.store[key].pop()
        save_json(self.path, self.store)

    def finish_lines(self):
        return [f"  dictionary: {len(self.store['accepted'])} abbreviations, {len(self.store['rejected'])} 'different' pairs",
                f"  saved to {self.path}",
                "  tip: run `mine` again — new expansions often reveal more pairs"]


class MatchSwiper(Swiper):
    title = "is this the same product?"
    extra_help = ["k same + learn the differing word as an abbreviation"]

    def __init__(self, cards, term, work: Path):
        super().__init__(cards, term)
        self.dpath, self.apath = work / "match_decisions.json", work / "abbreviations.json"
        self.dec = load_json(self.dpath, {"same": [], "different": []})

    def render_card(self, c, w):
        inner = w - 4
        fw = min(14, max(len(x) for x in c["cols"]) + 1)
        cw = (inner - fw - 3) // 2
        sep = sty("│", 2)
        lines = [center(f"score {sty(c['score'], 1)}" + sty("  ·  ", 2) + sty("; ".join(c["reasons"])[: inner - 20], 33), inner),
                 sty("─" * inner, 2)]
        hdr = padr(sty("field", 2), fw) + " " + padr(sty(f"A · id {c['ids'][0]}", 1, 36), cw) + " " + sep + " " + sty(f"B · id {c['ids'][1]}", 1, 35)
        lines += [hdr, sty("─" * inner, 2)]
        for col in c["cols"]:
            va, vb = str(c["a"].get(col, "") or ""), str(c["b"].get(col, "") or "")
            if col == c["desc_col"]:
                ia = wrap_items(highlight_raw(va, set(c["only_a"]), base_norm), cw)
                ib = wrap_items(highlight_raw(vb, set(c["only_b"]), base_norm), cw)
            else:
                blank_a, blank_b = not base_norm(va), not base_norm(vb)
                diff = not blank_a and not blank_b and base_norm(va) != base_norm(vb)
                fmt = lambda v, blank: [sty("·", 2)] if blank else wrap_items([(x, sty(x, 33, 1) if diff else x) for x in v.split()], cw)
                ia, ib = fmt(va, blank_a), fmt(vb, blank_b)
            for k in range(max(len(ia), len(ib))):
                f = sty(col[: fw - 1], 2) if k == 0 else ""
                a = ia[k] if k < len(ia) else ""
                b = ib[k] if k < len(ib) else ""
                lines.append(padr(f, fw) + " " + padr(a, cw) + " " + sep + " " + b)
        lines.append("")
        if c.get("pair"):
            x, y = c["pair"]
            lines.append(sty(f"k = same, and always read {min(x, y, key=len)} as {max(x, y, key=len)}", 2))
        return box(lines, w, color=35, title=f"#{self.i+1}")

    def can_learn(self, c):
        return bool(c.get("pair"))

    def apply(self, c, d):
        key = "same" if d in ("yes", "learn") else "different"
        self.dec[key].append(c["ids"])
        save_json(self.dpath, self.dec)
        if d == "learn":
            x, y = c["pair"]
            frm, to = sorted((x, y), key=len)
            ab = load_json(self.apath, {"accepted": [], "rejected": []})
            ab["accepted"].append({"from": frm, "to": to, "scope": "token", "count": 1})
            save_json(self.apath, ab)
            # cards whose only difference is this same pair are now settled
            pair = {x, y}
            rest = self.cards[self.i + 1:]
            keep = [k for k in rest if not (k.get("pair") and set(k["pair"]) == pair)]
            c["_cleared"] = [k for k in rest if k not in keep]
            self.cards[self.i + 1:] = keep
            extra = f" · cleared {len(c['_cleared'])} similar card{'s' * (len(c['_cleared']) != 1)}" if c["_cleared"] else ""
            self.say(f"📖 learned {frm} → {to}{extra}")

    def revert(self, c, d):
        key = "same" if d in ("yes", "learn") else "different"
        if self.dec[key] and self.dec[key][-1] == c["ids"]:
            self.dec[key].pop()
        save_json(self.dpath, self.dec)
        if d == "learn":
            ab = load_json(self.apath, {"accepted": [], "rejected": []})
            if ab["accepted"]:
                ab["accepted"].pop()
            save_json(self.apath, ab)
            idx = self.cards.index(c)
            self.cards[idx + 1:idx + 1] = c.pop("_cleared", [])

    def finish_lines(self):
        return [f"  decisions: {len(self.dec['same'])} same, {len(self.dec['different'])} different → {self.dpath}",
                "  rerun `dedupe` to apply them"]


def cmd_swipe_abbr(args):
    work = Path(args.work)
    cands = load_json(work / "abbr_candidates.json", None)
    if cands is None:
        sys.exit("No candidates yet — run `mine <file.csv>` first.")
    store = load_json(work / "abbreviations.json", {"accepted": [], "rejected": []})
    done = {(e["scope"], frozenset((e["from"], e["to"]))) for e in store["accepted"]}
    done |= {(e["scope"], frozenset((e["a"], e["b"]))) for e in store["rejected"]}
    cards = [c for c in cands if (c["scope"], frozenset((c["from"], c["to"]))) not in done]
    if args.only_likely:
        cards = [c for c in cards if c["likely"]]
    if not cards:
        print("Nothing left to swipe 🎉  (run `mine` again after dedupe/new data)")
        return
    with Term(anim=not args.no_anim) as t:
        AbbrSwiper(cards, t, work / "abbreviations.json").run()
    st = load_json(work / "abbreviations.json", {"accepted": [], "rejected": []})
    print(f"Saved: {len(st['accepted'])} abbreviations, {len(st['rejected'])} 'different' pairs → {work/'abbreviations.json'}")


def cmd_swipe_matches(args):
    work = Path(args.work)
    queue = load_json(work / "review_queue.json", None)
    if queue is None:
        sys.exit("No review queue yet — run `dedupe <file.csv>` first.")
    dec = load_json(work / "match_decisions.json", {"same": [], "different": []})
    done = {frozenset(p) for p in dec["same"] + dec["different"]}
    cards = [q for q in queue if frozenset(q["ids"]) not in done]
    if not cards:
        print("Nothing left to swipe 🎉  Rerun `dedupe` to apply your decisions.")
        return
    with Term(anim=not args.no_anim) as t:
        MatchSwiper(cards, t, work).run()
    d = load_json(work / "match_decisions.json", {"same": [], "different": []})
    print(f"Saved: {len(d['same'])} same, {len(d['different'])} different → {work/'match_decisions.json'}")
    print("Rerun `dedupe` to apply them.")


# ---------------------------------------------------------------------- evaluate


def cmd_evaluate(args):
    """Pairwise precision/recall of id_mapping.csv against a truth file (id,true_id)."""
    work = Path(args.work)
    _, mapping = load_csv(str(work / "id_mapping.csv"))
    _, truth = load_csv(args.truth)
    tcol = [c for c in truth[0] if c != "id"][0]
    t = {r["id"]: r[tcol] for r in truth}
    pred = {r["id"]: r["canonical_id"] for r in mapping}
    c2 = lambda n: n * (n - 1) // 2
    both = Counter((pred[i], t[i]) for i in pred if i in t)
    tp = sum(c2(n) for n in both.values())
    pp = sum(c2(n) for n in Counter(pred[i] for i in pred if i in t).values())
    ap = sum(c2(n) for n in Counter(t[i] for i in pred if i in t).values())
    p = tp / pp if pp else 1.0
    r = tp / ap if ap else 1.0
    print(f"pairwise precision {p:.3f}  (of pairs we merged, how many are true duplicates)")
    print(f"pairwise recall    {r:.3f}  (of true duplicate pairs, how many we merged)")
    print(f"rows {len(pred):,} → predicted {len(set(pred.values())):,} products, truth {len(set(t.values())):,}")


# -------------------------------------------------------------------------- demo


def cmd_demo(args):
    rng = random.Random(args.seed)
    brands = ["Acme Dairy", "Brookfield", "Golden Acre", "Hillside Farm", "Northpoint", "Sunny Valley", "Riverbend", "Oak & Ash"]
    syl = ["ka", "lo", "mi", "ren", "tor", "vel", "sa", "dun", "bri", "gal", "mor", "fen", "ash", "wyn", "cor", "tal"]
    while len(brands) < args.brands:
        b = "".join(rng.choice(syl) for _ in range(rng.randint(2, 3))).title() + rng.choice(["", " Foods", " Farm", " & Co"])
        if b not in brands:
            brands.append(b)
    brands = brands[: args.brands]
    T = [  # (category, subcategory, description template, variants, sizes)
        ("Dairy", "Cheddar", "{v} Cheddar Cheese {f}", [["Mild", "Mature", "Extra Mature"], ["Shredded", "Block", "Sliced"]], ["200g", "400g"]),
        ("Dairy", "Mozzarella", "Mozzarella {v} {f}", [["Ball", "Grated"], ["Original", "Light"]], ["125g", "250g"]),
        ("Dairy", "Milk", "{v} Milk {f}", [["Whole", "Semi Skimmed", "Skimmed"], ["Fresh", "UHT"]], ["1L", "2L"]),
        ("Dairy", "Butter", "{v} Butter {f}", [["Salted", "Unsalted"], ["Block", "Spreadable"]], ["250g", "500g"]),
        ("Dairy", "Yogurt", "{v} Yogurt {f}", [["Strawberry", "Vanilla", "Natural"], ["Greek Style", "Low Fat"]], ["4x125g", "500g"]),
        ("Bakery", "Bread", "{v} Bread {f}", [["Wholemeal", "White", "Seeded"], ["Sliced", "Thick Sliced"]], ["800g", "400g"]),
        ("Snacks", "Crackers", "{v} {f} Crackers", [["GF", "DF"], ["Original", "Sea Salt"]], ["150g"]),
        ("Drinks", "Beer", "{v} Craft Beer {f}", [["IPA", "APA"], ["Can", "Bottle"]], ["330ml", "440ml", "4x330ml"]),
        ("Confectionery", "Chocolate", "{v} Chocolate Bar {f}", [["Milk", "Dark"], ["Original", "Hazelnut"]], ["100g", "200g"]),
        ("Meat", "Chicken", "Chicken {v} {f}", [["Breast Fillets", "Thighs"], ["Free Range", "Organic"]], ["300g", "600g"]),
    ]
    ABBR = {"cheddar": "CHD", "mozzarella": "MOZZ", "shredded": "SHRD", "chocolate": "CHOC", "strawberry": "STRWB",
            "vanilla": "VAN", "original": "ORIG", "unsalted": "UNSLTD", "wholemeal": "WHLML", "chicken": "CHKN",
            "skimmed": "SKMD", "mature": "MAT", "organic": "ORG", "yogurt": "YOG", "breast": "BRST"}
    SUBABBR = {"Cheddar": "CHD", "Mozzarella": "MOZZ", "Chocolate": "Choc", "Chicken": "CHKN", "Yogurt": "Yog"}

    def size_variant(s):
        m = re.fullmatch(r"(\d+)x(\d+)(g|ml)", s)
        if m:
            return rng.choice([s, f"{m[1]} x {m[2]}{m[3]}", f"{m[1]}X{m[2]}{m[3].upper()}"])
        m = re.fullmatch(r"(\d+)(g|ml|L)", s)
        n, u = int(m[1]), m[2]
        opts = [s, f"{n} {u}", f"{n}{u.upper()}"]
        if u == "g" and n >= 100:
            opts.append(f"{n/1000:g}kg")
        if u == "ml" and n >= 100:
            opts.append(f"{n/10:g}cl")
        if u == "L":
            opts.append(f"{n*1000}ml")
        return rng.choice(opts)

    products = []
    for brand in brands:
        for cat, sub, tpl, (v1, v2), sizes in T:
            for a in v1:
                for b in v2:
                    for s in sizes:
                        if rng.random() < 0.55 or sub in ("Beer", "Crackers"):
                            products.append({"brand": brand, "category": cat, "subcategory": sub,
                                             "description": f"{brand} " + tpl.format(v=a, f=b) + f" {s}", "weight": s})
    rows, truth = [], []
    for pid, p in enumerate(products):
        copies = [dict(p)]
        for _ in range(rng.choices([0, 1, 2, 3], weights=[45, 30, 17, 8])[0]):
            q = dict(p)
            for _ in range(rng.randint(1, 3)):
                op = rng.choice(["blank", "blank", "abbr", "abbr", "subabbr", "size", "typo", "case", "exact"])
                if op == "blank":
                    q[rng.choice(["brand", "category", "subcategory", "weight"])] = ""
                elif op == "abbr":
                    words = q["description"].split()
                    idx = [i for i, wd in enumerate(words) if wd.lower() in ABBR]
                    if idx:
                        i = rng.choice(idx)
                        words[i] = ABBR[words[i].lower()]
                        q["description"] = " ".join(words)
                elif op == "subabbr" and q["subcategory"] in SUBABBR:
                    q["subcategory"] = SUBABBR[q["subcategory"]]
                elif op == "size":
                    sv = size_variant(p["weight"])
                    q["description"] = q["description"].replace(p["weight"], sv)
                    if q["weight"]:
                        q["weight"] = size_variant(p["weight"])
                elif op == "typo":
                    words = q["description"].split()
                    idx = [i for i, wd in enumerate(words) if len(wd) >= 7 and wd.isalpha()]
                    if idx:
                        i = rng.choice(idx)
                        k = rng.randrange(1, len(words[i]) - 1)
                        words[i] = words[i][:k] + words[i][k + 1:]
                        q["description"] = " ".join(words)
                elif op == "case":
                    q["description"] = q["description"].upper()
            copies.append(q)
        for q in copies:
            rows.append(q)
            truth.append(pid)
    order = list(range(len(rows)))
    rng.shuffle(order)
    out, tr = [], []
    for n, i in enumerate(order):
        rid = f"P{n+1:05d}"
        out.append({"id": rid, **rows[i]})
        tr.append({"id": rid, "true_id": truth[i]})
    write_csv(Path(args.out), ["id", "description", "brand", "category", "subcategory", "weight"], out)
    tpath = Path(args.out).with_name(Path(args.out).stem + "_truth.csv")
    write_csv(tpath, ["id", "true_id"], tr)
    print(f"Wrote {len(out):,} rows ({len(products):,} real products) → {args.out}")
    print(f"Ground truth → {tpath}  (use with `evaluate --truth`)")
    print("Includes traps: IPA vs APA, GF vs DF crackers, mild vs mature, 200g vs 400g, 0.2kg vs 200g, CHD vs Cheddar")


# --------------------------------------------------------------------------- cli


def main(argv=None):
    ap = argparse.ArgumentParser(prog="swipe_dedupe", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Run `python swipe_dedupe.py <command> -h` for options.")
    ap.add_argument("--work", default="dedupe_work", help="folder for dictionary, decisions and outputs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo", help="generate a sample supermarket CSV with duplicates and traps")
    p.add_argument("--out", default="demo_products.csv")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--brands", type=int, default=8, help="number of brands (~70 products each); 700 ≈ 60k rows")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("mine", help="find abbreviation/synonym candidates for the dictionary")
    p.add_argument("input")
    add_col_args(p)
    p.add_argument("--min-count", type=int, default=2, help="min contexts for non-abbreviation-looking pairs (default 2)")
    p.add_argument("--max-slot", type=int, default=12, help="skip slots with more distinct words than this (flavours etc.)")
    p.set_defaults(fn=cmd_mine)

    p = sub.add_parser("swipe-abbr", help="Tinder-style review of the abbreviation candidates")
    p.add_argument("--only-likely", action="store_true", help="only pairs that look like abbreviations")
    p.add_argument("--no-anim", action="store_true")
    p.set_defaults(fn=cmd_swipe_abbr)

    p = sub.add_parser("dedupe", help="run the pipeline and write deduped.csv / id_mapping.csv / review queue")
    p.add_argument("input")
    add_col_args(p)
    p.add_argument("--k", type=int, default=10, help="TF-IDF neighbours per description (default 10)")
    p.add_argument("--min-cos", type=float, default=0.5, help="min TF-IDF cosine for a candidate (default 0.5)")
    p.add_argument("--review-min", type=float, default=80, help="min fuzzy score to send a pair to review (default 80)")
    p.add_argument("--typo-max-freq", type=int, default=3,
                   help="a near-spelling only counts as a typo if the rarer form appears ≤ this many times (default 3)")
    p.set_defaults(fn=cmd_dedupe)

    p = sub.add_parser("swipe-matches", help="Tinder-style review of grey-zone pairs")
    p.add_argument("--no-anim", action="store_true")
    p.set_defaults(fn=cmd_swipe_matches)

    p = sub.add_parser("evaluate", help="precision/recall against a truth CSV (id,true_id)")
    p.add_argument("--truth", required=True)
    p.set_defaults(fn=cmd_evaluate)

    # allow --work after the subcommand too
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--work" in argv[1:]:
        i = argv.index("--work")
        if i > 0 and i + 1 < len(argv):
            argv = ["--work", argv[i + 1]] + argv[:i] + argv[i + 2:]
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye — progress is saved.")
