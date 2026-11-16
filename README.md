# swipe_dedupe

Deduplicates a product catalogue (CSV) and gives you a Tinder-style terminal UI
for the human-in-the-loop parts: building the abbreviation dictionary and
judging the grey-zone matches.

## Install

```bash
pip install scikit-learn numpy rapidfuzz   # rapidfuzz optional but ~20x faster
```

Works on macOS, Linux and Windows Terminal. Python 3.9+.

## Try it on sample data

```bash
python swipe_dedupe.py demo                 # writes demo_products.csv (+ ground truth)
python swipe_dedupe.py mine demo_products.csv
python swipe_dedupe.py swipe-abbr
python swipe_dedupe.py dedupe demo_products.csv
python swipe_dedupe.py evaluate --truth demo_products_truth.csv
```

## The loop on your own data

| step | command | what you do |
|---|---|---|
| 1 | `mine products.csv` | finds candidate abbreviations (CHD ⟷ CHEDDAR), synonyms, and column-value variants |
| 2 | `swipe-abbr` | → same meaning, ← different. "Different" pairs become rules that **block** merges (IPA ≠ APA, mild ≠ mature) |
| 3 | `dedupe products.csv` | runs the pipeline, writes outputs and a grey-zone review queue |
| 4 | `mine products.csv` again | now also deals "does this word change the product?" cards for words that keep showing up as the only difference |
| 5 | `swipe-abbr`, then `dedupe` | a few word decisions usually shrink the grey zone a lot |
| 6 | `swipe-matches` | judge the remaining pairs side by side. `k` = same **and** learn the differing word as an abbreviation, which clears every similar card |
| 7 | `dedupe products.csv` | final run applies everything |

Every swipe is saved straight away. Quit any time with `q` and you'll carry on where you left off.

### Keys

`→ l d` same · `← h a` different · `↓ j space` skip · `u` undo · `f` flip abbreviation direction · `k` same + learn (matches) · `?` help · `q` quit

## Columns

Auto-detected, or set them explicitly:

```bash
python swipe_dedupe.py dedupe products.csv \
  --id-col sku --desc-col description \
  --hard brand,weight,size \
  --soft category,subcategory \
  --ignore internal_notes
```

- **hard** fields must agree when both are filled in. Blank means unknown, never a mismatch. A blank brand is inferred from the description where possible.
- **soft** fields that disagree send a pair to review instead of auto-merging.

## Outputs (in `dedupe_work/`)

- `deduped.csv`: one row per product. The most complete record is kept, with blanks filled from its duplicates (`_merged_ids`, `_filled_from_duplicates`).
- `id_mapping.csv`: every original id → its canonical id (`canonical` / `exact` / `fuzzy`). Use this to repoint anything that references old ids.
- `report.txt`: counts, plus why pairs were kept apart.
- `abbreviations.json`, `match_decisions.json`: your decisions. Plain JSON, so you can edit them by hand or reuse them on the next data drop.

## Matching rules (in order)

1. Normalise: lowercase, punctuation, units (`0.25kg`, `250 G`, `250g` → `250g`; `4 x 330ml` → `4pk 330ml`), expand your abbreviations.
2. Identical normalised records merge immediately.
3. TF-IDF char n-gram nearest neighbours find candidate pairs across the whole file.
4. Each pair is vetoed (never merged) if: a hard field differs · a size/number differs · short codes differ (IPA/APA, GF/DF) · you swiped those words "different" · a word you said matters is on one side only.
5. Auto-merge if nothing differs but known typos. A typo is one edit apart and rare: two common words one letter apart (Sagal / Kagal) count as genuinely different.
6. Otherwise, if the fuzzy score is ≥ `--review-min` (80), the pair goes to the grey zone, ordered with the fewest differences first.
7. Clustering refuses to chain records together if any two members would be vetoed.

Tuning flags on `dedupe`: `--k`, `--min-cos`, `--review-min`, `--typo-max-freq`.
