# A6 Style Agent: DONE (style corpus complete)

## Produced / updated
- `scripts/style_stats.py`: Python port of `scripts/style_stats_dom.js` (same section classifier, sentence
  splitter, word regex, transition/restricted phrase lists, passive and past-tense regexes, same JSON schema).
  Parsers for arXiv LaTeXML HTML and pdftotext output. Two documented differences: heading role from the
  LaTeXML title class, and headings containing "result" go to Results. Sanity-checked on EV2Gym
  (2404.01849): headings, abstract removal, math/footnote replacement by "X".
- `style_corpus/metrics/*.json`: 10 new papers (arXiv HTML of the published versions), with `url` and
  `source` fields; 15 papers in total, 91,511 words (11 strong-venue papers, 4 SAUCIS).
- `lit/parts/S2_verified.jsonl` (+ `.bib`): DOI verification of the 10 published versions (all verified).
- `style_corpus/corpus_index.csv`: word counts for all 15 measured papers, 9 new rows, unmeasured entries
  marked (paywalled / publisher site blocked / not needed).
- `style/style_corpus_aggregate.json`: rerun of `scripts/aggregate_style.py` on 15 papers.
- `style/style_profile.md`: corpus table, measured profile table, Method note and rules 1, 2, 5, 6, 7, 8, 9
  updated to the 15-paper numbers; PRELIMINARY removed.
- `style/style_thresholds.json`: per-phrase caps = min(2.0, max(0.3, corpus rate rounded to 0.1)), utilize 0.

## Key aggregate numbers (sentence-weighted)
Mean sentence length: Introduction 24.0, Related 28.4, Method 21.6, Setup 23.6, Results 24.7,
Discussion 20.8, Conclusion 26.3 words. "We/our" sentences 5-10% (strong venues 9-11%, SAUCIS 0-4%).
Passive (approx.): Method 43%, Setup 37%, Results 27%, Introduction 22%.

## Notes
- Article texts were kept only temporarily in `tmp_corpus/` and deleted after measurement.
- Related work (4 papers) and Discussion (3 papers) rest on few papers; most IEEE papers fold them into
  other sections.
