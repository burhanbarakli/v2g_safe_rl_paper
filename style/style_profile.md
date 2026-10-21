# Style profile (A6, task 1) — PRELIMINARY

Status: **preliminary, built from 5 of the planned 13-15 corpus papers.** The remaining strong-venue papers
could not be measured yet (publisher sites are blocked from the cloud workspace; the in-app browser on the
author's computer was used for the 5 below and disconnected before the rest were done). The profile will be
recomputed with `scripts/aggregate_style.py` once the corpus is complete (needed before Phase 5, not before G1).

## Method
- Metrics were computed on the full article text inside the browser (HTML DOM for nature.com, pdf.js text
  extraction for DergiPark PDFs). Only the metrics left the page; no article text was copied.
  Scripts: `scripts/style_stats_dom.js` (HTML), `scripts/style_stats.js` + pdf.js extractor (PDF),
  aggregation `scripts/aggregate_style.py`; raw per-paper numbers in `literature/style_corpus/metrics/*.json`;
  aggregate in `style/style_corpus_aggregate.json`.
- Sentence splitter, word counter and phrase lists are the same as in `style/lint.py`.
- Caveats: passive voice and past tense are regex approximations; PDF paragraph boundaries are approximate
  (some paragraphs merge across page breaks), so paragraph medians are indicative only.

## Corpus measured so far (28,918 words)
| Key | Venue | Words | Source |
|---|---|---|---|
| orfanoudakis2025scalable | Communications Engineering (Nature portfolio) | 6,888 | nature.com HTML |
| bulut2024improving | SAUCIS | 5,020 | DergiPark PDF |
| zita2026explainable | SAUCIS | 4,009 | DergiPark PDF |
| arslan2025natural | SAUCIS | 3,996 | DergiPark PDF |
| lamraoui2026transformer | SAUCIS | 9,005 | DergiPark PDF |
Pending: wang2025datadriven, jacob2024realtime (Nature Communications), and arXiv HTML versions of journal papers
(EV2Gym/T-ITS, GNN-DT/Energy and AI, PI-TD3/Applied Energy, Diaz-Londono/EPSR), plus 1-2 open-access IEEE/MDPI papers.

## Measured profile (sentence-weighted means)
| Section | Mean sentence (words) | p90 | Share >35 words | Sentences with we/our | Passive (approx.) | Past-tense verb (approx.) | This/These/It/There openers | Median paragraph (words) |
|---|---|---|---|---|---|---|---|---|
| Introduction | 23.1 | 32.6 | 6.7% | 7.3% | 23% | 19% | 7.3% | ~120 |
| Related work | 25.8 | 35.5 | 11.6% | 3.2% | 11% | 7% | 10.6% | ~70 |
| Method | 22.0 | 34.0 | 8.6% | 2.1% | 41% | 19% | 13.7% | ~80 |
| Results | 24.8 | 38.0 | 11.4% | 1.7% | 29% | 21% | 14.2% | ~96 |
| Discussion | 20.8 | 30.4 | 2.9% | 7.7% | 23% | 17% | 14.4% | ~54 |
| Conclusion | 27.1 | 39.2 | 28.2% | 2.6% | 46% | 31% | 15.4% | n/a |

## Rules derived for this manuscript (A5 must follow; A6 enforces)
1. **Sentence length.** Target mean 20-24 words per section (corpus 21-25); hard cap 35 words except formal
   definitions. The corpus exceeds 35 words in 3-28% of sentences; we stay below the corpus on this point
   (orchestrator Section 5 rule wins over the corpus).
2. **Voice.** Corpus uses "we" sparingly (2-8% of sentences) and passive heavily in Method/Results (29-41%).
   For this paper: use "we" for contributions and design decisions (Introduction, Method choices, Discussion),
   aim for 10-25% "we" sentences there; use active voice with concrete non-human subjects elsewhere
   ("The projection layer clips ..."). Keep passive below 30% per section.
3. **Tense.** Present tense for the method, the theorem, and established facts; past tense for experiments
   that were run and their outcomes ("PPO-Attn reached ..."); present tense when interpreting a figure
   ("Fig. 4 shows ...").
4. **Paragraphs.** 60-150 words, one topic each, topic sentence first. Hard cap 180.
5. **Openers.** Corpus has 7-15% This/These/It/There openers. Limit: at most one per paragraph; "This"/"These"
   always followed by a noun.
6. **Transitions (corpus rate per 1,000 words):** however 0.90, first/second 0.73/0.97, specifically 0.62,
   in contrast 0.38, thus 0.35, therefore 0.31, for instance 0.31, as a result 0.21, finally 0.21.
   Use transitions at roughly these rates (about one every 1-3 paragraphs); do not open consecutive
   sentences with transitions.
7. **Limited words (per 1,000 words; corpus rate -> our cap):** significantly 1.11 -> 1.0 (statistical sense
   only); enhance 1.04 -> 1.0; effectively 0.87 -> 0.8; efficient 0.87 -> 0.8; robust 0.83 -> 0.8 (defined
   meaning only); additionally 0.62 -> 0.6; comprehensive 0.59 -> 0.5; furthermore 0.28 -> 0.3; novel 0.24 -> 0.3;
   leverage 0.24 -> 0.3; moreover 0.07 -> 0.3; utilize -> 0 (use "use"). Banned-list words (crucial, notably,
   underscore, intricate, landscape, seamless, meticulous ...) occur in the corpus but are banned here by
   Section 5.2.
8. **Numbers, units, statistics.** Corpus writes percentages without a space ("13.6%"). Use SI units with a
   space ("50 kW", "15 min"), mean ± standard deviation over seeds with the number of seeds stated, 95%
   confidence intervals in brackets, test name and p-value ("Wilcoxon signed-rank, p < 0.01").
9. **Em dashes:** corpus 1.1 per 1,000 words; banned here (Section 5.2).
10. **Structure (SAUCIS papers):** numbered sections "1. Introduction ... n. Conclusion(s)", subsections
    "2.1."; SAUCIS papers end with an "Article Information Form" (see `paper/journal_requirements.md`).

## Phrasing patterns observed (described, not copied)
- Contributions: a short numbered or bulleted list at the end of the Introduction, each item starting with a
  verb or "A/An <artefact> that ...".
- Limitations: a dedicated short subsection or the final paragraphs of the Discussion, stated directly.
- Results: each paragraph opens with the figure/table reference and the main quantitative finding, then the
  explanation.
