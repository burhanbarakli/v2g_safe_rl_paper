# Style profile (A6, task 1)

Status: **complete, built from 15 corpus papers (91,511 words)**: 11 from strong venues in our field
(IEEE T-ITS, IEEE TPWRS, IEEE TSG, IEEE TSTE, Applied Energy x2, Energy and AI x2, eTransportation,
Electric Power Systems Research, Communications Engineering) and 4 from the target journal (SAUCIS).
Recompute with `scripts/aggregate_style.py` if papers are added.

## Method
- Metrics were computed on the full article text; only the metrics are stored (no article text is kept).
  - The first 5 papers (orfanoudakis2025scalable, 4 SAUCIS papers) were measured in the browser with
    `scripts/style_stats_dom.js` (nature.com HTML DOM) and `scripts/style_stats.js` + pdf.js (DergiPark PDFs).
  - The other 10 were measured from the arXiv HTML (LaTeXML) versions of the published papers with
    `scripts/style_stats.py`, a Python port of the same logic (same section classifier, sentence splitter,
    word regex, phrase lists, passive and past-tense regexes). The published versions were verified by DOI
    (`lit/parts/S2_verified.jsonl`). Math, equations, figures, tables, footnotes, abstract, author block,
    acknowledgements, bibliography and appendices were removed before counting.
  - Two documented differences in the port: (i) LaTeXML heading level is taken from the title class
    (`ltx_title_section` etc.) because the tag level differs between document classes; (ii) a heading that
    contains "result" is classified as Results (the JS order sent "Experimental Results" to Setup). The 5
    browser-measured papers have no headings affected by (ii).
- Per-paper numbers: `style_corpus/metrics/*.json`; corpus list: `style_corpus/corpus_index.csv`;
  aggregate: `style/style_corpus_aggregate.json` (sentence-weighted means; a section enters the aggregate
  only if the paper has at least 8 sentences in it).
- Caveats: passive voice and past tense are regex approximations. Sections are classified by heading
  keywords, so a section titled after the artefact ("The EV2Gym Simulator") counts as "other" and is left out
  of the section table (it is in the word total). "Experiments" and "Case Studies" headings count as Setup and
  usually contain results too. Related work and Discussion rest on only 4 and 3 papers (most IEEE papers fold
  them into the Introduction and Results). One paper (ruddick2024realworld) uses author-year citations, which
  lengthens its sentences. PDF paragraph boundaries are approximate, so paragraph medians are indicative only.

## Corpus measured (91,511 words)
| Key | Venue | Words | Source |
|---|---|---|---|
| orfanoudakis2025scalable | Communications Engineering (Nature portfolio) | 6,888 | nature.com HTML |
| orfanoudakis2025ev2gym | IEEE Trans. Intelligent Transportation Systems | 5,180 | arXiv HTML 2404.01849 |
| orfanoudakis2026graph | Energy and AI | 5,157 | arXiv HTML 2502.01778 |
| orfanoudakis2026physics | Applied Energy | 5,115 | arXiv HTML 2510.12335 |
| diazlondono2026open | Electric Power Systems Research | 4,674 | arXiv HTML 2405.11963 |
| ruddick2024realworld | Energy and AI | 11,075 | arXiv HTML 2408.07435 |
| wang2025twostage | IEEE Trans. Power Systems | 7,166 | arXiv HTML 2507.18110 |
| yuan2026reinforcement | IEEE Trans. Sustainable Energy | 4,608 | arXiv HTML 2602.04578 |
| panda2026quantifying | eTransportation | 7,485 | arXiv HTML 2403.13367 |
| kamoona2025online | Applied Energy | 4,511 | arXiv HTML 2408.11828 |
| li2026multiview | IEEE Trans. Smart Grid | 7,622 | arXiv HTML 2511.22072 |
| bulut2024improving | SAUCIS | 5,020 | DergiPark PDF |
| zita2026explainable | SAUCIS | 4,009 | DergiPark PDF |
| arslan2025natural | SAUCIS | 3,996 | DergiPark PDF |
| lamraoui2026transformer | SAUCIS | 9,005 | DergiPark PDF |

Not measured (full text not reachable from the workspace): 8 entries of `corpus_index.csv` (marked there);
cihan2023timeseries (SAUCIS) was not needed once the corpus reached 15.

## Measured profile (sentence-weighted means, 15 papers)
| Section (papers) | Mean sentence (words) | p90 | Share >35 words | Sentences with we/our | Passive (approx.) | Past-tense verb (approx.) | This/These/It/There openers | Median paragraph (words) |
|---|---|---|---|---|---|---|---|---|
| Introduction (15) | 24.0 | 34.5 | 9.1% | 8.0% | 22% | 9% | 8.9% | ~98 |
| Related work (4) | 28.4 | 41.3 | 18.6% | 5.0% | 15% | 14% | 7.9% | ~107 |
| Method (13) | 21.6 | 33.6 | 8.4% | 6.7% | 43% | 11% | 10.6% | ~59 |
| Setup / case study (7) | 23.6 | 34.9 | 11.0% | 10.0% | 37% | 16% | 11.5% | ~97 |
| Results (9) | 24.7 | 37.1 | 11.7% | 8.0% | 27% | 22% | 11.5% | ~95 |
| Discussion (3) | 20.8 | 30.4 | 2.9% | 7.7% | 23% | 17% | 14.4% | ~54 |
| Conclusion (7) | 26.3 | 40.3 | 25.2% | 7.8% | 31% | 21% | 15.5% | ~61 |

Venue groups differ mainly in voice: the 11 strong-venue papers use "we/our" in 9-11% of sentences
(Introduction 9.2%, Method 8.7%, Results 9.8%), the 4 SAUCIS papers in 0-4%; SAUCIS uses more passive in
Results and Conclusion (36-46% vs 22-25%). Sentence length is similar in both groups (Method 21.4-21.6,
Introduction 22.5-24.4, Results 24.4-25.7).

## Rules derived for this manuscript (A5 must follow; A6 enforces)
1. **Sentence length.** Target mean 20-24 words per section (corpus 21.6-26.3, Related work 28.4); hard cap
   35 words except formal definitions. The corpus exceeds 35 words in 3-25% of sentences (9% in the
   Introduction, 8% in Method, 12% in Results, 25% in Conclusions); we stay below the corpus on this point
   (orchestrator Section 5 rule wins over the corpus).
2. **Voice.** The corpus uses "we" in 5-10% of sentences (strong venues 9-11%, SAUCIS 0-4%) and passive
   heavily in Method/Setup/Results (43%/37%/27%). For this paper: use "we" for contributions and design
   decisions (Introduction, Method choices, Discussion), aim for 10-25% "we" sentences there; use active
   voice with concrete non-human subjects elsewhere ("The projection layer clips ..."). Keep passive below
   30% per section.
3. **Tense.** Present tense for the method, the theorem, and established facts; past tense for experiments
   that were run and their outcomes ("PPO-Attn reached ..."); present tense when interpreting a figure
   ("Fig. 4 shows ...").
4. **Paragraphs.** 60-150 words, one topic each, topic sentence first. Hard cap 180.
5. **Openers.** Corpus has 8-16% This/These/It/There openers (Introduction 8.9%, Method 10.6%, Results
   11.5%, Conclusion 15.5%). Limit: at most one per paragraph; "This"/"These" always followed by a noun.
6. **Transitions (corpus rate per 1,000 words):** however 1.19, first/second 0.86/0.69, specifically 0.57,
   therefore 0.54, thus 0.47, overall 0.47, finally 0.36, in contrast 0.30, for example 0.28,
   similarly 0.25, as a result 0.18, for instance 0.17, hence 0.16. Use transitions at roughly these rates
   (about one every 1-3 paragraphs); do not open consecutive sentences with transitions.
7. **Limited words (per 1,000 words; corpus rate -> our cap):** significantly 0.64 -> 0.6 (statistical sense
   only); efficient 0.63 -> 0.6; robust 0.58 -> 0.6 (defined meaning only); effectively 0.56 -> 0.6;
   enhance 0.54 -> 0.5; additionally 0.48 -> 0.5; furthermore 0.47 -> 0.5; comprehensive 0.33 -> 0.3;
   leverage 0.31 -> 0.3; moreover 0.29 -> 0.3; novel 0.24 -> 0.3; efficiently 0.22 -> 0.3;
   facilitate 0.12 -> 0.3; utilize 0.37 -> 0 (use "use"). Banned-list words occur in the corpus (crucial
   0.27, notably 0.25, underscore 0.12, intricate 0.07, seamless 0.05, showcase 0.05 ...) but are banned
   here by Section 5.2.
8. **Numbers, units, statistics.** Corpus writes percentages without a space ("13.6%": 146 occurrences,
   0 with a space). Use SI units with a space ("50 kW", "15 min"), mean ± standard deviation over seeds with
   the number of seeds stated, 95% confidence intervals in brackets, test name and p-value ("Wilcoxon
   signed-rank, p < 0.01").
9. **Em dashes:** corpus 0.43 per 1,000 words; banned here (Section 5.2).
10. **Structure (SAUCIS papers):** numbered sections "1. Introduction ... n. Conclusion(s)", subsections
    "2.1."; SAUCIS papers end with an "Article Information Form" (see `paper/journal_requirements.md`).

## Phrasing patterns observed (described, not copied)
- Contributions: a short numbered or bulleted list at the end of the Introduction, each item starting with a
  verb or "A/An <artefact> that ...".
- Limitations: a dedicated short subsection or the final paragraphs of the Discussion, stated directly.
- Results: each paragraph opens with the figure/table reference and the main quantitative finding, then the
  explanation.
