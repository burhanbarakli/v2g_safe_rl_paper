# Style profiles (summary-level, NOT corpus text)

Status: no full text of any of the 15 corpus papers could be obtained in the sandbox. curl to every publisher and PDF host (dergipark.org.tr, saucis.sakarya.edu.tr, nature.com, sciencedirect.com, api.elsevier.com, ieeexplore, mdpi, springer, wiley, core.ac.uk, europepmc, zenodo, web.archive.org) returned proxy 403. WebFetch returns only model summaries, refuses verbatim text, and is blocked by robots.txt on sciencedirect.com and saucis.sakarya.edu.tr. As a stopgap, the structure of six papers was profiled through WebFetch (accessed 2026-10-21). All figures below are the WebFetch model's estimates, not counts from real text. To build the real corpus, use `scripts/build_style_corpus.py` once a human has downloaded the PDFs.

## SAUCIS conventions observed (5 papers)

| key | Vol(Issue), pages | Sections (top level) | Abstract / body words (est.) | Kw | Fig / Tab / Eq / Refs | Voice | Contributions | Back matter |
|---|---|---|---|---|---|---|---|---|
| cihan2023timeseries | 2023 (published online 30 Apr 2023) | 1 Introduction, 2 Related Work, 3 Materials and Methods (3.1–3.3), 4 Experiments, 5 Discussion, 6 Conclusion | ~200 / ~2,400–2,700 | – | 6 / 13 / 2 / 47 | mostly passive | bulleted list of 5, infinitive verbs ("To propose…") | Conflict of Interest; Availability of Data and Material; Ethical Approval and Informed Consent; Plagiarism Statement (no AI statement) |
| bulut2024improving | 7(3), 325–337 | 1 Introduction, 2 Material and Methods (2.1–2.4), 3 Results and Discussions, 4 Conclusions | ~290 / ~3,000–3,500 | 4 | 12 / 2 / 13 / 32 | impersonal/passive, no "we" | prose at end of Introduction | Article Information Form: Author(s) Contributions; Conflict of Interest Notice; Ethical Approval; Availability of data and materials; Plagiarism Statement (no AI statement) |
| arslan2025natural | 8(4), 773–784 | 1 Introduction, 2 Method (2.1–2.2), 3 Modeling and Consumption Forecasting, 4 Approach and Results, 5 Discussion, 6 Conclusions | ~190 / ~3,400 | 5 | 5 / 3 / 9 / 28 | passive + sparse "our" | prose | Article Information Form: Authors Contributions; Acknowledgments; Conflict of Interest Notice; Ethical Approval; Availability of data and material; Artificial Intelligence Statement ("No artificial intelligence tools were used while writing this article."); Plagiarism Statement |
| zita2026explainable | 9(2), 364–376 | 1 Introduction, 2 Related Works (2.1–2.5), 3 Proposed Methodology (3.1–3.7 incl. algorithm), 4 Results…, 5 Discussion and Findings, 6 Limitations and Future Work, 7 Conclusion | ~190–200 / ~3,900 | 6 | 4 / 6 / 4 / 25 (+1 algorithm) | "we" in intro, passive in methods | prose | Article Information Form: Authors Contributions; Conflict of Interest Notice; Ethical Approval; Artificial Intelligence Statement (same "No AI tools…" sentence) |
| lamraoui2026transformer | 9(3), 846–867 | 1 Introduction, 2 Related Works, 3 Proposed T-CCCRP (3 phases, sub-modules), 4 Experimental Setup, 5 Results and Discussion, 6 Conclusion and Future Work | ~200 / ~7,000 | 6 | 25 / 5 / 13 / 30 | mostly passive, occasional "we" | prose paragraph | Article Information Form: Author's Contributions; Acknowledgments; Conflict of Interest; Support Section; Ethical Approval; Artificial Intelligence Statement; Plagiarism Statement |

Sources: http://saucis.sakarya.edu.tr/en/download/article-file/2792201, https://dergipark.org.tr/en/download/article-file/4013362, https://dergipark.org.tr/en/download/article-file/5130772, https://dergipark.org.tr/en/download/article-file/5030792, https://dergipark.org.tr/en/download/article-file/5216396.

Take-aways for our manuscript:
- Typical SAUCIS ML papers run 12–22 journal pages, with a body of about 2,500–7,000 words, 25–47 references and abstracts of about 190–290 words (the limit is 400).
- The usual skeleton is Introduction, then Related Work(s), then Method / Proposed approach, then Experimental setup / Results, then Discussion (sometimes "Limitations and Future Work"), then Conclusion, then the Article Information Form, then References.
- Prose is formal and mostly passive. "We" is used sparingly, mainly in the Introduction.
- The WebFetch summaries reported a few mixed author-year citations in otherwise IEEE-numeric papers. We should stay strictly IEEE numeric.
- The 2025–2026 papers carry an Artificial Intelligence Statement.

## Strong-venue example (1 paper profiled; page truncated)

- wang2025datadriven (Nature Communications 2025, https://www.nature.com/articles/s41467-025-58192-9): Abstract (~190 words), Introduction, Results (5 subsections incl. 3 cases), Discussion, Methods (model, problem formulation, algorithm, baselines). It has 8 figures, ≥ 22 numbered equations (all in Methods) and ≥ 49 references. The voice is mostly "we". Contributions are given as a narrative plus a numbered list of four key features.
- WebFetch could not profile the other nine strong-venue papers: sciencedirect is robots-disallowed, and the nature.com pages for the Communications Engineering and Nature Communications 2024 papers returned a client error.
