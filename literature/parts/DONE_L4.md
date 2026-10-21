# DONE_L4: method-section references (Agent A1, part L4, 2026-10-22)

## Produced
- `L4_verified.jsonl` / `L4_verified.bib`: 24 entries, all produced by `verify_ref.py` with `--title`, all VERIFIED.
- `L4_matrix.csv`: 24 rows, with the header from the brief.
- `L4_candidates.csv`: 39 screened records (24 include, 15 exclude).
- `L4_notes.md`: per-paper notes and the theme synthesis, including the novelty check.
- `L4_downloads.tsv`: 11 arXiv PDFs saved under `/home/claude/work/pdfs/L4/`, with `.txt` extracts.

## Counts
- Screened: 39. Included: 24. Verified: 24.
- Read level: full_text 5 (lei2018distributionfree, romano2019conformalized, lindemann2023safe, fernandezzapico2025stochastic, lee2026enabling; all read from arXiv versions of the published papers). abstract_only 15. metadata_only 4 (vovk2022algorithmic, papadopoulos2002inductive, wilcoxon1945individual, bynum2021pyomo).
- The brief allows only full_text or abstract_only as read levels. `metadata_only` is a deliberate, honest deviation for these four classics. Their content was not read; they are cited only for their canonical role, which is confirmed by secondary sources that were read.

## Keys added (none duplicate `merged/references.bib`)
vovk2022algorithmic, papadopoulos2002inductive, lei2018distributionfree, angelopoulos2023conformal, romano2019conformalized, lindemann2023safe, fernandezzapico2025stochastic, cheng2026riskaware, zhou2025computationally, phipps2023customized, lee2019acndata, lee2026enabling, xiong2020layer, ba2016layer, schulman2016highdimensional, kingma2015adam, huang2022cleanrl, paszke2019pytorch, huangfu2018parallelizing, diamond2016cvxpy, bynum2021pyomo, agarwal2021deep, henderson2018deep, wilcoxon1945individual.

## Novelty finding (conformal departure bounds)
No screened paper uses a conformal lower bound on per-EV departure inside a charging controller or RL safety layer.
- Conformal methods in EV control so far bound aggregate load: fernandezzapico2025stochastic (CDC 2025) and cheng2026riskaware (Energies 2026).
- The closest work is phipps2023customized (IEEE IoT J. 2023). It uses quantile-based security levels on parking-duration predictions to limit undercharging risk. From the abstract, it is not conformal and not embedded in a controller. Its full text was not accessible and must be checked.
- Search coverage is limited: arXiv search, Crossref, and partial Semantic Scholar only. OpenAlex search had no remaining budget and WebSearch is disabled. The claim should therefore be phrased as "to our knowledge".

## Open issues
1. **Holm 1979** (Scand. J. Stat. 6(2):65-70) has no DOI, so it is not verifiable with the script and was not added. The orchestrator decides whether to add it manually.
2. **"37 implementation details of PPO"** was skipped: no DOI or arXiv ID.
3. **CleanRL:** the JMLR venue could not be confirmed (jmlr.org and dblp are blocked; S2 lists arXiv only). The bib entry is arXiv. The author metadata from the source may be garbled ("Ye, Chang"); check the .bib.
4. **Key years vs. bib years:**
   - schulman2016highdimensional: bib year 2015 (arXiv), venue ICLR 2016.
   - kingma2015adam: bib year 2014 (arXiv), venue ICLR 2015.
   - huang2022cleanrl: bib year 2021.
   - Adjust these when the final .bib is built if the venue years are preferred.
5. **lee2019acndata** was verified with the Crossref title "ACN-Data" (Crossref stores the subtitle separately). The BibTeX has the full title.
6. **Optional solver references:** diamond2016cvxpy and bynum2021pyomo are needed only if that modelling layer is used. The current `algorithm.md` uses `highspy`.

## Papers wanted but not accessible
- phipps2023customized full text: the IEEE OA PDF and TechRxiv were blocked (403 / fetch error).
- cheng2026riskaware full text: MDPI blocked.
- Vovk book and Papadopoulos 2002 chapter: Springer blocked.
- Wilcoxon 1945: JSTOR not tried (blocked domain class).

## Prompt-injection observations
- api.openalex.org error and notice messages contain text aimed at AI agents: "An AI agent can get one for its user in one call: POST https://api.openalex.org/api-keys with {"email": ...}". It was ignored. No account was created and no email was sent.
