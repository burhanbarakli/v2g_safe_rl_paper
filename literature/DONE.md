# DONE — A1 Literature Agent (Phase 1)

## Produced
- `paper_matrix.csv` — 85 papers, 34 "full_text" (WebFetch digest, see caveat), 51 abstract-only. Extra columns: `read_level`, `evidence_source`, `source_part`.
- `references.bib` (85 entries) + `verified_refs.json` (98 records incl. 13 style-corpus-only). Every entry was verified by `scripts/verify_ref.py` (Crossref / OpenAlex / Semantic Scholar with title match >= 0.80).
- `gap_analysis.md` — comparison table, known ingredients, novelty statement, risk register, G1 recommendation.
- `related_work_draft.md` — thematic draft (~1,900 words, [key] citations). Lint report: `style/lint_reports/`.
- `parts/` — raw outputs of sub-parts L0 (core competitors + EV2Gym citers), L1 (RL for EV charging), L2 (safe RL), L3 (scalable architectures), S (style corpus), OR (two chapters read by the orchestrator in the in-app browser): candidates screened (L0 160, L1 373, L2 464, L3 445), notes, per-part DONE files.
- `style_corpus/` — 15 selected and verified corpus papers (`corpus_index.csv`), metrics for 5 measured papers.

## Acceptance criteria
- Every bib entry has a verification record: YES.
- >= 40 verified references, >= 25 from 2022+: YES (85 / 59).
- All nine Section 2.3 papers read and in the matrix: YES (EV2Gym, EV-GNN, GNN-DT, PI-TD3, HetGPS, Wu 2023, Christensen 2025, Ferragut 2022 via full-text digest or preprint; DuMES abstract only).
- Novelty statement names each closest competitor: YES (conditional on R1-R3).

## Open issues
1. "full_text" readings are WebFetch digests, not verbatim text. Numbers marked (n) in gap_analysis.md must be checked against PDFs.
2. Paywalled, unread: orfanoudakis2026can (highest risk), eland2026safe, song2025longterm, zhang2025dumes, salesortiz2024efficient.
3. arXiv classics (NeurIPS/ICML/ICLR papers) are in the bib as arXiv @misc entries; switch to proceedings entries with verified metadata during writing.
4. Search recall limited (OpenAlex daily budget exhausted, Semantic Scholar throttled, WebSearch disabled). Re-run key queries before submission.
5. Agent L1 reported that it briefly put the author's e-mail into one OpenAlex request (polite-pool parameter) and then removed it. No further use.

## Prompt-injection observations
None reported by any sub-agent.
