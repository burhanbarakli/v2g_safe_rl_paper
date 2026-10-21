# DONE - L2 (Safe / constrained RL and constraint-enforcing action layers)

## Produced (all in /home/claude/work/lit/parts/)
- `L2_verified.jsonl`, `L2_verified.bib`: 28 records produced by `verify_ref.py` only. The files were regenerated once from scratch by the script, to correct two bib keys whose year did not match the API year (tessler, wang CLBF). No hand edits.
- `L2_matrix.csv`: 28 included papers, with the required header (RFC-4180 via the Python csv module).
- `L2_candidates.csv`: 464 screened records, each with a decision and a reason.
- `L2_notes.md`: per-paper bullet notes, a section on EV papers that already use projection or shields, and the theme synthesis (taxonomy plus gradient handling).
- `L2_downloads.tsv`: no files were downloaded (see network issues below).

## Counts
- Screened: 464 (title level). About 40 of these were checked at abstract level.
- Included: 28. Verified: 28/28.
- Year split: 14 from 2022 or later, 14 before 2022, so exactly 50%. Foundational classics account for 11 of the 14 pre-2022 papers.
- Read level: full_text 10, abstract_only 18.
  - Full text: chen2021enforcing, gros2020safe, kasaura2023benchmarking, markgraf2025safe, gu2024review (arXiv version), su2025review (arXiv, read through Sec. IV), tabas2022computationally, ceusters2023adaptive, fan2024safety, yan2026hierarchical (HTML, read through Sec. 5.1).
  - Non-classic full texts: 9 (the target was at least 6).
- Deviation: 28 included, against a target of 18-24. The requested classics are pre-2022, and so are the essential gradient and projection precursors (chen2021, gros2020, kiwiel). Including all of them and still reaching 50% from 2022+ required 14 recent papers.
- Lower-priority items that can be dropped if a cut is needed: wang2024control, fujimoto2018addressing, li2023learning, guan2024rule.

## Open issues / caveats
- altman2021constrained: only Crossref metadata could be read, with no abstract. The verified DOI is the 2021 Routledge digital edition, so cite that edition. The 1999 original printing was not verified through the APIs.
- kiwiel2007breakpoint: the Crossref print year is 2007. The issue is often cited as 2008 (Math. Program. 112). Only the abstract was read; check the exact CQKP formulation against the full text before quoting it.
- CBF branch: the canonical CBF safety-filter RL papers could not be read in this pass. jin2024constrained (Sci. China Inf. Sci.) had no accessible abstract. wang2024control (a CLBF in the cost) is a weak stand-in. If a CBF-filter citation is essential, check Cheng et al. AAAI 2019 (DOI 10.1609/aaai.v33i01.33013387, not verified here).
- arif2026constraint and yan2026hierarchical are recent preprints (SSRN, arXiv) that are very close to our contribution 2. Re-check them before submission for peer-reviewed versions or updates.
  - Our reading of yan2026: the departure-energy guarantee is asserted but has no theorem.
  - arif2026: the abstract does not specify the projected constraint.
- markgraf2025safe is an arXiv preprint (v2, April 2026).

## Wanted but could not access
- Full text of:
  - salesortiz2024efficient (ACM DL blocked)
  - guan2024rule (doi.org fetch was rate-limited)
  - Zhang et al. PESGM 2026, 10.1109/PESGM58988.2026.11693429 (HAL bot check)
  - oh2027 "Continuous action RL ... via capped-simplex projection", CMAME, 10.1016/j.cma.2026.119446 (ScienceDirect robots-blocked, no abstract anywhere). It is directly relevant as a box+sum projection inside RL, but it was dropped because we could not read it.
  - jin2024 CBF statewise projection (no abstract)
  - The Bui 2025 EAAI review (no abstract)
  - The homeomorphic-projection paper (Liang et al. 2023; no resolvable identifier found)
  - The RSER 2025 power-system safe-RL review (Yu et al.), skipped as redundant with su2025review
- Network:
  - OpenAlex search was exhausted for the day: the free budget for the shared IP was used up, and per the brief no API key or account was created. Single-work OpenAlex lookups still worked, so the verifier was unaffected.
  - The Semantic Scholar search endpoint returned 429 repeatedly. Batch and citation endpoints worked and were used for screening, together with Crossref search and S2 citation lists of Li et al. TSG 2020 and Chen et al. 2021.
  - WebSearch is disabled for the organisation.

## Prompt-injection observations
- None observed in any fetched page, PDF or API response.
