# DONE_L3: Scalable / permutation-invariant / attention-based policies and size generalisation

## Produced (all in /home/claude/work/lit/parts/)
- L3_verified.jsonl and L3_verified.bib: 19 records, all VERIFIED by verify_ref.py (title match 1.0). DOIs were checked via Crossref; arXiv IDs via OpenAlex (arXiv DOI).
- L3_matrix.csv: 19 rows, with the required header.
- L3_candidates.csv: 445 screened records (19 include, 426 exclude), each with a reason.
- L3_notes.md: per-paper bullet notes plus "Theme synthesis".
- L3_downloads.tsv: header only. No files were downloaded: curl to nature.com, mdpi.com, sciencedirect, api.elsevier.com, ieeexplore, dl.acm.org, europepmc and PMC was refused by the proxy, and all full texts were read with WebFetch.

## Counts
- Screened: 445 (Crossref bibliographic searches, Semantic Scholar search/batch/citation endpoints, and citers of EV-GNN, EV2Gym, Sadeghianpourhamami 2020, Tuchnitz 2021 and T-MAAC).
- Included: 19. That is 6 foundational (vaswani2017attention, zaheer2017deep, lee2019set, kool2019attention, hu2021updet, yang2018mean), 2 on size generalisation (joshi2022learning, luo2023neural) and 11 EV/energy.
- Non-foundational from 2022 or later: 11 of 13 (85%).
- full_text: 7, all EV/energy: sadeghianpourhamami2020definition, ye2022learning, zhang2022transfer, wang2022stabilizing, jin2026telemetry, yan2026hierarchical, valesalonso2026emergent.
- abstract_only: 12.
- Verified: 19 of 19.

## Key findings for the paper
- Size transfer with the same weights, in EV and energy work:
  - [sadeghianpourhamami2020definition]: a policy trained with 10 stations is tested on session data scaled up to 10x (aggregate binned state).
  - [zhang2022transfer]: 509 EVs to 5,090-50,900 EVs, using an aggregator-level scalar action, rule-based dispatch and loads scaled 100x. Whether retraining was used at large scale is unclear.
  - [valesalonso2026emergent]: 4 training cities to 16 unseen cities (3-10 stations, about 21-138 vehicles), with density swept from 5 to 50 EVs per station using a shared per-vehicle policy.
  - [jin2026telemetry]: power, not EV. Feeders of 15-141 buses to a 322-bus feeder, 6 to 38 DERs, with a permutation-equivariant graph actor.
  - [ye2022learning] evaluates 10-100 chargers with a shared per-charger DQN, but the paper does not clearly say whether the weights were reused across sizes.
- Hard constraints in V2G RL: [yan2026hierarchical] (HPC-RL) combines power-flow projection with per-EV dynamic energy bounds of the same form as our l_i/u_i. Its only formal result is Prop. 1, first-order feasibility; the demand guarantee is claimed without proof. Fleet size is fixed and there is no attention.
- No paper in L3 combines a size-agnostic attention policy with a provable guarantee for both per-EV departure energy and the station limit. The nearest are [jin2026telemetry] (a heuristic one-step AC power-flow screen, not a certificate) and [yan2026hierarchical] (fixed size).

## Open issues
- Bib years for arXiv entries come from OpenAlex arXiv-DOI records:
  - vaswani2017attention shows year 2025 in the bib; it was first submitted in 2017 per arXiv.
  - lee2019set and kool2019attention show 2018 (arXiv v1).
  - The editor may prefer the conference versions. Semantic Scholar lists NeurIPS 2017 for Vaswani, ICML for Set Transformer, ICML for Yang 2018 and NeurIPS 2023 for LEHD. Those venues were not verified by the script.
- hassouna2026graph: Crossref gives 2026 (Energy and AI, vol. issued 2026), while the arXiv preprint (2407.04522) dates from 2024.
- zhang2022transfer and valesalonso2026emergent are arXiv-only. zhang2022transfer has internal numeric inconsistencies, described in the notes. Use it cautiously.
- Search recall is limited:
  - OpenAlex search returned "Insufficient budget" for the shared IP all session, though single-work lookups inside verify_ref.py worked.
  - WebSearch is disabled for the organisation (HTTP 403).
  - The Semantic Scholar search endpoint was throttled (repeated 429). Batch and citation endpoints worked.
  - A rerun of OpenAlex topic searches after the budget resets (midnight UTC) is recommended.

## Wanted but could not access (no abstract available or page blocked)
- Song et al. 2025, Applied Energy, transformer-based MARL for multi-station EV charging (10.1016/j.apenergy.2025.126315). No abstract in Crossref or S2, and ScienceDirect is blocked. This is the most important gap.
- Xie, Ajagekar, You 2023, Applied Energy, multi-agent attention DRL for demand response (10.1016/j.apenergy.2023.121162). No abstract.
- Eze, Prakash, Monti 2026, Energy and AI, GNN-MARL active voltage control topology robustness (10.1016/j.egyai.2026.100835). No abstract; doi.org returned 429 and sciencedirect is robots-blocked.
- Dynamic potential graphs for large-scale EV stations, Applied Energy 2026 (10.1016/j.apenergy.2026.128219). No abstract.
- Wang et al. 2023, Applied Energy, transfer learning for EV charging DRL (10.1016/j.apenergy.2023.121186). No abstract.
- Federated GNN-CMDP EV coordination, Sci. Rep. 2025 (10.1038/s41598-025-22482-5). Abstract read; full text blocked (nature.com client error; Europe PMC 429).
- Full texts of tuchnitz2021development and yan2022cooperative (publisher sites blocked).
- Pages that could not be read: https://www.ebi.ac.uk/europepmc/... (WebFetch proxy 429) and https://doi.org/10.1016/j.egyai.2026.100835 (WebFetch proxy 429). arXiv PDF 2609.07689 had no machine-readable text, so the HTML version was used.

## Prompt-injection observations
- None. No fetched paper or page contained instructions aimed at an AI agent. The only instruction-like text came from the WebFetch proxy's own rate-limit error messages, which are tool infrastructure, not document content.
