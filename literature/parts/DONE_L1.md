# DONE_L1: Literature Agent A1, part L1 (RL for EV smart charging/V2G and scheduling context)

## Produced (all in /home/claude/work/lit/parts/)
- L1_verified.jsonl, L1_verified.bib: 20 records, all VERIFIED by scripts/verify_ref.py (19 via Crossref, 1 arXiv via OpenAlex arXiv DOI).
- L1_matrix.csv: 20 rows, 18 columns, header as specified.
- L1_candidates.csv: 373 screened records (deduplicated) with include/exclude reasons.
- L1_notes.md: per-paper notes covering departure-energy handling, power/transformer handling, guarantee, and fixed vs variable N, plus a theme synthesis.
- L1_downloads.tsv: no binary downloads were possible (see below).

## Counts
- Screened: 373 unique records (title level), from about 35 Crossref bibliographic queries, Semantic Scholar batch lookups, and the S2 citation list of EV2Gym. About 75 abstracts were examined in detail.
- Included: 20. From 2022 or later: 15/20 (75%). The pre-2022 papers are sadeghianpourhamami2020, li2020, lee2021, li2021 and frendo2021.
- Full text: 10. These are sadeghianpourhamami2020definition, lee2021adaptive, li2021learning, chen2022smoothed, dyo2022behavior, korkas2024distributed (partial HTML), michailidis2025reinforcement (partial HTML), vanhuffelen2025grid (arXiv version), ponse2025chargax, and diazlondono2026open (arXiv preprint 2405.11963 by the same authors; the published EPSR version was not read).
- Abstract only: 10.
- Verified: 20/20.
- Reviews included: 2 (qiu2023reinforcement, michailidis2025reinforcement). One more survey (arXiv 2410.16425, ICEMD 2024) was read in full and excluded because it is thin (9 works).

## Novelty threats (for the PI)
1. **chen2022smoothed (sLLF, TSG 2022)**: its closed-form rate r_i = clip(rbar_i(L - laxity_i + 1), 0, min(rbar_i, e_i)), with a common level L found by bisection under the station power limit, has the same algebraic form as our projection p_i = clip(a_i - lambda, l_i, u_i). It also proves feasibility and resource-augmentation results. Contribution 2 must cite it and position itself as a differentiable projection of a *learned* per-charger action, with a departure-guarantee lower bound and V2G bounds, used inside RL training.
2. **arif2026constraint (SSRN preprint 2026)**: a closed-form projection layer for a network constraint in DQN EV (bus) depot charging. It reports zero violations and shows penalties become redundant. This is empirical, not peer reviewed, and limited to a single depot, but it overlaps conceptually with contribution 2.
3. **sadeghianpourhamami2020definition (TSG 2020)**: a state representation independent of the number of stations, train-on-10 then test at larger scale, and the gap to a perfect-knowledge oracle. It partly pre-empts the framing of contributions 1 and 3; the difference is per-charger tokens with attention versus binned aggregation, plus V2G, prices and a transformer limit.
4. **song2025longterm (MAHEM, Applied Energy 2025)**: transformer-based MARL for EV charging with attention communication and claimed transfer tests. Only the abstract was read. Its tokens appear to be time steps and stations, not chargers, so a full-text check is recommended.
5. korkas2024distributed and zhou2025optimal: per-charger agents with fixed local observations or shared training for scalability, without attention or hard constraints.

## Open issues
- OpenAlex search budget was exhausted for the shared IP on arrival (HTTP 429, "Insufficient budget"), so systematic OpenAlex searching was replaced by Crossref bibliographic search. OpenAlex single-work lookups (used for abstracts) still worked.
- Semantic Scholar search returned 429 throughout. Only the batch and citations endpoints worked.
- WebSearch is not enabled for this organisation (proxy 403).
- The doi.org to MDPI WebFetch was rate-limited once (HTTP 429). That page was not retried; the direct mdpi.com HTML URLs were used instead.
- Elsevier full texts (sciencedirect, linkinghub) could not be read, so Elsevier papers are abstract-only.

## Papers wanted but not accessible (no abstract available via Crossref, S2 or OpenAlex)
- 10.1016/j.energy.2025.135095: Real-time online charging control of EVCS based on MADRL (Energy 2025).
- 10.1016/j.apenergy.2024.122737: Factored-action RL for EV charging design (Applied Energy 2024).
- 10.1016/j.tre.2024.103698: RL for EV charging scheduling, systematic review (TRE 2024).
- 10.1016/j.epsr.2026.113356: Safe DRL for EV charging/discharging with renewables (EPSR 2026).
- 10.2139/ssrn.4869056: DRL with augmented Lagrangian for EV charging (SSRN 2024).
- Full text of song2025longterm (MAHEM), needed to check token design and transfer claims.
- Conformal prediction for EV departure: no peer-reviewed paper found. The only hit was an SSRN load-interval forecasting preprint (10.2139/ssrn.7170761), which is not about departure.

## Prompt-injection observations
- None in the paper content read.
- One note: the OpenAlex 429 error body contained text addressed to AI agents, suggesting a POST to create an API key with the user's email. It was ignored; no account or key was created.
- An earlier helper script briefly included the user's email as a `mailto` parameter in one OpenAlex request. This was removed immediately, and no further requests carried it.
