# Citation fixes, number verification and bibliography upgrade (A1 revision task)

Literature Agent A1, 2026-10-22. Inputs: `review/consistency_report.md` (M10, M11, M12, M14), `merged/gap_analysis.md`, `merged/paper_matrix.csv`, manuscript sections in `paper_repo/manuscript/sections/`. The manuscript was not edited.

Reading levels used below:
- **verbatim**: the arXiv HTML was downloaded with curl, converted to text and searched with grep. The text is the paper's own text, not a digest. Copies are in `/home/claude/work/pdfs/L5/` (`arxiv_<id>.html/.txt`).
- **digest**: a WebFetch model summary. Numbers from a digest are marked (d).
- **API**: Crossref / Semantic Scholar / OpenAlex metadata or abstract.

---

## 1. Corrected sentences

### M10 (01:11): transformer-limit penalties over-attributed

Current: "Most RL controllers for EV charging encode the departure energy requirement and the transformer limit as weighted penalties in the reward [sadeghianpourhamami2020definition, ye2022learning, korkas2024distributed, orfanoudakis2025ev2gym]."

What the cited papers do:
- sadeghian2020, ye2022, korkas2024: penalty on unmet departure energy only. No transformer or station cap is modelled (paper_matrix; consistency report M10).
- EV2Gym (verbatim, arXiv 2404.01849, Sec. IV, Eq. (27)): the V2G reward is r_t = c_{t-1} - 100 x (transformer overload) - 100 x exp(-10 x user satisfaction). Both requirements are penalties.
- GNN-DT (verbatim, arXiv 2502.01778v3, Sec. 3.2, Eqs. (2)-(4)): overload penalty f2 = sum max{0, p_sum - p*} with weight 100, and a squared departure-energy deviation f3 with weight 10. Both are penalties.
- EV-GNN: our two readings disagree. The Research Square preprint (digest of the PDF, see Section 2.3) uses a setpoint-tracking reward with **no** overload or satisfaction term (Methods, Eq. 10, "the reward function does not require an explicit term to maximize user satisfaction"). The orchestrator's earlier reading of the published page (ORCH in gap_analysis) reported heavily weighted overload and satisfaction penalties. I could not reach the published text in this session (Section 3).

Proposed sentence:

> Most RL controllers for EV charging encode the departure energy requirement as a weighted penalty in the reward \cite{sadeghianpourhamami2020definition,ye2022learning,korkas2024distributed,orfanoudakis2025ev2gym}, and those that model a transformer or station limit usually penalise overloads in the same way \cite{orfanoudakis2025ev2gym,orfanoudakis2026graph}.

Add `orfanoudakis2025scalable` to the second citation group only after someone confirms the overload penalty in the published Methods section (open access at nature.com, DOI 10.1038/s44172-025-00457-8). Do not add `wang2026hetgps`: HetGPS does not constrain transformer loading at all (App. B: "ampacity and thermal limits are not control constraints").

### M11 (02:27-28): TD3 citation carries the claim of another paper

Current: "The EV2Gym authors have also added a safety layer to a graph-enhanced twin delayed deep deterministic (TD3) agent \cite{fujimoto2018addressing}. According to its abstract, ..."

`fujimoto2018addressing` only defines TD3. The safety-layer claim comes from the abstract of `orfanoudakis2026can` (re-read on link.springer.com today: "a safety layer designed to ensure grid stability, preserve EV charging requirements, and enforce power limits"; TD3 enhanced with GNNs; trained and evaluated on real charging sessions; no mechanism, guarantee, sizes or simulator named). The chapter's authors are Orfanoudakis, Elders, Palensky and Vergara, three of whom are EV2Gym authors. The full name of TD3 is "twin delayed deep deterministic policy gradient".

Proposed sentences (replace 02:27-28):

> Members of the EV2Gym team have also combined a graph-enhanced twin delayed deep deterministic policy gradient (TD3) agent \cite{fujimoto2018addressing} with a safety layer \cite{orfanoudakis2026can}.
> According to the chapter abstract, the layer is meant to preserve charging requirements and enforce power limits, but the abstract describes neither its mechanism nor a formal guarantee \cite{orfanoudakis2026can}.

The TODO at 02:29 stays, because the full text is still unread (Section 3).

### M12 (02:50-52, Table 1 rows 86-88): size ranges and digest numbers

Checked against the papers' own text (Section 2). Proposed replacements:

02:50 (EV-GNN):
> The heterogeneous graph policy (EV-GNN) of \cite{orfanoudakis2025scalable} handles up to 1000 charge points under 70 transformer groups, but in the version we read it was trained separately at each scale and enforces no constraint by construction.

(Scales 25/100/500/1000 charge points with 3/7/35/70 transformer groups come from the preprint, Results and Methods "Simulation environment" (d). The preprint has no cross-scale evaluation (d). Drop "with a reward that penalises transformer overload and unmet demand" unless the published Methods confirm it. See M10.)

02:51 (GNN-DT, verified):
> In contrast, the graph-enhanced decision transformer (GNN-DT) of \cite{orfanoudakis2026graph}, trained on 25 chargers, was tested without retraining on 5 to 100 chargers, and its performance degraded as size grew.

(arXiv 2502.01778v3 Sec. 6.8, Fig. 6(b): "trained on a 25-charger setup, was tested on 5, 50, 75, and 100-charger environments ... performance predictably declines at larger scales". The 250-charger case, Table 8, was retrained. The journal abstract confirms that the method "generalizes to unseen fleet sizes".)

02:52 (HetGPS, verified):
> A HetGPS policy trained on an 8-transformer feeder with 1,236 vehicles transfers without retraining to 16- and 32-transformer feeders with 2,239 and 3,218 vehicles, using the same 383,702 parameters at every scale \cite{wang2026hetgps}, and a graph actor for voltage control trained on feeders of 15 to 141 buses has been evaluated in simulation on a 322-bus feeder without retraining \cite{jin2026telemetry}.

("Deployed" in the current 02:52 suggests a field deployment. jin2026 reports only pandapower simulation. Checked on mdpi.com today (d): seven 15-141-bus source feeders, a 322-bus zero-shot target "approximately 2.28 times larger", and the statement that the screen is not "a formal multi-step safety certificate".)

Table 1 (02:86-88), cell updates:
- EV-GNN, "Zero-shot size test": keep "not in version read". "Departure energy / Transformer limit = penalty" holds for the published version only per ORCH. In the preprint the setpoint reward has neither term. Mark it "(published version, not re-checked)" or verify it first.
- GNN-DT, "Zero-shot size test": "25 → 5–100 chargers". "Gap to optimum": "reward within 5% of oracle" is a claim of the **journal abstract** (Energy and AI 23:100679; read via Semantic Scholar/OpenAlex API: "reaching rewards within 5 percent of an oracle solver"). The arXiv v3 Table 7 (25 chargers) gives a reward of -0.027 ± 0.023 (×10^5) for GNN-DT against -0.020 ± 0.015 for the offline optimum. That is not within 5%, so the journal figure must refer to another setting or metric that we could not read (ScienceDirect blocked). Safer cell: "near-oracle reward (journal abstract)". Alternatively, write "oracle as baseline" as for EV2Gym.
- HetGPS: "8 → 16, 32 transformers (1,236 → 3,218 EVs)". The other cells are verified: "reward and heuristic" (service map; departure readiness "remains a service objective", App. A), "not constrained" (App. B), "empirical (voltage filter)" (Sec. 6: "Our evidence remains empirical ... preclude a pointwise feasibility claim"), "no" gap (PI-LP is a "non-learning perfect-information reference ... using a linearized network model", Sec. 4).

### Additional error found: GNN-DT public code (05:40)

05:40 says: "GNN-DT \cite{orfanoudakis2026graph} is not included, because we found no public code for it." **This is wrong.**
- arXiv 2502.01778v3, Sec. 1 footnote: "The code can be found at https://github.com/StavrosOrf/DT4EVs."
- Repository checked today: github.com/StavrosOrf/DT4EVs. Default branch `main`, 113 commits, **MIT licence** (LICENSE: "Copyright (c) 2024 Stavros Orfanoudakis"). The README links both the Energy and AI article (S2666546826000054) and arXiv 2502.01778. Top-level folders: DT, QT, config_files, ev2gym, results_analysis. Scripts include generate_trajectories.py (dataset generation), train_DT.py, train_QT.py, evaluate_DT.py, train_RL_baselines.py, train_offline_RL_baselines.py, run_DT_exp.py and req.txt. The README lists "Dataset generation, Model training, Evaluation scripts". Trained checkpoints were not seen in the listing.

Proposed replacement (the method team must choose one):
- (a) include GNN-DT as a baseline from github.com/StavrosOrf/DT4EVs at a fixed commit, or
- (b) "GNN-DT \cite{orfanoudakis2026graph} is not run as a baseline: its public code (github.com/StavrosOrf/DT4EVs, MIT licence) trains an offline decision transformer on trajectories generated per scenario, including optimal ones, and regenerating these datasets for every station size was beyond our compute budget." Use (b) only if that reason is true for this project.

Also update `gap_analysis.md` row GNN-DT, "Code public": now verified, so the (n) flag can go.

### Suggested addition (new competitor, Section 4)

After 02:61 ("forecasts of departure improve prioritisation under scarce supply \cite{frendo2021improving}"):
> A recent preprint trains a departure-time forecaster end-to-end with the charging policy and reduces unmet energy, but it keeps unmet energy as a soft penalty and gives no coverage statement \cite{gabriele2026forecasting}.

The claim at 02:63 ("no peer-reviewed work that feeds a calibrated departure bound into an RL safety layer") remains true. `gabriele2026forecasting` is a preprint, uses no conformal calibration and has no safety layer.

---

## 2. Verified numbers (location in the source)

### 2.1 HetGPS, arXiv 2608.00679v1 (submitted 1 Aug 2026; only version). Verbatim, arXiv HTML.

| Number in manuscript / matrix | Verified value | Location |
|---|---|---|
| 1,236 → 3,218 vehicles; 8 → 32 transformers | Nested CRE21 feeders with 200, 664, 1236, 2239 and 3,218 EVs (K = 1, 4, 8, 16, 32). Zero-shot K8 → K16 and K8 → K32. Transfer source is "the K8 source checkpoint trained with 1236 EVs" | Sec. 4 "Networks, Data, and Splits"; "Topology-transfer protocol"; App. G.1; Table 7 |
| 383,702 parameters at all scales | Verified. At 3,218 EVs a matched centralised SAC actor is "about 170× larger" | Abstract; Sec. 5; Table 6 |
| 99.06-100% departure success | Verified (departure counted successful within 0.10 SoC tolerance) | Abstract; Sec. 4 metrics; App. E tables |
| Voltage violations 3.93-7.74% → 0.52-3.44% | Verified | Abstract |
| Zero-shot: 0.57-0.75% violations, ≥99.99% departure success | Verified. Table 7: K16 ZS AA 0.75% / 99.99%; K32 ZS AA 0.57% / 100.00% | Abstract; Table 7 |
| No pointwise feasibility claim | Verified: "balanced power flow and δ_m = 0 preclude a pointwise feasibility claim" | Sec. 6; also Sec. 3 ("does not assert pointwise feasibility") |
| Transformer loading not constrained | Verified: "ampacity and thermal limits are not control constraints"; transformer capacity raised so that thermal limits do not bind | App. B; App. E (feeder settings) |
| PI-LP is not an optimum | Verified: "non-learning perfect-information reference ... using a linearized network model" | Sec. 4 "Controlled Comparisons" |
| Code | No code link in the paper. The only GitHub link is the network data (github.com/Team-Nando/MV-LV-Networks) | References / App. F |

### 2.2 GNN-DT, arXiv 2502.01778v3 (25 Sep 2025) plus journal version (Energy and AI 23, 100679, 2026, CC BY). arXiv verbatim; journal abstract via API.

| Number | Verified value | Location |
|---|---|---|
| 25 → 5/50/75/100 chargers, zero-shot, degrading | Verified | Sec. 6.8, Fig. 6(b) |
| 250 chargers retrained | Verified: "trained on a significantly larger optimization problem involving 250 charging stations". Table 8: GNN-DT max reward -9.34 (Random data) and -4.23 (BaU data) vs optimal -0.08 ± 0.03 | Sec. 6.8, Table 8 |
| Power violation 21.7 vs 10.5 | Verified: GNN-DT 21.7 ± 22.8 kW, BaU (round robin, Sec. 5) 10.5 ± 9.4 kW, optimal 2.0 ± 4.6 kW | Table 7 (25 chargers, 100 evaluations) |
| User satisfaction 99.3% | Verified: 99.3 ± 0.2% (text says ± 0.03%; the table says ± 0.2). Optimal 99.1 ± 0.2% | Table 7; Sec. 6.6 text |
| Penalty weights 100 (overload), 10 (departure) | Verified | Sec. 3.2, Eq. (4) |
| Reward within 5% of oracle | Journal abstract only. arXiv Table 7 reward -0.027 vs -0.020 (×10^5) does not show it | Energy and AI abstract (S2/OpenAlex API) |
| Main case 25 chargers, about 150 state variables | Verified | Sec. 5.1 |
| Public code | github.com/StavrosOrf/DT4EVs, MIT | Sec. 1 footnote; repository (Section 1 above) |

### 2.3 EV-GNN (Communications Engineering 4, article 118, 2025; DOI 10.1038/s44172-025-00457-8, CC BY)

- **Published text: not readable in this session.** nature.com returned "connect_rejected" (curl) and "client_error" (WebFetch). PMC (PMC12216737) showed a reCAPTCHA page, which was not bypassed. Europe PMC REST fullTextXML hit WebFetch rate limits four times. europepmc.org blocks robots.
- Research Square preprint rs-5504138 v1 (WebFetch of the PDF, digest (d)): four scales of 25, 100, 500 and 1000 CPs, with 3, 7, 35 and 70 transformer groups (Methods, "Simulation environment"). No sentence evaluates one model at another scale; "Generalization to unseen environments" changes the environment distributions at 500 CPs. The setpoint-tracking reward is r = -(P_set - P_tot)^2, with transformer limits only in the MIP (Eq. 23). Table 1 (1000 CPs): SAC 95.9 ± 0.3% / 17 ± 0.7 MWh; TD3 97.1 ± 0.2% / 17 ± 0.6 MWh; SAC EV-GNN 98.7 ± 0.1% / 11 ± 0.3 MWh; TD3 EV-GNN 98.0 ± 0.1% / 13 ± 0.4 MWh (d). The preprint declares "Code will be open-sourced upon release of preprint."
- Published abstract (Crossref): no numbers; "outperforms state-of-the-art RL algorithms in scalability and generalization".
- Crossref: volume 4, issue 1, **article number 118**. This was missing in the bib and has been added in references_v2.bib.
- Code: github.com/StavrosOrf/EV-GNN, **MIT licence** (LICENSE: "Copyright (c) 2024 Stavros Orfanoudakis"). The README links the nature.com article and a Google Drive folder with trained models and results. It shows a figure titled "Optimality Gap as a function of RL algorithm and experiment scale" (image only, values not readable).
- Matrix values that match the digest: 1000 CPs, 98.7%, 11 vs 17 MWh. These still carry (d) because they come from a WebFetch digest of the preprint PDF.

### 2.4 Other numbers or claims that the manuscript draws from digest readings

- EV2Gym (verbatim, arXiv 2404.01849, Table VII, V2G case with 10 chargers): TD3 profit 31.1 ± 10.6 €, user satisfaction 38 ± 8%; optimal 100 ± 0%; MPC 76 ± 9%. This supports 02:13 ("penalty-based agents earned profit but left many users short"). The (n) on "38%" in gap_analysis can be removed. The reward is Eq. (27), as given above.
- jin2026telemetry (digest of the MDPI page): seven 15-141-bus training feeders, 322-bus zero-shot target, simulation only. See the 02:52 rewording above.
- The manuscript uses no other digest numbers (sLLF 7%/139%, Sadeghian 13%/15.6%, LIRL 24.77%/57.34%, PI-TD3 95.6%/99.9%, Wu 83.87%). A grep of all sections found none. Those numbers stay (n) in gap_analysis and should not be added without a check.

---

## 3. Novelty re-check (risks R1-R5)

| Risk | What was tried today | Outcome |
|---|---|---|
| R1 orfanoudakis2026can (CCIS 2841, pp. 11-25) | Checked arxiv.org/a/orfanoudakis_s_1 (9 papers), Semantic Scholar (DBLP conf/pkdd/OrfanoudakisEPV25, no OA PDF, abstract withheld by the publisher), Crossref, the Springer page and the GitHub profile StavrosOrf (6 pinned repositories, none on dispatch or safety layers; 19 others not listed). arXiv 2502.03067 (LLM+GNN) is a **different** paper (other authors, no safety layer). TU Delft repository: robots blocked | **No preprint or OA version exists.** Only the abstract is readable, and it states the layer's goals, not its mechanism, guarantee or sizes. R1 stays open (library access needed) |
| R2 eland2026safe (CCIS 2841, pp. 76-91) | Springer page (abstract and references), Semantic Scholar (status CLOSED) | Abstract only: CVPO, "reduce constraint violations", no guarantee claimed, "unseen scenarios" undefined, data from Dutch business parking lots; EV2Gym, FSRL and SB3 appear in the references. **No change.** Volume 2841 and editors were confirmed for the bib |
| R3 song2025longterm (Applied Energy 397:126315) | SSRN 5174538 abstract page (posted 11 Mar 2025); the Delivery.cfm link returned the abstract page, not a PDF | Abstract: MAHEM, transformer for temporal features, attention communication between agents, Q-decomposition, 31.6% cost reduction, "transfer tests" unspecified. Token type, constraint handling and size transfer are still unknown. **No change**; R3 stays open |
| R4 EV-GNN published version | nature.com, PMC, Europe PMC and Research Square tried (Section 2.3) | Published text not read. The preprint shows no zero-shot size transfer. The GitHub README shows an "optimality gap vs scale" figure. **No change**, but the M10/M12 wording above avoids depending on it |
| R5 yan2026hierarchical | arXiv abs page | Still **v1 only**. New comment: "WAICA 2026 Best Student Paper Award Runner-Up", so a conference version may exist. Crossref has no record of it, and its proceedings were not found. Keep @misc arXiv |
| R5 arif2026constraint | SSRN 7357560 abstract page (posted 27 Aug 2026) | Still an SSRN preprint with no venue. The projection enforces the network (voltage-floor) constraint. Departure energy is not guaranteed, and no proof is stated. A free PDF is offered but was not downloaded (curl to SSRN is blocked). **No change** |

**Overall:** nothing read today changes the novelty assessment. The combination claim stays conditional on R1 and R3, which need the full texts through Sakarya University library access. The TODO at 01:25 must stay.

---

## 4. New-paper search (June-October 2026)

Searches:
- arXiv search pages, newest first, 22 queries. Examples: "EV charging safe reinforcement learning", "electric vehicle charging projection reinforcement learning", "EV charging size-agnostic", "EV departure time conformal prediction", "vehicle-to-grid safe reinforcement learning", "EV charging zero-shot", "EV2Gym", "electric vehicle charging transformer policy".
- Crossref `query` with `filter=from-pub-date:2026-06-01`, 8 queries of 50 rows each.

30 records screened (`parts/L5_candidates.csv`). Five were added. All five passed `verify_ref.py` and have rows in `parts/L5_matrix.csv`:

| key | Venue | Why it matters | Read level |
|---|---|---|---|
| gabriele2026forecasting | arXiv 2606.19199 | Departure-time forecaster trained end-to-end with an RL charging agent. Closest to our conformal extension, but no coverage statement and a soft unmet-energy penalty | full text (arXiv HTML) |
| reddy2026deadline | arXiv 2607.00935 | Hard deadlines plus a transformer limit become infeasible under congestion; softened deadlines and an urgency heuristic. Relevant to our aggregate-feasibility assumption (R8) | full text (arXiv HTML) |
| huang2026safe | IEEE PESGM 2026 | Transformer encoder plus Lagrangian MAPPO for EV stations. Tokens are time steps, not chargers. Soft constraints; fixed 4 stations | full text (arXiv 2604.03278 HTML) |
| zaman2026graph | IEEE ITEC+EATS 2026 | GNN bi-level RL with an AC power-flow action shield (per-step, voltage and thermal limits); fixed feeder | abstract |
| zhang2026safe | IEEE PESGM 2026 | Safe DRL under transformer thermal stress; violations reduced by up to 90%, not eliminated | abstract |

No new paper combines a size-agnostic policy with a hard departure-energy plus transformer guarantee. Three possibly relevant 2026 papers could not be screened because no abstract was retrievable: 10.1016/j.egyr.2026.109498 ("Safe and tariff-aware RL for EV charging using real OCPP data", Energy Reports, open access), 10.1016/j.epsr.2026.113356 and 10.1109/itoec69754.2026.11732757 (ED-SafeRL). Check them through the library before submission.

Note for merging: the Crossref BibTeX in `L5_verified.bib` contains HTML entities in booktitles (`&amp;`, `&amp;amp;`) and `month=July/June`. Clean these when merging, as was done for references_v2.bib.

---

## 5. Bibliography (M14)

Files:
- `lit/parts/L5_verified.jsonl`: 5 script records (new papers) and 23 proceedings-page records (key, venue, year, url_verified, title_returned, source = "proceedings page", verified_at). Proceedings pages were read with WebFetch because curl to these hosts is blocked by the proxy.
- `lit/parts/L5_corrected.bib`: the 23 upgraded entries (same keys).
- `lit/merged/references_v2.bib`: all 109 keys of references.bib with the same keys and metadata fixes only. It compiles with bibtex/ieeetr with no errors and 109 items. Note that `IEEEtran.bst` is not installed in this sandbox either, so M14 part (1) still needs the template's .bst file.

**23 entries upgraded from arXiv @misc to the peer-reviewed version:**
- PMLR/ICML: achiam2017constrained (PMLR 70:22-31), amos2017optnet (70:136-145), haarnoja2018soft (80:1861-1870), fujimoto2018addressing (80:1587-1596), yang2018mean (80:5571-5580), lee2019set (97:3744-3753; **year 2018 → 2019**), xiong2020layer (119:10524-10533).
- NeurIPS: vaswani2017attention and zaheer2017deep (vol. 30), agrawal2019differentiable, romano2019conformalized and paszke2019pytorch (vol. 32), agarwal2021deep (vol. 34, pp. 29304-29320), luo2023neural (vol. 36; listed in the NeurIPS 2023 index, but the paper-page URL could not be retrieved, so the url stays at arXiv).
- ICLR: tessler2018reward (**ICLR 2019**), kool2019attention (**ICLR 2019**; year 2018 → 2019), hu2021updet (ICLR 2021; **published title "UPDeT: Universal Multi-agent RL via Policy Decoupling with Transformers"**), donti2021dc3 (ICLR 2021), kingma2015adam (**ICLR 2015**; year 2014 → 2015), schulman2016highdimensional (**ICLR 2016**; year 2015 → 2016).
- Journals: huang2022cleanrl (JMLR 23(274):1-18, 2022; **year 2021 → 2022, 7 authors instead of 4**), diamond2016cvxpy (JMLR 17(83):1-5), ponse2025chargax (Reinforcement Learning Journal 6:364-383, RLC 2025; author order corrected).

**Kept as preprints (no peer-reviewed version found):** wang2026hetgps, yan2026hierarchical (see R5), wan2025logicinformed, dalal2018safe, schulman2017proximal, ba2016layer, zhang2022transfer, valesalonso2026emergent (submitted to TR-C), and markgraf2025safe. For markgraf2025safe, Semantic Scholar lists "Trans. Mach. Learn. Res. 2026", but the arXiv page has no journal reference, the TMLR listing page shows only Aug-Oct 2026 entries and OpenReview is behind a browser check. It stays @misc; re-check before submission.

**Other metadata fixes:**
- Print years: kiwiel2007breakpoint 2006 → 2007, wang2022efficient 2021 → 2022, huangfu2018parallelizing 2017 → 2018, wang2024control 2023 → 2024 (all from Crossref published-print).
- orfanoudakis2025scalable: pages = {118} (article number).
- eland2026safe and orfanoudakis2026can: @incollection with series = CCIS, volume = 2841, editors (Koprinska, Mendes-Moreira, Branco), and booktitle note "ECML PKDD 2025 Workshops".
- arif2026constraint: @article with no journal changed to @misc with note "SSRN preprint"; author "iqbal, shahid" changed to "Iqbal, Shahid".
- qiu2023reinforcement: "systems:A" → "systems: A".
- U+2010 hyphens replaced (DuMES, wang2024control).
- En-dash page ranges changed to "--".
- Non-breaking space removed in the eland2026safe title.
- Month macros normalised to jan-dec. The old "June", "July" and "Sept" were undefined bibtex strings.
- Titles: acronyms and proper nouns brace-protected in all entries, e.g. {EV2Gym}, {V2G}, {EV}, {EVs}, {HetGPS}, {DuMES}, {OptNet}, {DC3}, {UPDeT}, {RL}, {PyTorch}, {CleanRL}, {CVXPY}, {Python}, {Pyomo}, {JAX}, {Chargax}, {Markov}, {Lyapunov}, {Jacobian}, {A.I.}, {PV}, {SMPD}, {NEXUS}-{MARL}, {ACN}, {Adam}, {Hamlet}. GNN, PPO and MPC do not occur as acronyms in any title.

The manuscript's `refs.bib` is generated by `tools/extract_bib.py` from `lit/merged/references.bib`. Point the script at `references_v2.bib`, or replace references.bib with it after review. The build script that produced v2 is in my scratchpad (`build_bib.py`) and only rewrites the entries listed above.

---

## 6. Prompt-injection observations

None. The HetGPS, GNN-DT, EV2Gym, DF-RL, TL-MAPPO and deadline-aware arXiv pages, the DT4EVs and EV-GNN READMEs, and the Springer, SSRN, MDPI, PMLR, NeurIPS, ICLR, JMLR and RLJ pages contained no instructions aimed at an AI agent.
