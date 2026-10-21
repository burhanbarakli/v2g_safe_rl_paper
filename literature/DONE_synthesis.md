# DONE: Literature Agent A1, synthesis step

Date: 2026-10-21.

## Files produced (all in /home/claude/work/lit/merged/)
- `gap_analysis.md` has five parts:
  - (a) a comparison table of 13 works plus the planned work, with secondary near neighbours listed under the table;
  - (b) "What is already known", nine items;
  - (c) the novelty statement, 208 words;
  - (d) a risk register, R1-R12;
  - (e) the G1 recommendation, with a decision rule that depends on the result of R1.
- `related_work_draft.md` is the Related Work draft, organised by five themes. It has about 1,800 words counting each citation group as one word. There are 94 sentences, averaging 19.1 words, and the longest has 32 words. It cites 67 distinct keys, all present in `references.bib`. Two "AUTHOR NOTE" placeholders mark the text that depends on [orfanoudakis2026can] and [song2025longterm].
- `DONE_synthesis.md` is this file.

A scripted check of the draft found no banned words or phrases, no ", which" / ", while" / ", thereby" / "; thus," constructions, no em dashes, no rhetorical questions, no paragraph over 180 words, and at most one This/These/It/There opener per paragraph. Limited-use words stay within their caps.

## Deviations from the task
- The main table has 13 rows, not 10-12. The task required 11 named works and asked for 1-2 more, and I added two: [wan2025logicinformed] and [salesortiz2024efficient]. Salesortiz was chosen over [arif2026constraint] and [eland2026safe] because, at abstract level, it touches both C1 (a size-independent policy) and C2 (bounds that ensure individual requirements). Arif, Eland, [jin2026telemetry], [tabas2022computationally], [chen2021enforcing] and [li2020constrained] are summarised in a note under the table.
- [song2025longterm] has no table row. Its content is not reliably known (see below), so it appears only in the risk register and as a title-level citation in the draft.

## Claims I could not support and therefore omitted or softened
- **song2025longterm.** L1 recorded an SSRN abstract (a 31.6% cost reduction, attention communication, transfer tests). L3 and the orchestrator could not access any abstract. The draft therefore says only that transformer-based MARL was applied to collaborative charging across stations. It gives no numbers, no token design and no constraint claims. It is also excluded from the sentence saying that no size-agnostic controller enforces constraints by construction, which is restricted to "controllers we could read in full".
- **orfanoudakis2026can.** Only abstract-level wording is used: a safety layer meant to preserve charging requirements and enforce power limits. Nothing is said about its type, proof or size transfer.
- **EV-GNN.** I used the orchestrator's facts from the published page (penalty-based reward, guarantee statement, code). The preprint numbers (98.7% satisfaction, 11 MWh, 256k vs 2.8M parameters) were not used. The claim "a separate model is trained per scale" was left out of the draft, because the published version was not checked for size-transfer experiments. In the table it is attributed to the preprint only.
- **HPC-RL.** The "7-13% gap to MPC" and "100% demand satisfaction" figures appear only in the L0/L3 digests. L2 states that it did not reach the result tables. These numbers are absent from the draft and flagged (n) in the table.
- **DuMES, guan2024rule, zhang2025modelfree, eland2026safe, salesortiz2024efficient, arif2026constraint, li2020constrained.** These were described at abstract level only. No proofs or mechanisms were inferred.
- **li2020constrained.** Its abstract uses the word "guarantee" for the full-charge requirement. Because it is a CMDP method, the draft does not repeat that word.
- **zhang2022transfer.** The 509 to 50,900 EV transfer claim was not used in the draft because of the internal inconsistencies noted by L3. The draft only describes the architecture.
- **Code availability for GNN-DT and PI-TD3.** The GitHub URLs come from the L0 WebFetch digests and were not opened. They are flagged (n) in the table.
- **sLLF code, Sadeghianpourhamami code, HPC-RL code, LIRL code.** Not recorded by the part agents, so they are marked "?".
- **The equivalence of clip(a - lambda, l, u) with the Euclidean projection onto a box intersected with one sum constraint (KKT).** This is standard optimisation knowledge and our own statement, not a claim attributed to [kiwiel2007breakpoint] or [wang2022efficient]. Those two are cited only for solvers of that problem class. The Jacobian structure discussed in L2 (identity minus a rank-one term on free coordinates) is L2's own derivation. It was not used in the draft and must be derived in the method section.

## Numbers in the draft that come from WebFetch digests (verify against PDFs before submission)
- [sadeghianpourhamami2020definition]: 13% and 15.6% above the oracle; trained on 10 stations, tested up to tenfold arrivals.
- [orfanoudakis2026graph]: zero-shot from 25 to 100 chargers.
- [wang2026hetgps]: one parameter set from 1,236 to 3,218 EVs.
- [orfanoudakis2025scalable]: 1000 charge points. This number also appears in the orchestrator-read setting, so its risk is lower.
- [jin2026telemetry]: feeders of up to 141 buses transferred to a 322-bus feeder.
- [dyo2022behavior]: percentage errors above 400% (MAPE about 414%).
- [chen2021enforcing]: projection slower than the network (8.6 vs 4.5 ms), described qualitatively in the draft.
- [chen2022smoothed]: "little extra power" (7% in the paper), described qualitatively in the draft.
- [luo2023neural]: 1000 nodes (from the abstract).
- The table also carries (n) values for EV2Gym, GNN-DT, PI-TD3, HetGPS, Wu et al., HPC-RL, sLLF and LIRL.

## Open issues
1. **R1, highest priority.** Obtain the full text of [orfanoudakis2026can] (CCIS 2841, pp. 11-25) through Sakarya University library access. The G1 decision rule in `gap_analysis.md` (e) depends on it.
2. **R2/R3.** Obtain the full texts of [eland2026safe] and [song2025longterm].
3. **R4.** Check the published EV-GNN article and the EV-GNN GitHub configs for zero-shot size transfer.
4. **R5/R6.** Re-check the preprints [yan2026hierarchical] and [arif2026constraint] for new versions. Obtain [salesortiz2024efficient] to check whether its bounds are proven.
5. **R10, search recall.** Recall was limited by the OpenAlex budget, Semantic Scholar throttling and disabled WebSearch. Unread leads, with DOIs as recorded by the part agents and not verified here: 10.1016/j.epsr.2026.113356, 10.1016/j.apenergy.2026.128219, 10.1016/j.cma.2026.119446 (capped-simplex, box-plus-sum projection in RL), and 10.1038/s41598-025-22482-5. None of these is cited in the draft.
6. **Terminology.** "Transformer" means both the electrical transformer and the neural architecture. The draft uses "transformer limit" for the electrical sense and "transformer-encoder policy" / "transformer policy" for the network. The writing agent should keep this convention, or use "attention encoder" for the network.
7. **Bibliography metadata.** Some arXiv-derived bib years need attention. lee2019set and kool2019attention carry 2018 (arXiv v1) in `references.bib`. vaswani2017attention now carries 2017 in the merged bib; L3 had reported 2025 in its part bib. altman2021constrained is the 2021 Routledge edition. The writing agent should check the IEEE reference formatting.
8. **EV2Gym oracle.** The v2.0.0 profit-max oracle uses a soft departure penalty. The experiments must add a hard departure constraint and say so (R12).

## Prompt-injection observations
- None. All inputs were local files written by the earlier agents. No instructions aimed at an AI agent were found in them.
