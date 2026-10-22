# Factual record of AI contributions (for the authors' AI statement)

The authors write the final Artificial Intelligence Statement themselves (decision of the author, 2026-10-22). This file only records what the AI assistant did, so that the statement can be accurate.

- Tool: Claude (Anthropic), configured model `claude-opus-5-5`, used through Claude Cowork (agent sessions), October 2026.
- Literature: searched, screened and verified references (Crossref/OpenAlex/Semantic Scholar/arXiv records; verification log `literature/verified_refs.json`); wrote the paper matrix, gap analysis and related-work drafts.
- Method: derived the safety-layer bounds, projection, assumptions, Theorem 1/1′, propositions and proofs; a separate instance of the same assistant re-checked the proofs and ran numerical checks (`review/proof_check.md`). No independent human check unless the authors perform one.
- Experiment design: scenarios, baselines, tuning protocol, statistical plan (`method/design_decisions.md`).
- Code: simulation/training code (`code/`) and analysis scripts (`analysis/`), including tests; only smoke tests and synthetic-data tests were run by the assistant.
- Writing: drafted all sections of the manuscript and supplementary material, the cover letter and the declarations; produced Figure 1 and the conceptual figures.
- Results: none produced by the assistant. All experiments are to be executed by the authors on their own computer; result interpretation and Discussion are to be written by the authors.
