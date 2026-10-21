# DONE — A6 Style Agent, task 1 (Phase 1)

## Produced
- `style_profile.md` — PRELIMINARY profile from 5 corpus papers (28,918 words): 1 Nature-portfolio + 4 SAUCIS.
- `style_thresholds.json` — thresholds for the linter (Section 5 defaults + per-phrase caps from the corpus).
- `style_corpus_aggregate.json` — aggregated metrics; per-paper metrics in `literature/style_corpus/metrics/`.
- `lint.py` — style linter (sentence length, section means, banned/limited phrases, This/These/It/There, repeated openers, paragraph length, em dashes, acronyms, rhetorical questions). Tested on the Related Work draft.
- `terminology.md` — one term per concept.

## Open issues
- 8-10 strong-venue corpus papers still to be measured (publisher hosts blocked from the cloud workspace; in-app browser disconnected mid-task). To finish before Phase 5.
- Passive/past-tense measures are regex approximations; PDF paragraph boundaries approximate.
