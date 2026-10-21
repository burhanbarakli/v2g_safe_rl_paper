# DONE: A5 task 1 (journal requirements) + A1 task 7 (style corpus)

Date: 2026-10-21

## Part A: SAUCIS requirements (finished)
Output: `/home/claude/work/journal/journal_requirements.md`. It has a requirements table with a source URL for every rule, a detailed generative-AI section (SAUCIS, DergiPark, TR Dizin, COPE), a list of 13 unclear or unverified items, and 2 draft AI-disclosure wordings.

Key rules:
- English only, and no Turkish abstract is required.
- Abstract ≤ 400 words, 3–5 keywords.
- No hard page limit. Guidance for experimental papers is ≤ 25 double-spaced pages and ≤ 7 figures + tables.
- IEEE numeric references.
- LaTeX template only (Overleaf "Sakarya University Journal of Computer and Information Sciences LaTeX Template", class `saucis`, IEEEtran; ZIP saucisLLatex_v_6.zip on Google Drive).
- Table captions go above tables, figure captions below figures, equations are numbered on the right. EPS figures are requested.
- Double-blind review with ≥ 2 referees. An anonymised PDF, a full PDF, the LaTeX source and the copyright form must be uploaded. ORCID is required for all authors.
- Similarity < 20 % excluding references (3 % per single source on the policy page). iThenticate is named in the template.
- CC BY-NC 4.0 and no fees of any kind.
- An Article Information Form is required (Author contributions, Acknowledgments, Conflict of Interest, Support, Ethical Approval, Availability, AI Statement, Plagiarism Statement).
- AI policy (11 Apr 2025): AI cannot be an author. AI generation of substantial sections (abstract, introduction, literature review, discussion) is "deemed inappropriate" unless it is treated only as a preliminary draft that the authors thoroughly revise. AI must not produce, report or interpret results. Disclosure must give tool names, version numbers and purpose, in Methods or Acknowledgements, plus the AI Statement item.
- DergiPark (announcement of 15 Oct 2026) will add AI-use and CRediT declarations at submission and require CC-BY for all journals. No start date has been announced.

## Part B: style corpus (BLOCKED for full text; selection and verification done)
- **15 papers selected and verified** with `verify_ref.py`. All 15 VERIFIED via Crossref. Outputs are `lit/parts/S_verified.jsonl` and `S_verified.bib`.
  - 10 from strong venues: Applied Energy ×2, Energy and AI ×2, Communications Engineering, Nature Communications ×2, IEEE TSG, IEEE TPWRS, and IEEE Access ×1.
  - 5 from SAUCIS (2023–2026).
- **0 full texts obtained, so no `<key>.txt` files were written.** Every PDF/HTML host was blocked by the proxy (403): nature.com, sciencedirect/api.elsevier.com, ieeexplore, dergipark.org.tr, saucis.sakarya.edu.tr, mdpi, springer, wiley, core, europepmc, zenodo, web.archive.org. WebFetch only returns summaries and refuses verbatim text, as tested on a CC BY-NC SAUCIS PDF and on arXiv. sciencedirect and saucis.sakarya.edu.tr are robots-disallowed for WebFetch. The OpenAlex daily budget was also exhausted (shared IP), so Crossref and Semantic Scholar were used for discovery.
- `style_corpus/corpus_index.csv` is written with `word_count = NA` and the OA `text_source_url` for each paper, so a human can download them.
- `scripts/build_style_corpus.py` is a helper. After the human saves the PDFs as `pdfs/style/<key>.pdf`, it runs pdftotext, cuts Introduction to Conclusion, strips headers and footers, adds `## ` headings, writes `<key>.txt` and fills in word_count. The heuristics were smoke-tested on synthetic text only.
- `style_corpus/style_profiles.md` is a stopgap: structure profiles of the 5 SAUCIS papers and 1 Nature Communications paper, taken from WebFetch summaries (headings, estimated lengths, counts, voice, back matter). It is labelled as summary-level, not corpus text.
- `lit/parts/S_downloads.tsv` logs the 7 download attempts, all failed (size 0).

Note: SAUCIS DOIs are registered literally with three dots, e.g. `10.35377/saucis...1209519`. Crossref resolves them as written, so cite them exactly like that.

## Open issues
1. The human needs to download the 15 PDFs (URLs in corpus_index.csv) and run the builder. Alternatively, an admin can allow nature.com and dergipark.org.tr for the sandbox. Note that the Elsevier and IEEE OA PDFs need a browser.
2. The human should download and check the current SAUCIS LaTeX template (Overleaf template updated about 18 days ago) and the copyright form, since neither could be fetched.
3. Re-check the DergiPark submission screens at submission time for the new AI/CRediT forms and a possible CC-BY licence change.
4. **Policy risk for this project:** under the SAUCIS AI policy, the AI-drafted abstract, introduction, related work and discussion must be thoroughly revised by the human authors, and the interpretation of results must be the authors' own. The disclosure must be accurate.
5. WebSearch is disabled for this org (403), so the journal's own website was not checked independently.

## Prompt-injection observations
None. No page, PDF or template that was fetched contained instructions aimed at AI agents (each WebFetch on the PDFs and announcement pages was explicitly asked to check). The only AI-related text found was the journal's AI policy and authors' AI statements, which are normal content.
