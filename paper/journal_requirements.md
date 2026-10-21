# SAUCIS: submission requirements

Journal: Sakarya University Journal of Computer and Information Sciences (SAUCIS), e-ISSN 2636-8129, published by Sakarya University (founded 2018), hosted on DergiPark.
Compiled by Writing Agent A5 (task 1). All pages accessed **2026-10-21** with the WebFetch tool. WebFetch returns a model summary, not the raw page. Quotes are short excerpts that the tool returned, and paraphrases are marked as such. WebSearch was disabled for this organisation (HTTP 403), so every page was reached from links on the journal's DergiPark pages. curl was blocked for dergipark.org.tr, saucis.sakarya.edu.tr, drive.google.com and overleaf.com, so **no template file was downloaded**.

Abbreviated source URLs (used in the table):
- [HOME] https://dergipark.org.tr/en/pub/saucis
- [WR] Author Guidelines / Writing rules: https://dergipark.org.tr/en/pub/saucis/writing-rules (Turkish interface https://dergipark.org.tr/tr/pub/saucis/writing-rules gave the same content)
- [POL] Ethical Principles and Publication Policy: https://dergipark.org.tr/en/pub/saucis/policy
- [AI] Artificial Intelligence (AI) Policy (page last updated 11 April 2025): https://dergipark.org.tr/en/pub/saucis/page/18416
- [PR] Peer-Review Process: https://dergipark.org.tr/en/pub/saucis/page/14831
- [PLAG] Plagiarism Policy: https://dergipark.org.tr/en/pub/saucis/page/14832
- [COPY] Copyright Notice: https://dergipark.org.tr/en/pub/saucis/page/14176
- [PRICE] Price Policy: https://dergipark.org.tr/en/pub/saucis/price-policy
- [ABOUT] https://dergipark.org.tr/en/pub/saucis/about-journal
- [SCOPE] https://dergipark.org.tr/en/pub/saucis/aim-and-scope
- [OL] Overleaf template page: https://www.overleaf.com/latex/templates/sakarya-university-journal-of-computer-and-information-sciences-latex-template/dvrwhwmzkgmp (preview PDF: same URL + `.pdf`)
- [GD] Google Drive template archive: https://drive.google.com/file/d/1X0vc1f7kZehvKSVPLSw35jcryzY-wfGD/view?usp=drive_link
- [CF] Copyright Agreement form: https://dergipark.org.tr/tr/download/journal-file/12296
- [DP40] DergiPark announcement no. 40 (15 Oct 2026): https://dergipark.org.tr/en/pub/announcement/40
- [TRD] TR Dizin journal criteria: https://trdizin.gov.tr/kriterler/
- [COPE] COPE position "Authorship and AI tools": https://publicationethics.org/guidance/cope-position/authorship-and-ai-tools
- [EX] Recent SAUCIS articles checked to see how the rules are applied in practice (PDFs read through WebFetch): https://dergipark.org.tr/en/download/article-file/5030792 (Vol. 9(2), 2026), https://dergipark.org.tr/en/download/article-file/5216396 (Vol. 9(3), 2026), https://dergipark.org.tr/en/download/article-file/5130772 (Vol. 8(4), 2025), https://dergipark.org.tr/en/download/article-file/4013362 (Vol. 7(3), 2024)

## 1. Requirements table

| Requirement | Rule | Source URL |
|---|---|---|
| Language | English only: "Any other languages are not accepted for publication." | [WR], [ABOUT] |
| Turkish abstract | Not required. The guidelines ask only for an English abstract and English keywords. TR Dizin only *recommends* a second-language abstract (criterion 3.5-5). | [WR], [TRD] |
| Article types | Original research articles ("high-quality, refereed articles"). "A smaller number of timely tutorial and surveying contributions will be published occasionally." The template header label is "RESEARCH ARTICLE". | [SCOPE], [OL] |
| Scope | Topics listed include AI, deep learning, IoT, computer networks, cybersecurity, "Other fields on computer sciences". Energy, smart grids and EVs are **not named**. The journal has published EV energy-demand forecasting, hydro-power LSTM and DQN-for-IoT papers (see the style corpus), so in practice the topic is in scope. | [SCOPE], [EX] |
| Length | "No page limit" as a rule. Guidance: experimental papers ≤ **25 double-spaced pages** and ≤ **7 figures + tables**. Theoretical papers ≤ 30 pages and ≤ 10 figures + tables. | [WR] |
| Abstract | ≤ **400 words** ("should not exceed to 400 words"). Template: self-contained, citation-free, no references to figures/tables/sections, third person, covering motivation, methodology, contribution and performance results. | [WR], [OL] |
| Keywords | **3–5** English keywords, under a bold "Keywords" title, separated by commas (max 5). Template asks for alphabetical order. | [WR], [OL] |
| Title | No explicit title rules (length, capitalisation) on the guidelines page. | [WR] |
| Author information | Full names, e-mails, affiliations and **ORCID iDs of all authors** are required. Corresponding author marked. Template first page has an "Article History" block (Received/Revised/Accepted/Published Online), filled in by the journal. | [WR], [OL] |
| Paragraphs | Paragraphs should not start with an indentation. | [WR] |
| Section structure | Not prescribed by the guidelines. Template body: numbered sections ("Section Title", subsections, "Algorithms, Codes, and Pseudocodes", "Conclusion"), then **Article Information Form**, then References. Recent papers use 1 Introduction / 2 Related Work(s) / 3 Method / 4 Experiments or Results / 5 Discussion / 6 Conclusion(s). | [WR], [OL], [EX] |
| Article Information Form (back matter) | Template items: **Author's Contributions**, **Acknowledgments**, **Conflict of Interest**, **Support Section** (funding; may be omitted if none), **Ethical Approval** (required for all articles: research-and-publication ethics declaration, plus committee name/date/number if applicable), **Availability** ("Not applicable / or link"), **Artificial Intelligence Statement**, **Plagiarism Statement** ("This article has been scanned by iThenticate™."). | [OL] preview PDF |
| Reference style | **IEEE**. The section is titled "References". In-text citations are numbered in square brackets, e.g. [1], [3-5]. Template uses `IEEEtran.bst`. | [WR], [OL] |
| Reference examples (as given by the journal; the longer ones were truncated by the fetch tool) | Book: "B. Klaus and P. Horn, Robot Vision. Cambridge, MA, USA: MIT Press, 1986." Book chapter: "L. Stein, "Random patterns," in Computers and You, J. S. Brake, Ed. New York, NY, USA: Wiley, 1994, pp. 55–70." Dataset: "S. Ansolabehere, M. Palmer, and A. Lee. "Precinct-level election data. V1." January 20, 2014." Journal article (start only): "M. Ito et al., "Application of amorphous oxide TFT to electrophoretic display," J. Non-Cryst. Solids, vol. 354 …". Conference (start only): "G. Veruggio, "The EURON roboethics roadmap," in Proc. Humanoids '06: 6th IEEE-RAS Int. Conf. Humanoid Robots …". The guidelines give no examples for web pages, theses or arXiv. | [WR] |
| Equations | Centred, with the number in parentheses on the right. Refer to them in full in the text, e.g. "As seen in Equation 1". Template: number only the equations that are referred to later. | [WR], [OL] |
| Tables | Centred. Number and caption **above** the table. Referred to as "As seen in Table 1". | [WR], [OL] |
| Figures | Centred. Number and caption **below** the figure. Referred to as "As seen in Figure 1". Template asks for figures as separate **EPS** files. **No resolution (dpi) rule found.** | [WR], [OL] |
| Template | LaTeX only. (1) Overleaf template "Sakarya University Journal of Computer and Information Sciences LaTeX Template" (author Ahmet Zengin, CC BY 4.0, class `\documentclass[final, 9pt, onecolumn]{saucis}`, main file `saucis.tex`, pdfLaTeX, bibliography style IEEEtran). (2) ZIP archive linked as "saucisLatex.zip" on the guidelines page. The Google Drive file name is **saucisLLatex_v_6.zip**. **No Word template was found.** | [WR], [OL], [GD] |
| Required uploads | (1) PDF with author information, ORCIDs and corresponding-author details. (2) **Anonymised PDF** with no author information (double-blind). (3) **Copyright Agreement and Author Acceptance Form**. (4) **LaTeX source** file. | [WR] |
| Recommended uploads | Plagiarism (similarity) report. Approved ethics committee permit, if applicable. | [WR] |
| Submission channel | DergiPark system only, no e-mail: https://dergipark.org.tr/en/journal/2493/submission/step/manuscript/new (login required, not accessed). | [WR], [HOME] |
| Ethics committee approval | Required for clinical/experimental studies on humans, and for studies that collect participant data by survey, interview, focus group, observation or experiment. Board name, date and number must appear in the Article Information Form and in the Methods section. Not applicable to a pure simulation study like ours. The Ethical Approval item is still filled in, as in recent papers. | [WR], [POL], [TRD], [EX] |
| Conflict of interest | Must be disclosed (financial or other substantive conflicts). Template wording: "The authors declare that there is no conflict of interest regarding the publication of this paper." | [POL], [OL] |
| Author contributions | Required. For multi-author papers, each author's contribution is stated (template: "contribution rate" per author). **CRediT is not required by SAUCIS today.** DergiPark has announced that its renewed submission system will add a CRediT declaration (see §3). | [POL], [OL], [DP40] |
| Data availability | "Availability" item in the Article Information Form ("Not applicable / or link"). The policy encourages public data sharing and recommends a data availability statement. | [POL], [OL] |
| Funding | "All sources of financial support for the project should be disclosed" (Support Section). | [POL], [OL] |
| Acknowledgements | Contributors who do not meet authorship criteria "should be acknowledged". | [POL], [OL] |
| Similarity threshold | Overall similarity **< 20 % excluding references**. The policy page also gives a **single-source maximum of 3 %**, which the plagiarism page does not mention. The software is not named on the policy pages. The template's plagiarism statement names **iThenticate**. | [PR], [PLAG], [POL], [OL] |
| Duplicate submission / reuse | No simultaneous submission. Reuse of the authors' own earlier work must be cited, with a statement of the new contribution. | [POL] |
| Licence / copyright | Authors keep copyright. Articles are published under **CC BY-NC 4.0**. The copyright agreement is accepted during upload, and the form [CF] is uploaded. A change of authorship after submission needs a letter signed by all authors and a new form. | [COPY], [POL], [WR] |
| Fees / APC | **None**: "No fee is charged from the author or institution under any name." There are no submission, review, page or colour charges. | [PRICE], [ABOUT] |
| ORCID | Required for all authors. | [WR], [TRD] |
| Cover letter | Not mentioned in the guidelines. | [WR] |
| Suggested reviewers | Not mentioned in the guidelines. | [WR] |
| Generative-AI policy | See §2. AI cannot be an author. Using AI to generate substantial sections is "deemed inappropriate". AI "must not be employed to produce, report, or interpret research findings". Disclosure is mandatory: name and version of each tool, how and why it was used, in "Methods" or "Acknowledgements". The template also has an "Artificial Intelligence Statement" item. | [AI], [OL] |
| Review type | **Double-blind** peer review, with "at least two referees". The manuscript **must be anonymised**: a separate anonymised PDF is uploaded. | [PR], [POL], [WR] |
| Review workflow | Pre-check plus similarity check (up to 7 days), then editor/associate editor check (up to 30 days), then ≥ 2 reviewers, then revision, then editorial board decision, then an English-language check (up to 15 days), then typesetting by the journal. | [PR] |
| Frequency / indexing | Quarterly (March, June, September, December). Indexed in Scopus, DOAJ and TR Dizin. EBSCO, Google Scholar and Crossref logos also appear. | [HOME], [ABOUT] |
| Ethics framework | Follows COPE guidelines and flowcharts (authorship disputes, corrections, retractions). | [POL], [WR] |

## 2. Generative-AI policy in detail

**SAUCIS AI Policy [AI]** (heading "Artificial Intelligence (AI) Policy", last updated 11 April 2025):
- **Authorship:** "Under no circumstances should generative artificial intelligence (AI) tools be designated as authors or co-authors". Authors remain fully responsible for content, accuracy and originality.
- **Content creation:** "The use of AI to generate substantial sections of a scholarly manuscript" (for example "the abstract, introduction, literature review, or discussion") "is deemed inappropriate". AI-generated content "should be treated solely as preliminary drafts or suggestions" that "must be thoroughly reviewed, revised, and refined by the author(s)". "AI may serve only as a supplementary aid in the writing process."
- **Results:** "AI tools must not be employed to produce, report, or interpret research findings."
- **References:** fabricated, unverifiable or non-existent references are strictly prohibited.
- **Allowed:** conceptual diagrams and explanatory visuals that reflect the authors' own understanding, help with data visualisation (graphs, charts, tables), and illustrative visuals.
- **Not addressed:** language editing and proofreading, translation, code writing, literature searching.
- **Disclosure:** "This disclosure should be included in the "Methods" or "Acknowledgements" section, as appropriate." It "must clearly specify the full names and version numbers of the AI tools used" and give "a detailed explanation of how and for what purposes these tools were employed". In addition, the template's Article Information Form has an **"Artificial Intelligence Statement"** item: "Please specify tools used in paper or state no artificial intelligence tools were used while writing this article." [OL]
- **Sanctions:** rejection during review. If found after publication: retraction or correction. For repeated or serious breaches: future submissions may be rejected.
- **Reviewers/editors:** must not upload manuscripts to generative-AI tools.
- The page does **not** cite COPE, WAME, ICMJE, DergiPark/ULAKBİM or TR Dizin AI policies.

**DergiPark / ULAKBİM [DP40]:** Announcement of 15 Oct 2026 (event on 22 Oct 2026). The renewed DergiPark article-process system will include an **AI-use declaration and a CRediT author-contribution declaration during submission**. It also states that "tüm dergilerin CC-BY lisansı belirlemesi zorunlu olacaktır" (all journals will be required to set a CC-BY licence). No go-live date is given. No separate DergiPark-wide written generative-AI policy was found from the DergiPark home page links.

**TR Dizin [TRD]:** The criteria page has **no AI-use criterion**. Relevant criteria: ethics approval stated in the article (board, date, number in Methods). An author contribution statement, a support/acknowledgement statement and a conflict-of-interest statement at the end of each article (criterion 3.2-13). ORCID on the first page.

**COPE [COPE]** (last reviewed 13 Feb 2023, version listed 20 Aug 2024): AI tools cannot be authors. Use of AI for writing, images or data collection/analysis must be disclosed in the Materials and Methods (or a similar section), naming the tool and how it was used. Authors are fully responsible. SAUCIS's AI policy is consistent with COPE but does not cite it.

**Practice in recent SAUCIS papers [EX]:** The 2025–2026 papers checked all include "Artificial Intelligence Statement: No artificial intelligence tools were used while writing this article." Earlier papers (2023–2024) have no AI statement. The back matter is titled "Article Information Form", with variants of the item names, for example "Conflict of Interest Notice" and "Availability of data and materials".

**Implication for this project (for the human to decide):** This manuscript is being drafted with substantial help from AI agents (literature search, drafting). Under the SAUCIS policy, AI-drafted abstract, introduction, literature review or discussion text is only acceptable as a *preliminary draft* that the authors thoroughly revise. AI must not report or interpret the results. The human authors must therefore rewrite or substantially revise these sections, write the interpretation of results themselves, and disclose AI use with tool name and version, purpose and extent.

## 3. Unclear / could not verify

1. **Template files not inspected directly.** curl to drive.google.com and overleaf.com was blocked (proxy 403). The ZIP name (saucisLLatex_v_6.zip, link text "saucisLatex.zip") and the template contents come from WebFetch summaries of the Google Drive viewer page, the Overleaf page and the preview PDF. The human should download [GD] or open [OL] in Overleaf. The Overleaf template showed "last updated 18 days ago", so it was recently revised. Check the current Article Information Form in it.
2. **Copyright form [CF]** is a binary file, so WebFetch could not read it. Its file name, format and fields are unknown.
3. **Word template:** none found. The required uploads list a LaTeX source file, so submission appears to be LaTeX-only. Confirm with the journal if a Word submission is wanted.
4. **Figure resolution / format:** no dpi rule found. The template asks for EPS figure files. Vector PDF or PNG acceptability is unverified.
5. **Single-source similarity limit:** 3 % on [POL], not mentioned on [PLAG]. Plan for ≤ 3 % per source anyway.
6. **Similarity software:** iThenticate is named only in the template's plagiarism statement, not on the policy pages.
7. **Turkish abstract:** not required by SAUCIS. This was confirmed only through the English-language pages. The Turkish-interface page returned the same English content.
8. **Scope fit:** energy, EVs and RL are not explicitly listed in the aims and scope. Fit is inferred from published papers. An optional pre-submission e-mail to saucis@sakarya.edu.tr is possible.
9. **Upcoming DergiPark changes [DP40]:** the mandatory CC-BY licence (SAUCIS currently uses CC BY-NC 4.0), the AI-use form and the CRediT form at submission have no announced effective date. The licence and declarations may change before or during our submission. Re-check the submission screens at submission time.
10. **Cover letter / suggested reviewers:** not mentioned. The submission form (login required, not accessed) may ask for them.
11. **Journal's own website:** WebSearch was unavailable. saucis.sakarya.edu.tr appears as a PDF host in Semantic Scholar records, but it was blocked for curl and robots-disallowed for WebFetch, so it was not checked separately. All rules above come from the DergiPark pages and the linked templates.
12. **Abstract keyword minimum:** "3–5" on [WR] versus "at least one" in the template. Follow 3–5.
13. **Verbatim reference examples:** the WebFetch tool truncates quotes at about 125 characters, so the full IEEE examples for journal and conference papers are not reproduced. Follow the IEEEtran style shipped with the template.

## 4. Draft AI-use disclosure options (the human decides; tool names and versions to be filled in accurately)

Placement: the "Artificial Intelligence Statement" item of the Article Information Form, **and** a short sentence in Methods or Acknowledgements (required by [AI]). Both options assume the authors have actually revised all AI-assisted text, and that results and interpretations are the authors' own.

**Option A (detailed, transparent):**
> Artificial Intelligence Statement: The authors used [Claude (Anthropic), model/version …] during the preparation of this work to (i) support literature searching and the screening of candidate references, every one of which was then checked by the authors against the publisher record via its DOI; (ii) assist with code development for the simulation experiments; and (iii) suggest preliminary wording for parts of the manuscript and improve language. All AI-suggested text was treated as a preliminary draft and was thoroughly reviewed, rewritten and verified by the authors. No AI tool was used to generate, report or interpret the research results. The authors take full responsibility for the content of this article.

**Option B (concise, language/code-support only; use only if the authors rewrite the AI-drafted sections themselves so that this is accurate):**
> Artificial Intelligence Statement: [Tool name, version] was used for language editing and for assistance with programming of the simulation code. The authors reviewed and edited all outputs and take full responsibility for the content of the publication. No AI tool generated scientific content, results or their interpretation, and no AI tool is listed as an author.
