#!/usr/bin/env python3
"""Style metrics for the style corpus (A6) -- Python port of scripts/style_stats_dom.js.

The section classifier, sentence splitter, word regex, phrase lists and passive/past regexes are
identical to the browser version (style_stats_dom.js / style_stats.js) and to style/lint.py, so that
metrics computed here are comparable with the first five corpus papers measured in the browser.

API
  classify(heading)               -> section type or None (skipped section)
  stats_from_pairs(pairs)         pairs = [(heading, paragraph_text), ...]; heading is the raw
                                  top-level heading (classified here). Returns the metrics dict.
  stats_from_paras(paras)         paras = [(section_type, paragraph_text), ...] (already classified)
  paras_from_arxiv_html(html)     LaTeXML (arxiv.org/html/<id>) -> (paras, heads)
  paras_from_pdf_text(text)       pdftotext output with numbered headings (SAUCIS) -> (paras, heads)

CLI
  python3 style_stats.py --html FILE  [--url URL --key KEY --source TEXT --out OUT.json] [--debug]
  python3 style_stats.py --pdftext FILE [...]
"""
import argparse, json, re, sys

SECT = [("intro", re.compile(r"introduc|background|motivation", re.I)),
        ("related", re.compile(r"related|literature|prior work|state of the art", re.I)),
        ("method", re.compile(r"method|model|formulat|framework|approach|proposed|architecture|algorithm|design|system|problem", re.I)),
        ("setup", re.compile(r"setup|set-up|experiment|case stud|simulation|implementation|data", re.I)),
        ("results", re.compile(r"result|evaluation|performance|analysis|comparison", re.I)),
        ("discussion", re.compile(r"discussion|limitation|implication", re.I)),
        ("conclusion", re.compile(r"conclu|summary|future", re.I))]
SKIP = re.compile(r"reference|acknowledg|funding|author contribution|competing|conflict|data availability|code availability|"
                  r"ethics|additional information|supplementary|rights|about this|appendix|nomenclature|abbreviation|"
                  r"declaration|credit|article information|bibliograph", re.I)

ABBR = re.compile(r"\b(e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Ref|Refs|Sec|vs|approx|resp|cf|No|Tab|al)\.$", re.A)
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
TRANS = ["however", "therefore", "thus", "hence", "moreover", "furthermore", "additionally", "in addition", "in contrast",
         "consequently", "similarly", "meanwhile", "specifically", "in particular", "for example", "for instance", "finally",
         "first", "second", "nevertheless", "instead", "accordingly", "notably", "overall", "as a result", "on the other hand"]
RESTR = ["delve", "crucial", "pivotal", "paramount", "landscape", "realm", "intricate", "multifaceted", "seamless", "cutting-edge",
         "unlock", "harness", "foster", "underscore", "showcase", "holistic", "synergy", "plethora", "myriad", "meticulous",
         "noteworthy", "groundbreaking", "notably", "it is worth noting", "it is important to note", "paves the way",
         "sheds light", "furthermore", "moreover", "additionally", "significantly", "robust", "leverage", "comprehensive",
         "novel", "effectively", "efficient", "efficiently", "enhance", "facilitate", "utilize"]
PASS = re.compile(r"\b(is|are|was|were|be|been|being)\s+(\w+ly\s+)?(\w+ed|\w+en|built|made|shown|known|set|put|kept|held|run|"
                  r"done|given|taken|found|seen|chosen|used|sent)\b", re.I | re.A)
PASTV = re.compile(r"\b(was|were|achieved|obtained|observed|reduced|increased|outperformed|reached|showed|resulted|trained|"
                   r"tested|evaluated|compared|performed|yielded|decreased)\b", re.I | re.A)
WE = re.compile(r"\b(we|our|us)\b", re.I | re.A)
FIRST = re.compile(r"^[^A-Za-z]*([A-Za-z]+)")
DEMO = re.compile(r"^(This|These|It|There)$")


def ws(t):
    return re.sub(r"\s+", " ", t).strip()


# Deviation from the browser version (documented): a top-level heading containing "result" is
# classified as results even if it also matches an earlier pattern ("Experimental Results",
# "Simulation Results"); the JS order sent such headings to setup. The five browser-measured papers
# have no such headings in their section lists, so their labels are unaffected.
RESULT_FIRST = True


def classify(h):
    if SKIP.search(h):
        return None
    if RESULT_FIRST and re.search(r"result", h, re.I):
        return "results"
    for k, r in SECT:
        if r.search(h):
            return k
    return "other"


def split_sentences(p):
    out = []
    for s in SPLIT.split(p):
        if out and ABBR.search(out[-1]):
            out[-1] += " " + s
        else:
            out.append(s)
    return [s for s in out if len(s.split(" ")) >= 3]


def wc(s):
    return len(WORD.findall(s))


def cnt(txt, w):
    pat = r"\b" + w.replace("-", r"\-").replace(" ", r"\s+") + r"\w*"
    return len(re.findall(pat, txt, re.I | re.A))


def fix(x, d):  # JS +x.toFixed(d)
    v = float(f"{x:.{d}f}")
    return int(v) if v == int(v) else v


def stat(a):
    if not a:
        return None
    s = sorted(a)
    return {"n": len(a), "mean": fix(sum(a) / len(a), 1), "median": s[len(s) // 2], "p90": s[int(len(s) * 0.9)], "max": s[-1]}


def stats_from_paras(paras):
    by, all_txt = {}, ""
    for sec, p in paras:
        S = by.setdefault(sec, dict(paras=[], sents=[], words=0, we=0, passive=0, past=0, demo=0, consec=0))
        ss = split_sentences(p)
        S["paras"].append(wc(p))
        all_txt += " " + p
        prev = None
        for s in ss:
            n = wc(s)
            S["sents"].append(n); S["words"] += n
            if WE.search(s): S["we"] += 1
            if PASS.search(s): S["passive"] += 1
            if PASTV.search(s): S["past"] += 1
            m = FIRST.match(s); first = m.group(1) if m else ""
            if DEMO.match(first): S["demo"] += 1
            if prev and first and prev.lower() == first.lower(): S["consec"] += 1
            prev = first
    out = {"sections": {}, "total_words": 0}
    for k, S in by.items():
        out["total_words"] += S["words"]
        d = max(1, len(S["sents"]))
        out["sections"][k] = {"words": S["words"], "sentence_len": stat(S["sents"]), "para_len": stat(S["paras"]),
                              "frac_over35": fix(sum(1 for x in S["sents"] if x > 35) / d, 3),
                              "frac_we": fix(S["we"] / d, 3), "frac_passive": fix(S["passive"] / d, 3),
                              "frac_pastverb": fix(S["past"] / d, 3), "frac_demo_openers": fix(S["demo"] / d, 3),
                              "consec_same_opener": S["consec"]}
    W = max(1, out["total_words"])
    out["transitions_per1k"] = {t: fix(1000 * cnt(all_txt, t) / W, 2) for t in TRANS if cnt(all_txt, t)}
    out["restricted_per1k"] = {t: fix(1000 * cnt(all_txt, t) / W, 2) for t in RESTR if cnt(all_txt, t)}
    out["em_dashes_per1k"] = fix(1000 * all_txt.count("—") / W, 2)
    out["percent_with_space"] = len(re.findall(r"\d\s%", all_txt))
    out["percent_no_space"] = len(re.findall(r"\d%", all_txt))
    out["pm_usage"] = all_txt.count("±")
    return out


def stats_from_pairs(pairs):
    """pairs: [(top-level heading, paragraph)]; paragraphs under skipped headings and paragraphs
    shorter than 15 space-separated tokens are dropped (as in the browser version)."""
    paras = []
    for h, p in pairs:
        c = classify(h)
        p = ws(p)
        if c is None or len(p.split(" ")) < 15:
            continue
        paras.append((c, p))
    return stats_from_paras(paras)


# ---------------------------------------------------------------- arXiv LaTeXML HTML
REMOVE_SEL = ("math, .MathJax, mjx-container, .ltx_Math, .ltx_equation, .ltx_equationgroup, script, style, figure, "
              "figcaption, table, .c-article-references, .ltx_bibliography, .html-fig_show, .html-table_show, .ltx_note")
EXCL_ANCESTOR = ("figure, figcaption, table, nav, header, footer, aside, .ltx_abstract, .abstract, .ltx_bibliography, "
                 ".ltx_authors, .ltx_keywords, .ltx_classification, .ltx_acknowledgements, .ltx_page_footer")


def paras_from_arxiv_html(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    root = soup.find("article") or soup.find("main") or soup.body
    for e in root.select(REMOVE_SEL):
        if e.parent is not None:
            e.replace_with(" X ")
    # LaTeXML keeps acknowledgements in a div with its own h2/h6; drop it entirely
    for e in root.select(".ltx_acknowledgements, .ltx_abstract, .ltx_authors"):
        e.decompose()
    # LaTeXML heading levels depend on the document class (h2 in IEEEtran, h3 under a chapter level in
    # elsarticle/llncs builds), so the role is taken from the LaTeXML title class, not from the tag.
    TOP = {"ltx_title_section", "ltx_title_appendix", "ltx_title_bibliography", "ltx_title_chapter"}
    SUB = {"ltx_title_subsection"}

    def role(n):
        c = set(n.get("class") or [])
        if c & TOP: return "h2"
        if c & SUB: return "h3"
        if n.name == "p": return "p"
        return None

    cur, paras, heads = "other", [], []
    for n in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        r = role(n)
        if r is None:
            continue
        if r == "h2":
            h = ws(n.get_text()); heads.append(h); cur = classify(h); continue
        if r == "h3":
            if cur is not None and SKIP.search(n.get_text()): cur = None
            continue
        if cur is None:
            continue
        if any(True for _ in n.find_parents(lambda t: t.name in ("figure", "figcaption", "table", "nav", "header", "footer", "aside")
                                             or bool(set(t.get("class") or []) & {"ltx_abstract", "abstract", "ltx_bibliography", "ltx_authors",
                                                                                    "ltx_keywords", "ltx_classification", "ltx_page_footer"}))):
            continue
        t = ws(n.get_text())
        if len(t.split(" ")) < 15:
            continue
        paras.append((cur, t))
    return paras, heads


# ---------------------------------------------------------------- PDF text (pdftotext, single column)
HEAD_RE = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][A-Za-z ,&\-:/()]{2,80})\s*$")


def paras_from_pdf_text(text):
    """Top-level numbered headings ('1. Introduction'); paragraphs = blank-line separated blocks.
    Lines that look like page furniture (short, mostly non-letters) are dropped. Approximate."""
    cur, paras, heads, buf = "other", [], [], []
    started = False

    def flush():
        if buf and cur is not None and started:
            t = ws(" ".join(buf)).replace("- ", "-")
            if len(t.split(" ")) >= 15:
                paras.append((cur, t))
        buf.clear()

    for line in text.splitlines():
        m = HEAD_RE.match(line)
        if m and len(m.group(2).split()) <= 8:
            flush(); h = ws(m.group(0)); heads.append(h); cur = classify(h); started = True; continue
        if re.match(r"^\s*(References|REFERENCES|Article Information Form)\s*$", line):
            flush(); cur = None; heads.append(line.strip()); continue
        if not line.strip():
            flush(); continue
        buf.append(line.strip())
    flush()
    return paras, heads


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--html"); g.add_argument("--pdftext")
    ap.add_argument("--url", default=""); ap.add_argument("--key", default=""); ap.add_argument("--source", default="")
    ap.add_argument("--out"); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    if a.html:
        paras, heads = paras_from_arxiv_html(open(a.html, encoding="utf-8").read())
    else:
        paras, heads = paras_from_pdf_text(open(a.pdftext, encoding="utf-8").read())
    res = stats_from_paras(paras)
    res.update(url=a.url, heads=heads[:20], source=a.source, key=a.key)
    if a.debug:
        import random
        random.seed(1)
        print("HEADS:", heads, file=sys.stderr)
        for c in sorted({c for c, _ in paras}):
            ps = [p for cc, p in paras if cc == c]
            print(f"--- {c}: {len(ps)} paras", file=sys.stderr)
            for s in random.sample(split_sentences(ps[0]), min(2, len(split_sentences(ps[0])))):
                print("   >", s[:220], file=sys.stderr)
    js = json.dumps(res)
    if a.out:
        open(a.out, "w").write(js)
    else:
        print(js)


if __name__ == "__main__":
    main()
