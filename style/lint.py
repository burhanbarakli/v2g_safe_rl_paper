#!/usr/bin/env python3
"""Style lint for the manuscript (A6).

Usage:
    python3 style/lint.py paper/manuscript.md [--profile style/style_thresholds.json] [--out style/lint_reports/report.md]

Input: Markdown (headings '#', '##', ...) or LaTeX (\\section{...}). LaTeX math, citations, refs,
figure/table environments and comments are stripped before analysis.

Checks per section (thresholds from Section 5 of the orchestrator prompt; can be tightened by the
style profile JSON):
  - sentences longer than MAX_SENT words (default 35)                      [flag]
  - section mean sentence length outside [MEAN_LO, MEAN_HI] (16-26)        [flag]
  - restricted phrases: banned list (max 1 per manuscript) and limited list (max N per 1,000 words)
  - sentences starting with This/These/It/There (>1 per paragraph -> flag; "This" + verb -> flag)
  - repeated sentence openings in consecutive sentences                     [flag]
  - paragraphs longer than MAX_PARA words (default 180)                     [flag]
  - em dashes                                                               [flag]
  - undefined acronyms and acronyms defined more than once
  - vague intensifiers, "not only ... but also" (>1), rhetorical questions
Exit code: 0 if no blocking flags, 1 otherwise.
"""
import argparse, json, re, sys, collections

DEFAULTS = {
    "MAX_SENT": 35, "MEAN_LO": 16.0, "MEAN_HI": 26.0, "MAX_PARA": 180,
    "LIMITED_PER_1K": 2.0,
}
BANNED = ["delve", "crucial", "pivotal", "paramount", "landscape", "realm", "tapestry", "intricate",
          "multifaceted", "seamless", "seamlessly", "cutting-edge", "game-changer", "revolutionize",
          "revolutionise", "unlock", "harness", "foster", "underscore", "underscores", "showcase",
          "showcases", "holistic", "synergy", "plethora", "myriad", "meticulous", "meticulously",
          "noteworthy", "commendable", "groundbreaking", "in today's world", "in the era of",
          "ever-evolving", "rapidly evolving", "plays a vital role", "plays a crucial role",
          "plays a pivotal role", "paves the way", "sheds light on", "a testament to",
          "it is worth noting that", "it is important to note that", "notably"]
LIMITED = ["furthermore", "moreover", "additionally", "significantly", "robust", "leverage",
           "leverages", "leveraging", "comprehensive", "novel", "effectively", "efficient",
           "efficiently", "enhance", "enhances", "enhanced", "facilitate", "facilitates", "utilize",
           "utilizes", "utilized", "utilise"]
INTENSIFIERS = ["very", "highly", "extremely", "greatly", "substantially"]
ABBR = re.compile(r"\b(e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Ref|Refs|Sec|vs|approx|resp|cf|No|Tab|al|Thm|Def|Prop|Alg)\.$")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")
DEMO = {"This", "These", "It", "There"}
VERBS_AFTER_THIS = re.compile(r"^This\s+(is|was|shows|showed|means|suggests|indicates|demonstrates|leads|allows|makes|gives|implies|results|can|will|would|may|has|had|confirms|highlights|explains|ensures|helps)\b")
COMMON_NONACRONYMS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "A", "AND", "OR",
                      "NOTE", "AUTHOR", "TODO", "PDF", "PNG", "URL", "DOI", "ID", "CO", "USA", "UK", "EU", "MW", "MWh", "kW",
                      "kWh", "GHz", "CPU", "GPU", "RAM"}


def strip_latex(t):
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\begin\{(equation|align|figure|table|algorithm|eqnarray|gather|multline)\*?\}.*?\\end\{\1\*?\}", " X ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", " X ", t, flags=re.S)
    t = re.sub(r"\\\[.*?\\\]", " X ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " X ", t)
    t = re.sub(r"\\(cite|ref|eqref|label|autoref|cref)\{[^}]*\}", "[0]", t)
    t = re.sub(r"\\(section|subsection|subsubsection)\*?\{([^}]*)\}", lambda m: "\n\n" + ("#" * (1 + ["section", "subsection", "subsubsection"].index(m.group(1)))) + " " + m.group(2) + "\n\n", t)
    t = re.sub(r"\\(textbf|textit|emph|text|mathrm)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    return t


def strip_md(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", " X ", t, flags=re.S)
    t = re.sub(r"\$[^$\n]*\$", " X ", t)
    t = re.sub(r"^\s*\|.*\|\s*$", " ", t, flags=re.M)  # tables
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[[a-z][a-z0-9_:\-]*(,\s*[a-z][a-z0-9_:\-]*)*\]", "[0]", t)  # [key] citations
    t = re.sub(r"[*_`]", "", t)
    return t


def sections(text):
    secs, cur, buf = [], "Front matter", []
    for line in text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            secs.append((cur, "\n".join(buf)))
            cur, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    secs.append((cur, "\n".join(buf)))
    return [(h, b) for h, b in secs if b.strip()]


def paragraphs(body):
    out = []
    for p in re.split(r"\n\s*\n", body):
        p = re.sub(r"\s+", " ", p).strip()
        if not p or p.startswith(("- ", "* ", "1. ")) and len(WORD.findall(p)) < 25:
            continue
        if len(WORD.findall(p)) >= 8:
            out.append(p)
    return out


def sentences(p):
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", p)
    out = []
    for s in raw:
        if out and ABBR.search(out[-1]):
            out[-1] += " " + s
        else:
            out.append(s)
    return [s for s in out if len(WORD.findall(s)) >= 3]


def count_phrase(text, ph):
    return len(re.findall(r"(?<![A-Za-z])" + re.escape(ph).replace(r"\ ", r"\s+") + r"(?![A-Za-z])", text, flags=re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    th = dict(DEFAULTS)
    if a.profile:
        th.update(json.load(open(a.profile)))
    raw = open(a.path, encoding="utf-8").read()
    text = strip_latex(raw) if a.path.endswith(".tex") else strip_md(raw)
    rep, blocking = [], 0
    all_words = len(WORD.findall(text))
    rep.append(f"# Style lint report: `{a.path}`\n\nTotal words analysed: {all_words}\n")

    # global phrase counts
    rep.append("## Restricted phrases (whole manuscript)\n")
    rows = []
    for ph in BANNED:
        c = count_phrase(text, ph)
        if c > 1:
            rows.append(f"| {ph} | {c} | banned (max 1) | FLAG |"); blocking += 1
        elif c == 1:
            rows.append(f"| {ph} | 1 | banned (max 1) | check precise use |")
    lim = th["LIMITED_PER_1K"]
    for ph in LIMITED:
        c = count_phrase(text, ph)
        per = 1000 * c / max(1, all_words)
        lim_ph = th.get("LIMITED_OVERRIDES", {}).get(ph.rstrip("sd"), th.get("LIMITED_OVERRIDES", {}).get(ph, lim))
        if per > lim_ph and c > 1:
            rows.append(f"| {ph} | {c} ({per:.2f}/1k) | limited ({lim_ph}/1k) | FLAG |"); blocking += 1
    for ph in INTENSIFIERS:
        c = count_phrase(text, ph)
        if c:
            rows.append(f"| {ph} | {c} | vague intensifier | check quantified |")
    c = len(re.findall(r"not only\b.*?\bbut also", text, flags=re.I | re.S))
    if c > 1:
        rows.append(f"| not only ... but also | {c} | max 1 | FLAG |"); blocking += 1
    em = text.count("\u2014") + len(re.findall(r"(?<!-)---(?!-)", raw))
    if em:
        rows.append(f"| em dash | {em} | none allowed | FLAG |"); blocking += 1
    rq = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip().endswith("?")]
    if rq:
        rows.append(f"| rhetorical question | {len(rq)} | none allowed | FLAG |"); blocking += 1
    rep.append("| Phrase | Count | Rule | Status |\n|---|---|---|---|\n" + ("\n".join(rows) if rows else "| (none) | | | OK |") + "\n")

    # acronyms
    defs = collections.defaultdict(list)
    for m in re.finditer(r"([A-Za-z][A-Za-z\- ]{3,80}?)\s*\(([A-Z][A-Za-z0-9\-]*[A-Z][A-Za-z0-9\-]*s?)\)", text):
        defs[m.group(2).rstrip("s") if m.group(2).endswith("s") and m.group(2)[:-1].isupper() else m.group(2)].append(m.start())
    uses = collections.defaultdict(list)
    for m in re.finditer(r"\b([A-Z][A-Z0-9\-]*[A-Z0-9])s?\b", text):
        uses[m.group(1)].append(m.start())
    undefined, multi, before = [], [], []
    for ac, pos in uses.items():
        if ac in COMMON_NONACRONYMS or len(ac) < 2:
            continue
        if ac not in defs:
            undefined.append(ac)
        elif min(pos) < min(defs[ac]) - 2:
            before.append(ac)
    for ac, pos in defs.items():
        if len(pos) > 2:  # abstract + main text allowed
            multi.append(f"{ac} ({len(pos)}x)")
    rep.append("## Acronyms\n")
    rep.append(f"- Used but never defined: {', '.join(sorted(undefined)) or 'none'}")
    rep.append(f"- Used before first definition: {', '.join(sorted(before)) or 'none'}")
    rep.append(f"- Defined more than twice (abstract + main text allowed): {', '.join(multi) or 'none'}\n")
    blocking += len(undefined) + len(multi)

    # per section
    rep.append("## Per-section checks\n")
    rep.append("| Section | Words | Sentences | Mean len | >%d words | Paras >%d words | Demo-opener paras | 'This'+verb | Consecutive same opener | Status |" % (th["MAX_SENT"], th["MAX_PARA"]))
    rep.append("|---|---|---|---|---|---|---|---|---|---|")
    details = []
    for head, body in sections(text):
        if re.search(r"reference|bibliograph", head, re.I):
            continue
        paras = paragraphs(body)
        sl, long_s, long_p, demo_p, this_v, consec = [], [], [], [], [], []
        for p in paras:
            pw = len(WORD.findall(p))
            if pw > th["MAX_PARA"]:
                long_p.append(pw)
            ss = sentences(p)
            demos = 0
            prev = None
            for s in ss:
                n = len(WORD.findall(s)); sl.append(n)
                if n > th["MAX_SENT"]:
                    long_s.append((n, s))
                first = (re.match(r"^[^A-Za-z]*([A-Za-z]+)", s) or [None, ""])[1]
                if first in DEMO:
                    demos += 1
                if VERBS_AFTER_THIS.match(s):
                    this_v.append(s)
                if prev and first and prev.lower() == first.lower():
                    consec.append(s)
                prev = first
            if demos > 1:
                demo_p.append(p[:60])
        if not sl:
            continue
        mean = sum(sl) / len(sl)
        flags = []
        is_front = re.search(r"front matter|title|keyword|nomenclature|declaration|information form", head, re.I)
        if not is_front and not (th["MEAN_LO"] <= mean <= th["MEAN_HI"]):
            flags.append("mean")
        if long_s: flags.append("long sent")
        if long_p: flags.append("long para")
        if demo_p: flags.append("demo openers")
        if this_v: flags.append("This+verb")
        if consec: flags.append("repeat opener")
        blocking += len(flags)
        rep.append(f"| {head[:40]} | {sum(sl)} | {len(sl)} | {mean:.1f} | {len(long_s)} | {len(long_p)} | {len(demo_p)} | {len(this_v)} | {len(consec)} | {'FLAG: ' + ', '.join(flags) if flags else 'OK'} |")
        for n, s in long_s:
            details.append(f"- [{head[:30]}] {n} words: {s[:160]}...")
        for s in this_v:
            details.append(f"- [{head[:30]}] 'This' + verb: {s[:120]}")
        for s in consec:
            details.append(f"- [{head[:30]}] repeated opener: {s[:120]}")
        for p in demo_p:
            details.append(f"- [{head[:30]}] >1 This/These/It/There opener in paragraph starting: {p}...")
    rep.append("\n## Details\n")
    rep.append("\n".join(details) if details else "No sentence-level issues.")
    rep.append(f"\n\n**Blocking flags: {blocking}**\n")
    out = "\n".join(rep)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(out)
    print(out)
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
