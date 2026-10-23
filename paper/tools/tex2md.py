#!/usr/bin/env python3
"""Build the Markdown review mirrors from the LaTeX sources.

Usage: python3 tools/tex2md.py [main|supp|all]   (default: all; run after compiling both
documents, so that manuscript/main.aux and supplementary/supplementary.aux exist)
  main -> manuscript.md (from manuscript/main.tex)
  supp -> supplementary.md (from supplementary/supplementary.tex)

The converter is tailored to the macros and environments used in this manuscript
(sections, equations, theorem-like environments, figures, booktabs tables,
algorithms, enumerate). Cross-references are resolved from main.aux, citations
become [key, key], math stays in $...$ / $$...$$. Result placeholders become
"[Result: ...]" and author tasks "[TODO: ...]".
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MS = os.path.join(REPO, "manuscript")
SUPP = os.path.join(REPO, "supplementary")
DOC = MS  # directory of the document being converted (set in main)
ANALYSIS = "../../analysis"  # default of \\analysisdir in preamble.tex

MACROS = {  # simple zero-argument macros expanded everywhere (incl. math)
    r"\Ctil": r"\tilde C", r"\Crev": r"C_{\mathrm{rev}}", r"\Bh": r"\hat B",
    r"\Pnom": r"\bar P^{\mathrm{nom}}", r"\Pdis": r"P^{\mathrm{dis}}", r"\Ereq": r"E^{\mathrm{req}}",
    r"\pmn": r"p^{\min}", r"\R": r"\mathbb R", r"\clip": r"\operatorname{clip}",
}


def read(p):
    return open(p, encoding="utf-8").read()


def _aux(path, prefix=""):
    lab = {}
    if os.path.exists(path):
        for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", read(path)):
            lab[prefix + m.group(1)] = m.group(2)
    return lab


def labels(doc):
    """Own labels plus the external labels used through xr (S- in main, M- in supp)."""
    main_aux, supp_aux = os.path.join(MS, "main.aux"), os.path.join(SUPP, "supplementary.aux")
    if doc == "supp":
        lab = _aux(supp_aux); lab.update(_aux(main_aux, "M-"))
    else:
        lab = _aux(main_aux); lab.update(_aux(supp_aux, "S-"))
    return lab


LAB = labels("main")
PROTECT = []  # verbatim Markdown blocks (generated tables) restored after conversion


def protect(md):
    PROTECT.append(md)
    return "\x02" + str(len(PROTECT) - 1) + "\x02"


def arg(s, i):
    """Return (content, end) of the brace group starting at s[i] == '{'."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "{" and (k == 0 or s[k - 1] != "\\"):
            depth += 1
        elif s[k] == "}" and s[k - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:k], k + 1
    raise ValueError("unbalanced braces near: " + s[i:i + 60])


def sub_cmd(s, name, nargs, fn):
    """Replace \\name{a1}..{an} by fn(*args), handling nested braces."""
    out, i, pat = [], 0, re.compile(r"\\" + name + r"(?![A-Za-z])\s*")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:]); break
        j = m.end()
        if nargs and (j >= len(s) or s[j] != "{"):
            out.append(s[i:j]); i = j; continue
        args = []
        for _ in range(nargs):
            a, j = arg(s, j)
            args.append(a)
            while j < len(s) and s[j] == " " and nargs > 1:
                j += 1
        out.append(s[i:m.start()]); out.append(fn(*args)); i = j
    return "".join(out)


def ref(k):
    return LAB.get(k.strip(), "??")


def inline(s):
    """Convert inline LaTeX to Markdown (text and math)."""
    s = re.sub(r"(?<!\\)%.*", "", s)
    for a, b in MACROS.items():
        s = re.sub(re.escape(a) + r"(?![A-Za-z])", lambda m: b, s)
    s = re.sub(r"\\sm(?![A-Za-z])", r"\\mathrm", s)
    maths = []

    def keep(m):
        maths.append(m.group(0))
        return "\x00" + str(len(maths) - 1) + "\x00"
    s = re.sub(r"\$\$.*?\$\$", keep, s, flags=re.S)
    s = re.sub(r"(?<!\\)\$[^$]*?(?<!\\)\$", keep, s)
    s = sub_cmd(s, "method", 0, lambda: "SA-Attn")
    s = s.replace("SA-Attn{}", "SA-Attn")
    s = sub_cmd(s, "RES", 1, lambda a: "[Result: " + a.strip() + "]")
    s = sub_cmd(s, "TODO", 1, lambda a: "[TODO: " + a.strip() + "]")
    s = sub_cmd(s, "Eqs", 2, lambda a, b: f"Equations {ref(a)}-{ref(b)}")
    s = sub_cmd(s, "Eq", 1, lambda a: f"Equation {ref(a)}")
    s = sub_cmd(s, "Fig", 1, lambda a: f"Figure {ref(a)}")
    s = sub_cmd(s, "Tab", 1, lambda a: f"Table {ref(a)}")
    s = sub_cmd(s, "Sec", 1, lambda a: f"Section {ref(a)}")
    s = sub_cmd(s, "Alg", 1, lambda a: f"Algorithm {ref(a)}")
    s = sub_cmd(s, "ref", 1, ref)
    s = sub_cmd(s, "cite", 1, lambda a: "[" + ", ".join(k.strip() for k in a.split(",")) + "]")
    s = sub_cmd(s, "textcolor", 2, lambda c, a: a)
    s = sub_cmd(s, "textbf", 1, lambda a: "**" + a + "**")
    s = sub_cmd(s, "emph", 1, lambda a: "*" + a + "*")
    s = sub_cmd(s, "texttt", 1, lambda a: "`" + a + "`")
    s = sub_cmd(s, "detokenize", 1, lambda a: a)
    s = sub_cmd(s, "multicolumn", 3, lambda a, b, c: c)
    s = s.replace(r"\texteuro{}", "€").replace(r"\texteuro", "€").replace(r"\texttrademark", "™")
    s = s.replace(r"\euro{}", "€").replace(r"\euro", "€")
    s = s.replace(r"\%", "\x01").replace(r"\&", "&").replace(r"\_", "_").replace(r"i.e.\ ", "i.e. ").replace(r"e.g.\ ", "e.g. ")
    s = s.replace("``", '"').replace("''", '"').replace("~", " ").replace("---", "-").replace("--", "–")
    s = re.sub(r"\\(noindent|centering|footnotesize|small|par|medskip|bigskip)\b\s*", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\\\(\[[^\]]*\])?", "\n", s)
    for _ in range(3):
        s = re.sub("\x00(\\d+)\x00", lambda m: maths[int(m.group(1))], s)
    return s


def convert_table(body, num, caption):
    tab = re.search(r"\\begin\{tabular\}(\{.*?\})?\{(.*?)\}\n(.*?)\\end\{tabular\}", body, re.S)
    rows = []
    if tab:
        for line in tab.group(3).split("\\\\"):
            line = re.sub(r"\\(toprule|midrule|bottomrule)", "", line)
            line = re.sub(r"\\cmidrule\([a-z]*\)\{[^}]*\}", "", line).strip()
            if not line:
                continue
            cells = [inline(c).strip().replace("\n", " ") for c in line.split("&")]
            rows.append("| " + " | ".join(cells) + " |")
    md = [f"**Table {num}.** {inline(caption).strip()}", ""]
    if rows:
        ncol = rows[0].count("|") - 1
        md += [rows[0], "|" + "---|" * ncol] + rows[1:]
    return "\n".join(md)


def convert_algorithm(body, num, caption):
    alg = re.search(r"\\begin\{algorithmic\}(\[\d\])?(.*?)\\end\{algorithmic\}", body, re.S).group(2)
    lines = []
    depth = 0
    for raw in alg.strip().splitlines():
        l = raw.strip()
        if not l:
            continue
        l = re.sub(r"\\(EndFor|EndWhile|EndIf)\b", "end", l)
        l = re.sub(r"\\Require\b", "Input:", l)
        l = re.sub(r"\\Return\b", "return", l)
        l = re.sub(r"\\State\b\s*", "", l)
        l = sub_cmd(l, "For", 1, lambda a: "for " + a + ":")
        l = sub_cmd(l, "While", 1, lambda a: "while " + a + ":")
        l = sub_cmd(l, "If", 1, lambda a: "if " + a + ":")
        l = sub_cmd(l, "Comment", 1, lambda a: "  # " + a)
        l = re.sub(r"\\Else\b", "else:", l)
        l = sub_cmd(l, "textbf", 1, lambda a: a)
        l = inline(l).replace("\n", " ")
        lines.append(l)
    return f"**Algorithm {num}.** {inline(caption).strip()}\n\n@@FENCE@@\n" + "\n".join(lines) + "\n@@FENCE@@"


def convert_figure(body, num, caption):
    m = re.search(r"\\(figorbox|resultfig)\{([^}]*)\}", body)
    path = m.group(2) if m else ""
    fc = re.search(r"\\figchain\{[^}]*\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", body)
    if fc:
        cands = [fc.group(1), fc.group(2), fc.group(3)]
        path = next((c for c in cands if os.path.exists(os.path.join(DOC, c))), cands[-1])
    note = ""
    if m and m.group(1) == "resultfig":
        fname = path
        path = ANALYSIS + "/figures/" + fname
        if not any(os.path.exists(c) for c in (os.path.join(DOC, ANALYSIS, "figures", fname),
                                                os.path.join(DOC, "figures", fname))):
            rest = body[m.end():]
            j = rest.index("{")
            note, _ = arg(rest, j)
            note = " (result figure placeholder: " + inline(note).strip() + ")"
    return f"![Figure {num}]({path})\n\n*Figure {num}. {inline(caption).strip()}*{note}"


ENV_NAMES = {"theorem": "Theorem", "lemma": "Lemma", "proposition": "Proposition", "corollary": "Corollary",
             "assumption": "Assumption", "remark": "Remark"}


def block(s):
    """Convert block structure of one section file."""
    s = re.sub(r"(?m)^\s*%.*\n", "", s)
    s = re.sub(r"\\ifanonymous\s.*?\\else\s(.*?)\\fi", lambda m: m.group(1), s, flags=re.S)
    s = sub_cmd(s, "resulttable", 2, lambda path, ph: read(os.path.join(DOC, path))
                if os.path.exists(os.path.join(DOC, path)) else ph)
    s = s.replace("\\analysisdir", ANALYSIS)
    def _restable(name, ph):
        # Prefer the Markdown version written by the analysis pipeline; the LaTeX table is
        # used for the label (table number) only.
        for d in (os.path.join(DOC, ANALYSIS, "tables"), os.path.join(DOC, "tables")):
            md, tex = os.path.join(d, name + ".md"), os.path.join(d, name + ".tex")
            if os.path.exists(md):
                lm = re.search(r"\\label\{([^}]*)\}", read(tex) if os.path.exists(tex) else ph)
                num = ref(lm.group(1)) if lm else name
                txt = read(md).strip()
                txt = re.sub(r"^\*\*" + re.escape(name) + r"\.\*\*", f"**Table {num}.**", txt, count=1)
                # separator rows with four hyphens: valid Markdown, and not read as an em dash by the lint
                txt = re.sub(r"(?m)^\|(?:\s*:?-+:?\s*\|)+\s*$",
                             lambda m: re.sub(r"-+", "----", m.group(0)), txt)
                return "\n\n" + protect(txt) + "\n\n"
            if os.path.exists(tex):
                return read(tex)
        return ph
    s = sub_cmd(s, "restable", 2, _restable)
    s = sub_cmd(s, "placeholdertable", 3, lambda lab, cap, f: "\\begin{table}\\caption{" + cap
                + " (placeholder until the analysis scripts have run)}\\label{" + lab
                + "}\\end{table}")
    s = re.sub(r"(?<!\\)%.*", "", s)
    out = []

    def caption_of(b):
        i = b.index(r"\caption")
        j = b.index("{", i)
        c, _ = arg(b, j)
        return c

    def labnum(b):
        m = re.search(r"\\label\{([^}]*)\}", b)
        return ref(m.group(1)) if m else "?"

    # figures, tables, algorithms
    def envrep(m):
        env, b = m.group(1), m.group(2)
        if env == "table":
            return "\n\n" + convert_table(b, labnum(b), caption_of(b)) + "\n\n"
        if env == "figure":
            return "\n\n" + convert_figure(b, labnum(b), caption_of(b)) + "\n\n"
        return "\n\n" + convert_algorithm(b, labnum(b), caption_of(b)) + "\n\n"
    s = re.sub(r"\\begin\{(table|figure|algorithm)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}", envrep, s, flags=re.S)
    # inline (non-float) tables inside center, e.g. the conditions box of Section 4.4
    def centrep(m):
        b = m.group(1)
        if "\\begin{tabular}" not in b:
            return m.group(0)
        head = re.search(r"\\textbf\{(.*?)\}\\\\", b)
        title = head.group(1) if head else ""
        b = re.sub(r"\\textbf\{.*?\}\\\\(\[[^\]]*\])?", "", b, count=1)
        md = convert_table(b, "", title)
        return "\n\n" + md.replace("**Table .** ", "**") .replace(inline(title).strip(), inline(title).strip() + "**", 1) + "\n\n"
    s = re.sub(r"\\begin\{center\}(.*?)\\end\{center\}", centrep, s, flags=re.S)

    # display math
    def eqrep(m):
        b = m.group(2)
        num = labnum(b)
        b = re.sub(r"\\label\{[^}]*\}", "", b).strip()
        b = inline(b) if False else b
        for a, r in MACROS.items():
            b = re.sub(re.escape(a) + r"(?![A-Za-z])", lambda mm: r, b)
        b = re.sub(r"\\sm(?![A-Za-z])", r"\\mathrm", b)
        tag = f" \\qquad ({num})" if num != "?" else ""
        return "\n\n$$" + b + tag + "$$\n\n"
    s = re.sub(r"\\begin\{(equation)\}(.*?)\\end\{equation\}", eqrep, s, flags=re.S)
    s = re.sub(r"\\\[(.*?)\\\]", lambda m: "\n\n$$" + m.group(1).strip() + "$$\n\n", s, flags=re.S)

    # theorem-like
    def thmrep(m):
        env, opt, b = m.group(1), m.group(2), m.group(3)
        num = labnum(b)
        head = ENV_NAMES[env] + ("" if env == "assumption" else " " + num)
        if opt:
            head += f" ({inline(opt)})"
        return f"\n\n**{head}.** " + b.strip() + "\n\n"
    s = re.sub(r"\\begin\{(theorem|lemma|proposition|corollary|assumption|remark)\}(?:\[([^\]]*)\])?(.*?)\\end\{\1\}",
               thmrep, s, flags=re.S)

    def proofrep(m):
        opt, b = m.group(1), m.group(2)
        head = inline(opt) if opt else "Proof"
        sep = ":" if head.startswith("Proof of") else "."
        return f"\n\n*{head}{sep}* " + b.strip() + " ∎\n\n"
    s = re.sub(r"\\begin\{proof\}(?:\[([^\]]*)\])?(.*?)\\end\{proof\}", proofrep, s, flags=re.S)

    # lists
    def listrep(m):
        env, b = m.group(1), m.group(2)
        items = re.split(r"\\item(?:\[([^\]]*)\])?\s*", b)
        res, n = [], 0
        k = 1
        while k < len(items):
            lab, txt = items[k], items[k + 1]
            n += 1
            mark = (lab + " ") if lab else (f"{n}. " if env == "enumerate" else "- ")
            res.append(mark + " ".join(txt.split()))
            k += 2
        return "\n\n" + "\n\n".join(res) + "\n\n"
    s = re.sub(r"\\begin\{(enumerate|itemize)\}(.*?)\\end\{\1\}", listrep, s, flags=re.S)

    # abstract and sections
    s = re.sub(r"\\begin\{abstract\}", "\n## Abstract\n\n", s)
    s = re.sub(r"\\end\{abstract\}", "\n", s)
    return s


def headings(s, appendix=False, counters=None):
    out, i = [], 0
    pat = re.compile(r"\\(section|subsection)(\*?)\{")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:]); break
        title, j = arg(s, m.end() - 1)
        lab = None
        lm = re.match(r"\\label\{([^}]*)\}", s[j:])
        if lm:
            lab, j = lm.group(1), j + lm.end()
        out.append(s[i:m.start()])
        if m.group(2):
            out.append(f"\n## {title}\n")
        else:
            num = ref(lab) if lab else ""
            out.append(("\n## " if m.group(1) == "section" else "\n### ") + (f"{num} " if num else "") + title + "\n")
        i = j
    return "".join(out)


def references(keys=None):
    bib = read(os.path.join(MS, "refs.bib"))
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", bib):
        key = m.group(2)
        if keys is not None and key not in keys:
            continue
        seg = bib[m.end():bib.find("\n@", m.end()) if bib.find("\n@", m.end()) > 0 else len(bib)]
        t = re.search(r"title\s*=\s*\{(.*?)\}\s*,", seg, re.S)
        y = re.search(r"year\s*=\s*\{?(\d{4})", seg)
        out.append(f"[{key}] {t.group(1) if t else ''} ({y.group(1) if y else ''})")
    return "```\n" + "\n".join(out) + "\n```"


def convert(doc):
    global DOC, LAB
    DOC = MS if doc == "main" else SUPP
    LAB = labels(doc)
    if doc == "main":
        src = read(os.path.join(MS, "main.tex"))
        title = re.search(r"\\title\{(.*?)\}\n", src).group(1)
        out = os.path.join(REPO, "manuscript.md")
        note = ("*This Markdown mirror is generated by `tools/tex2md.py` from the LaTeX source for review purposes only. "
                "The LaTeX files and `manuscript/main.pdf` are authoritative, and author names are still placeholders "
                "[TODO: fill in main.tex].*")
    else:
        src = read(os.path.join(SUPP, "supplementary.tex"))
        raw_title = re.search(r"\\title\{(.*?)\}\n", src).group(1)
        title = "Supplementary Material: " + re.sub(r"^.*?\\large\s*", "", raw_title)
        out = os.path.join(REPO, "supplementary.md")
        note = ("*This Markdown mirror is generated by `tools/tex2md.py` from the LaTeX source of the supplementary material. "
                "The LaTeX files and `supplementary/supplementary.pdf` are authoritative for review and submission.*")
    parts = [f"# {title}", "", note, ""]
    body = src[src.index("\\begin{document}"):]
    intro = re.search(r"\\maketitle\s*(.*?)\\input", body, re.S)
    if intro and intro.group(1).strip():
        parts.append(inline(block(intro.group(1))))
    order = re.findall(r"\\input\{sections/([^}]*)\}", src)
    cited = set()
    for name in order:
        raw = read(os.path.join(DOC, "sections", name + ".tex"))
        for m in re.finditer(r"\\cite\{([^}]*)\}", re.sub(r"(?<!\\)%.*", "", raw)):
            cited.update(k.strip() for k in m.group(1).split(","))
        s = block(raw)
        s = headings(s)
        s = inline(s)
        parts.append(s)
    parts.append("\n## References\n\n" + references(cited))
    md = "\n".join(parts).replace("@@FENCE@@", "```")
    md = re.sub("\x02(\\d+)\x02", lambda m: PROTECT[int(m.group(1))], md)
    md = md.replace("\x01", "%")
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    open(out, "w", encoding="utf-8").write(md)
    print("wrote", out, len(md.split()), "words")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for doc in (["main", "supp"] if which == "all" else [which]):
        convert(doc)


if __name__ == "__main__":
    main()
