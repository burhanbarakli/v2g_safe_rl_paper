#!/usr/bin/env python3
"""Build the style corpus from PDFs that a human has downloaded.

The sandbox could not download any corpus PDF (all publisher hosts and dergipark.org.tr are
blocked). Download each paper from the text_source_url in style_corpus/corpus_index.csv,
save it as <pdf_dir>/<key>.pdf, then run:

  python3 -I /home/claude/work/scripts/build_style_corpus.py --pdf-dir /home/claude/work/pdfs/style \
      --index /home/claude/work/style_corpus/corpus_index.csv --out-dir /home/claude/work/style_corpus

For each PDF the script:
- runs pdftotext -layout;
- keeps the text from the Introduction heading up to the References/Bibliography heading;
- drops the back matter (Article Information Form, acknowledgements, declarations);
- drops running headers/footers (lines that repeat on 3 or more pages), page numbers and
  short figure-internal fragments;
- writes numbered headings as '## <heading>' lines;
- writes <key>.txt and updates word_count in the index.

The heuristics are rough. Check every output by eye, especially the Nature-style papers,
whose Methods section comes after the Discussion.
"""
import argparse, collections, csv, os, re, subprocess

HEAD_RE = re.compile(r"^\s*((\d+(\.\d+)*\.?)\s+[A-Z][^.]{2,90}|Introduction|Results|Discussion|Methods|Conclusions?|Related [Ww]ork[s]?)\s*$")
END_RE = re.compile(r"^\s*(References|REFERENCES|Bibliography|Article Information Form|Acknowledg(e)?ments?|Data availability|Declaration of competing interest|CRediT authorship)\b")
START_RE = re.compile(r"^\s*(1\.?\s+)?(Introduction|INTRODUCTION)\s*$")


def clean(raw):
    pages = raw.split("\f")
    counts = collections.Counter()
    for p in pages:
        for l in set(x.strip() for x in p.splitlines() if x.strip()):
            counts[l] += 1
    repeated = {l for l, c in counts.items() if c >= 3 and len(l) < 150}
    lines = []
    for p in pages:
        for l in p.splitlines():
            s = re.sub(r"\s+", " ", l).strip()
            if not s or s in repeated or re.fullmatch(r"\d{1,4}", s):
                continue
            lines.append(s)
    start = next((i for i, l in enumerate(lines) if START_RE.match(l)), 0)
    body = []
    for l in lines[start:]:
        if END_RE.match(l):
            break
        body.append(l)
    out, para = [], []
    for l in body:
        if HEAD_RE.match(l) and len(l.split()) <= 12:
            if para:
                out.append(" ".join(para)); para = []
            out.append("## " + l)
        elif len(l.split()) < 3 and not l.endswith((".", ":")):
            continue  # likely figure-internal text or an axis label
        else:
            para.append(l)
    if para:
        out.append(" ".join(para))
    txt = "\n\n".join(out)
    return re.sub(r"-\s+(?=[a-z])", "", txt)  # re-join hyphenated line breaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.index, encoding="utf-8")))
    for r in rows:
        pdf = os.path.join(a.pdf_dir, r["key"] + ".pdf")
        if not os.path.exists(pdf):
            print("missing", pdf); continue
        raw = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
        txt = clean(raw)
        open(os.path.join(a.out_dir, r["key"] + ".txt"), "w", encoding="utf-8").write(txt + "\n")
        r["word_count"] = str(len(re.findall(r"\b\w+\b", re.sub(r"^## .*$", "", txt, flags=re.M))))
        print(r["key"], r["word_count"])
    with open(a.index, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
