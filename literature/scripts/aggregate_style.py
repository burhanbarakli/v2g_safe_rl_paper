#!/usr/bin/env python3
"""Aggregate per-paper style metrics (style_corpus/metrics/*.json) into a profile table + thresholds JSON."""
import json, glob, os, sys, collections, statistics as st
d = sys.argv[1] if len(sys.argv) > 1 else "style_corpus/metrics"
papers = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))}
SECS = ["intro", "related", "method", "setup", "results", "discussion", "conclusion"]
rows = []
for sec in SECS:
    acc = collections.defaultdict(list); n_s = 0; maxes = []
    for k, p in papers.items():
        s = p["sections"].get(sec)
        if not s or not s["sentence_len"] or s["sentence_len"]["n"] < 8: continue
        n = s["sentence_len"]["n"]; n_s += n
        for f in ["frac_over35", "frac_we", "frac_passive", "frac_pastverb", "frac_demo_openers"]:
            acc[f].append((s[f], n))
        acc["mean"].append((s["sentence_len"]["mean"], n)); acc["p90"].append((s["sentence_len"]["p90"], n)); maxes.append(s["sentence_len"]["max"])
        if s["para_len"] and s["para_len"]["n"] >= 3: acc["para_median"].append((s["para_len"]["median"], s["para_len"]["n"]))
    if not n_s: continue
    w = lambda f: round(sum(v * n for v, n in acc[f]) / max(1, sum(n for _, n in acc[f])), 3) if acc[f] else None
    rows.append(dict(section=sec, papers=len(acc["mean"]), sentences=n_s, mean_len=w("mean"), p90_len=w("p90"), median_max=st.median(maxes),
                     over35=w("frac_over35"), we=w("frac_we"), passive=w("frac_passive"), past=w("frac_pastverb"), demo=w("frac_demo_openers"), para_median=w("para_median")))
tw = sum(p["total_words"] for p in papers.values())
def per1k(field):
    c = collections.Counter()
    for p in papers.values():
        for k, v in p.get(field, {}).items(): c[k] += v * p["total_words"] / 1000
    return {k: round(1000 * v / tw, 2) for k, v in c.most_common()}
out = dict(papers=list(papers), total_words=tw, sections=rows, transitions_per1k=per1k("transitions_per1k"), restricted_per1k=per1k("restricted_per1k"),
           em_dash_per1k=round(sum(p["em_dashes_per1k"] * p["total_words"] for p in papers.values()) / tw, 2),
           percent_no_space=sum(p["percent_no_space"] for p in papers.values()), percent_with_space=sum(p["percent_with_space"] for p in papers.values()))
print(json.dumps(out, indent=1))
