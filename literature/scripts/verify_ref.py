#!/usr/bin/env python3
"""Shared reference verification helper for all literature sub-agents.

Usage:
  python3 verify_ref.py --doi 10.1109/TITS.2024.3510945 --key orfanoudakis2024ev2gym --out parts/X_verified.jsonl
  python3 verify_ref.py --arxiv 2608.00679 --key wang2026hetgps --out parts/X_verified.jsonl

For a DOI it queries Crossref first (https://api.crossref.org/works/{doi}); if that fails it
falls back to OpenAlex (https://api.openalex.org/works/doi:{doi}). For an arXiv ID it queries
the arXiv API. On success it appends one JSON verification record to --out and appends a BibTeX
entry (from doi.org content negotiation for DOIs, generated from arXiv metadata otherwise) to
--bib (default: same stem with .bib). Exit code 0 = verified, 1 = not verified.
"""
import argparse, json, re, sys, time, datetime, urllib.parse
import xml.etree.ElementTree as ET
import requests

UA = {"User-Agent": "v2g-safe-rl-literature-check/1.0 (academic reference verification)"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def get(url, headers=None, tries=3):
    h = dict(UA)
    if headers:
        h.update(headers)
    for i in range(tries):
        try:
            r = requests.get(url, headers=h, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return r
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def verify_doi(doi):
    doi = doi.strip().lower().replace("https://doi.org/", "")
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    r = get(url)
    if r is not None and r.status_code == 200:
        m = r.json()["message"]
        authors = [f"{a.get('given','')} {a.get('family','')}".strip() for a in m.get("author", [])]
        year = None
        for k in ("published-print", "published-online", "issued", "created"):
            if m.get(k, {}).get("date-parts"):
                year = m[k]["date-parts"][0][0]
                break
        return {
            "doi": doi, "source": "crossref", "url_queried": url,
            "title": (m.get("title") or [""])[0],
            "authors": authors, "year": year,
            "venue": (m.get("container-title") or [""])[0] or m.get("publisher", ""),
            "type": m.get("type"),
        }
    url2 = f"https://api.openalex.org/works/doi:{doi}"
    r = get(url2)
    if r is not None and r.status_code == 200:
        m = r.json()
        loc = (m.get("primary_location") or {}).get("source") or {}
        return {
            "doi": doi, "source": "openalex", "url_queried": url2,
            "title": m.get("title") or m.get("display_name"),
            "authors": [a["author"]["display_name"] for a in m.get("authorships", [])],
            "year": m.get("publication_year"), "venue": loc.get("display_name", ""),
            "type": m.get("type"),
        }
    return None


def verify_arxiv(aid):
    aid = aid.strip().replace("arXiv:", "")
    # 1) OpenAlex via the arXiv DataCite DOI (arxiv.org itself is not reachable from this sandbox)
    url0 = f"https://api.openalex.org/works/doi:10.48550/arXiv.{aid}"
    r = get(url0)
    if r is not None and r.status_code == 200:
        m = r.json()
        return {"arxiv": aid, "doi": f"10.48550/arXiv.{aid}", "source": "openalex(arXiv DOI)", "url_queried": url0,
                "title": m.get("title") or m.get("display_name"),
                "authors": [a["author"]["display_name"] for a in m.get("authorships", [])],
                "year": m.get("publication_year"), "venue": "arXiv", "type": "preprint"}
    # 2) Semantic Scholar
    url1 = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{aid}?fields=title,year,authors,venue"
    for _ in range(3):
        r = get(url1)
        if r is not None and r.status_code == 200:
            m = r.json()
            return {"arxiv": aid, "source": "semanticscholar", "url_queried": url1, "title": m.get("title"),
                    "authors": [a["name"] for a in m.get("authors", [])], "year": m.get("year"),
                    "venue": "arXiv", "type": "preprint"}
        time.sleep(4)
    # 3) arXiv API (may be blocked)
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    r = get(url)
    if r is None or r.status_code != 200:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    e = root.find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return None
    title = re.sub(r"\s+", " ", e.find("a:title", ns).text).strip()
    if title.lower() == "error":
        return None
    pub = e.find("a:published", ns).text[:4]
    authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    return {"arxiv": aid, "source": "arxiv", "url_queried": url, "title": title,
            "authors": authors, "year": int(pub), "venue": "arXiv", "type": "preprint"}


def bib_for_doi(doi, key):
    r = get(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"})
    if r is None or r.status_code != 200 or "@" not in r.text:
        return None
    txt = r.text.strip()
    return re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@" + m.group(1) + "{" + key + ",", txt, count=1)


def bib_for_arxiv(rec, key):
    auth = " and ".join(rec["authors"])
    return (f"@misc{{{key},\n  title = {{{rec['title']}}},\n  author = {{{auth}}},\n  year = {{{rec['year']}}},\n"
            f"  eprint = {{{rec['arxiv']}}},\n  archivePrefix = {{arXiv}},\n  url = {{https://arxiv.org/abs/{rec['arxiv']}}}\n}}")


def norm(t):
    return re.sub(r"[^a-z0-9 ]", " ", (t or "").lower()).split()


def title_match(expected, got):
    import difflib
    a, b = " ".join(norm(expected)), " ".join(norm(got))
    if not a or not b:
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    sa, sb = set(a.split()), set(b.split())
    j = len(sa & sb) / max(1, len(sa | sb))
    return max(r, j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi")
    ap.add_argument("--arxiv")
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bib")
    ap.add_argument("--title", required=True, help="Expected title; the returned record must match (>=0.80)")
    a = ap.parse_args()
    rec = verify_doi(a.doi) if a.doi else verify_arxiv(a.arxiv)
    if rec is not None and title_match(a.title, rec.get("title")) < 0.80 and a.arxiv:
        # OpenAlex arXiv-DOI records are sometimes mis-linked; retry Semantic Scholar only
        bad = rec
        rec = None
        url1 = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{a.arxiv}?fields=title,year,authors,venue"
        for _ in range(4):
            r = get(url1)
            if r is not None and r.status_code == 200:
                m = r.json()
                rec = {"arxiv": a.arxiv, "source": "semanticscholar", "url_queried": url1, "title": m.get("title"),
                       "authors": [x["name"] for x in m.get("authors", [])], "year": m.get("year"),
                       "venue": "arXiv", "type": "preprint"}
                break
            time.sleep(5)
    if rec is None:
        print(f"NOT VERIFIED: {a.doi or a.arxiv}")
        sys.exit(1)
    score = title_match(a.title, rec.get("title"))
    if score < 0.80:
        print(f"TITLE MISMATCH ({score:.2f}): expected '{a.title}' but source returned '{rec.get('title')}'. NOT VERIFIED.")
        sys.exit(1)
    rec["title_match_score"] = round(score, 3)
    rec.update({"key": a.key, "verified_at": now(), "status": "verified"})
    with open(a.out, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    bibpath = a.bib or re.sub(r"\.jsonl$", "", a.out) + ".bib"
    bib = bib_for_doi(rec["doi"], a.key) if a.doi else bib_for_arxiv(rec, a.key)
    if bib is None and a.doi:
        auth = " and ".join(rec["authors"])
        bib = (f"@article{{{a.key},\n  title = {{{rec['title']}}},\n  author = {{{auth}}},\n  journal = {{{rec['venue']}}},\n"
               f"  year = {{{rec['year']}}},\n  doi = {{{rec['doi']}}}\n}}")
    with open(bibpath, "a") as f:
        f.write(bib + "\n\n")
    print(f"VERIFIED [{rec['source']}] {rec['year']} | {rec['title']} | {', '.join(rec['authors'][:4])} | {rec['venue']}")


if __name__ == "__main__":
    main()
