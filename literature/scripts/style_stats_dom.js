window.__parasFromDoc = (doc) => {
  const SECT = [["intro", /introduc|background|motivation/i],["related", /related|literature|prior work|state of the art/i],["method", /method|model|formulat|framework|approach|proposed|architecture|algorithm|design|system|problem/i],["setup", /setup|set-up|experiment|case stud|simulation|implementation|data/i],["results", /result|evaluation|performance|analysis|comparison/i],["discussion", /discussion|limitation|implication/i],["conclusion", /conclu|summary|future/i]];
  const SKIP = /reference|acknowledg|funding|author contribution|competing|conflict|data availability|code availability|ethics|additional information|supplementary|rights|about this|appendix|nomenclature|abbreviation|declaration|credit|article information|bibliograph/i;
  const cls = h => { if (SKIP.test(h)) return null; for (const [k, r] of SECT) if (r.test(h)) return k; return "other"; };
  const root = doc.querySelector("article") || doc.querySelector("main") || doc.body;
  root.querySelectorAll("math, .MathJax, mjx-container, .ltx_Math, .ltx_equation, .ltx_equationgroup, script, style, figure, figcaption, table, .c-article-references, .ltx_bibliography, .html-fig_show, .html-table_show, .ltx_note").forEach(e => e.replaceWith(doc.createTextNode(" X ")));
  let cur = "other"; const paras = []; const heads = [];
  for (const n of root.querySelectorAll("h2, h3, p")) {
    if (n.tagName === "H2") { const h = n.textContent.replace(/\s+/g, " ").trim(); heads.push(h); cur = cls(h); continue; }
    if (n.tagName === "H3") { if (cur !== null && SKIP.test(n.textContent)) cur = null; continue; }
    if (cur === null) continue;
    if (n.closest("figure, figcaption, table, nav, header, footer, aside, .ltx_abstract, .c-article-section__content--abstract, #Abs1-content, .abstract, .html-abstract, .ltx_bibliography")) continue;
    const t = n.textContent.replace(/\s+/g, " ").trim();
    if (t.split(" ").length < 15) continue;
    paras.push([cur, t]);
  }
  return { paras, heads };
};
window.__statsFromParas = (paras) => {
  const ABBR = /\b(e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Ref|Refs|Sec|vs|approx|resp|cf|No|Tab|al)\.$/;
  const splitS = p => { const raw = p.split(/(?<=[.!?])\s+(?=[A-Z(\[])/); const out = []; for (const s of raw) { if (out.length && ABBR.test(out[out.length - 1])) out[out.length - 1] += " " + s; else out.push(s); } return out.filter(s => s.split(" ").length >= 3); };
  const wc = s => (s.match(/[A-Za-z0-9][A-Za-z0-9\-']*/g) || []).length;
  const TRANS = ["however","therefore","thus","hence","moreover","furthermore","additionally","in addition","in contrast","consequently","similarly","meanwhile","specifically","in particular","for example","for instance","finally","first","second","nevertheless","instead","accordingly","notably","overall","as a result","on the other hand"];
  const RESTR = ["delve","crucial","pivotal","paramount","landscape","realm","intricate","multifaceted","seamless","cutting-edge","unlock","harness","foster","underscore","showcase","holistic","synergy","plethora","myriad","meticulous","noteworthy","groundbreaking","notably","it is worth noting","it is important to note","paves the way","sheds light","furthermore","moreover","additionally","significantly","robust","leverage","comprehensive","novel","effectively","efficient","efficiently","enhance","facilitate","utilize"];
  const cnt = (txt, w) => (txt.match(new RegExp("\\b" + w.replace(/[-]/g, "\\-").replace(/ /g, "\\s+") + "\\w*", "gi")) || []).length;
  const PASS = /\b(is|are|was|were|be|been|being)\s+(\w+ly\s+)?(\w+ed|\w+en|built|made|shown|known|set|put|kept|held|run|done|given|taken|found|seen|chosen|used|sent)\b/i;
  const PASTV = /\b(was|were|achieved|obtained|observed|reduced|increased|outperformed|reached|showed|resulted|trained|tested|evaluated|compared|performed|yielded|decreased)\b/i;
  const by = {}; let all = "";
  for (const [sec, p] of paras) {
    const S = by[sec] || (by[sec] = { paras: [], sents: [], words: 0, we: 0, passive: 0, past: 0, openersDemo: 0, consecSame: 0 });
    const ss = splitS(p); S.paras.push(wc(p)); all += " " + p; let prev = null;
    for (const s of ss) { const n = wc(s); S.sents.push(n); S.words += n; if (/\b(we|our|us)\b/i.test(s)) S.we++; if (PASS.test(s)) S.passive++; if (PASTV.test(s)) S.past++; const first = (s.match(/^[^A-Za-z]*([A-Za-z]+)/) || [, ""])[1]; if (/^(This|These|It|There)$/.test(first)) S.openersDemo++; if (prev && first && prev.toLowerCase() === first.toLowerCase()) S.consecSame++; prev = first; }
  }
  const stat = a => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); const m = a.reduce((x, y) => x + y, 0) / a.length; return { n: a.length, mean: +m.toFixed(1), median: s[Math.floor(s.length / 2)], p90: s[Math.floor(s.length * 0.9)], max: s[s.length - 1] }; };
  const out = { sections: {}, total_words: 0 };
  for (const [k, S] of Object.entries(by)) { out.total_words += S.words; const d = Math.max(1, S.sents.length); out.sections[k] = { words: S.words, sentence_len: stat(S.sents), para_len: stat(S.paras), frac_over35: +(S.sents.filter(x => x > 35).length / d).toFixed(3), frac_we: +(S.we / d).toFixed(3), frac_passive: +(S.passive / d).toFixed(3), frac_pastverb: +(S.past / d).toFixed(3), frac_demo_openers: +(S.openersDemo / d).toFixed(3), consec_same_opener: S.consecSame }; }
  const W = Math.max(1, out.total_words);
  out.transitions_per1k = Object.fromEntries(TRANS.map(t => [t, +(1000 * cnt(all, t) / W).toFixed(2)]).filter(x => x[1] > 0));
  out.restricted_per1k = Object.fromEntries(RESTR.map(t => [t, +(1000 * cnt(all, t) / W).toFixed(2)]).filter(x => x[1] > 0));
  out.em_dashes_per1k = +(1000 * (all.match(/\u2014/g) || []).length / W).toFixed(2);
  out.percent_with_space = (all.match(/\d\s%/g) || []).length; out.percent_no_space = (all.match(/\d%/g) || []).length; out.pm_usage = (all.match(/±/g) || []).length;
  return out;
};
window.__runAll = async (items) => {
  const res = {};
  for (const [k, url] of Object.entries(items)) {
    try { const html = await (await fetch(url)).text(); const doc = new DOMParser().parseFromString(html, "text/html"); const { paras, heads } = window.__parasFromDoc(doc); const s = window.__statsFromParas(paras); s.url = url; s.heads = heads.slice(0, 20); s.source = "HTML full text, parsed in browser"; res[k] = s; }
    catch (e) { res[k] = "ERR " + e; }
  }
  return JSON.stringify(res);
};
"defined"
