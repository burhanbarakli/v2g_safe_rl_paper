"""Generate fig_concepts.svg: main-text Fig. 2 (DD29), two compact panels at 12 px text on a 940 px canvas
(= 6 pt at 6.5 in width):
  (a) per-EV bounds from backward reachability (same toy model as make_lower_bound.py)
  (b) safety-layer projection for two EVs (same example as make_projection_geometry.py)

Numbers (checked below):
  (a) E_min=10, E_req=55, E_cap=60, phi(E)=E+min(6, 0.35(E_cap-E)); B(k) = 55, 52.31, 48.17, 42.17, 36.17,
      30.17, 24.17, 18.17, 12.17, 10, ...; t_arr=1, d=12, t=7, E(t)=27: l*dt=3.17, u*dt=6, p_min*dt=-6
  (b) l=(-5,-3), u=(6,5), C_tilde=6, C_rev=4, x=(12,3): clip(x)=(6,3) infeasible; lambda*=3;
      x-lambda*1=(9,0); p*=(6,0) (verified against method/projection_ref.project if available)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE.parents[1] / "method"
sub = '<tspan class="sub" baseline-shift="sub">{}</tspan>'
sup = '<tspan class="sub" baseline-shift="super">{}</tspan>'

# ============================================================== (a) numbers
Emin, Ereq, Ecap = 10.0, 55.0, 60.0
d, tarr, tnow = 12, 1, 7


def phi(E):
    return E + min(6.0, 0.35 * (Ecap - E))


def B(k):
    b = Ereq
    for _ in range(k):
        if phi(Emin) >= b:
            b = Emin
            continue
        lo, hi = Emin, b
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if phi(mid) >= b else (mid, hi)
        b = hi
    return b


Bs = [B(k) for k in range(13)]
hist = {1: 18, 2: 24, 3: 28, 4: 25, 5: 21, 6: 22, 7: 27}
for s_, e in hist.items():
    assert e >= B(d - s_ + 1) - 1e-9
E = hist[tnow]
tau = d - tnow + 1
l_dt, u_dt, pmin_dt = B(tau - 1) - E, phi(E) - E, -min(6, E - Emin)
assert abs(l_dt - 3.17) < 0.01 and u_dt == 6 and pmin_dt == -6
env = [E]
for _ in range(d + 1 - tnow):
    env.append(phi(env[-1]))
assert env[-1] >= Ereq

# ============================================================== (b) numbers
l = np.array([-5.0, -3.0]); u = np.array([6.0, 5.0]); Ct, Crev = 6.0, 4.0; x = np.array([12.0, 3.0])
lo_, hi_ = 0.0, 100.0
for _ in range(200):
    mid = 0.5 * (lo_ + hi_)
    lo_, hi_ = (mid, hi_) if np.clip(x - mid, l, u).sum() > Ct else (lo_, mid)
lam = hi_
shifted, p, clipx = x - lam, np.clip(x - lam, l, u), np.clip(x, l, u)
assert abs(lam - 3) < 1e-9 and np.allclose(p, [6, 0]) and np.allclose(clipx, [6, 3])
if (REF / "projection_ref.py").exists():
    sys.path.insert(0, str(REF))
    import projection_ref as pr
    out = pr.project(x[None, :], l[None, :], u[None, :], np.array([Ct]), np.array([Crev]))
    assert np.allclose(np.asarray(out[0]).reshape(-1)[:2], p, atol=1e-9)

W, H = 940, 350

# ============================================================== (a) drawing, panel origin (0, 0)
ax0, ax1, ay0, ay1 = 62, 428, 300, 52       # t in [0, 14], E in [0, 64]


def XA(t):
    return ax0 + (ax1 - ax0) * t / 14.0


def YA(e):
    return ay0 + (ay1 - ay0) * e / 64.0


def pa(pts):
    return " ".join(f"{XA(a):.1f},{YA(b):.1f}" for a, b in pts)


latest = sorted([(d + 1 - k, Bs[k]) for k in range(13) if d + 1 - k >= tarr])
region = [(tarr, 0)] + latest + [(d + 1, 0)]
envpts = [(tnow + k, v) for k, v in enumerate(env)]
histpts = sorted(hist.items())


def dots(pts, cls, f):
    return "".join(f'<circle class="{cls}" cx="{f[0](a):.1f}" cy="{f[1](b):.1f}" r="2.3"/>' for a, b in pts)


A = f'''
  <g id="panel-a">
  <text class="pl" x="8" y="20">(a)</text>
  <text class="pt" x="34" y="20">Per-EV bounds from backward reachability</text>
  <polygon points="{pa(region)}" fill="url(#hatch)"/>
  <rect x="{XA(4.0) - 40:.1f}" y="{YA(5.3) - 11:.1f}" width="80" height="15" fill="#fff"/>
  <text class="s" x="{XA(4.0):.1f}" y="{YA(5.3):.1f}" text-anchor="middle">unreachable</text>
  <line class="ax" x1="{ax0}" y1="{ay0}" x2="{ax1 + 14}" y2="{ay0}" marker-end="url(#arr)"/>
  <line class="ax" x1="{ax0}" y1="{ay0}" x2="{ax0}" y2="{ay1 - 12}" marker-end="url(#arr)"/>
  <rect x="{ax1 - 40}" y="{ay0 - 16}" width="54" height="13" fill="#fff"/>
  <text class="s" x="{ax1 + 14}" y="{ay0 - 5}" text-anchor="end">time step</text>
  <text class="s" x="{ax0 + 6}" y="{ay1 - 8}">energy <tspan class="it">E</tspan>{sub.format("i")}</text>
  <line class="guide" x1="{ax0}" y1="{YA(Ecap):.1f}" x2="{XA(13.6):.1f}" y2="{YA(Ecap):.1f}"/>
  <line class="guide" x1="{ax0}" y1="{YA(Ereq):.1f}" x2="{XA(d + 1):.1f}" y2="{YA(Ereq):.1f}"/>
  <line class="floor" x1="{ax0}" y1="{YA(Emin):.1f}" x2="{XA(13.6):.1f}" y2="{YA(Emin):.1f}"/>
  <line class="guide" x1="{ax0}" y1="{YA(E):.1f}" x2="{XA(tnow):.1f}" y2="{YA(E):.1f}"/>
  <text class="s it" x="{ax0 - 5}" y="{YA(Ecap) + 2:.1f}" text-anchor="end">E{sup.format("cap")}</text>
  <text class="s it" x="{ax0 - 5}" y="{YA(Ereq) + 8:.1f}" text-anchor="end">E{sup.format("req")}</text>
  <text class="s it" x="{ax0 - 5}" y="{YA(E) + 4:.1f}" text-anchor="end">E{sub.format("i")}(t)</text>
  <text class="s it" x="{ax0 - 5}" y="{YA(Emin) + 4:.1f}" text-anchor="end">E{sup.format("min")}</text>
  {"".join(f'<line class="guide" x1="{XA(tt):.1f}" y1="{ay0}" x2="{XA(tt):.1f}" y2="{YA(ee):.1f}"/>' for tt, ee in [(tarr, hist[tarr]), (tnow, E), (tnow + 1, env[1]), (d, Bs[1]), (d + 1, Ereq)])}
  <text class="s it" x="{XA(tarr):.1f}" y="{ay0 + 15}" text-anchor="middle">t{sub.format("i")}{sup.format("arr")}</text>
  <text class="s it" x="{XA(tnow):.1f}" y="{ay0 + 15}" text-anchor="middle">t</text>
  <text class="s it" x="{XA(tnow + 1) + 2:.1f}" y="{ay0 + 15}" text-anchor="middle">t+1</text>
  <text class="s it" x="{XA(d) - 3:.1f}" y="{ay0 + 15}" text-anchor="middle">d{sub.format("i")}</text>
  <text class="s it" x="{XA(d + 1) + 6:.1f}" y="{ay0 + 15}" text-anchor="middle">d{sub.format("i")}+1</text>
  <path class="ax" style="stroke-width:1" d="M{XA(tnow):.1f},{ay0 + 22} L{XA(tnow):.1f},{ay0 + 27} L{XA(d + 1):.1f},{ay0 + 27} L{XA(d + 1):.1f},{ay0 + 22}"/>
  <text class="s" x="{(XA(tnow) + XA(d + 1)) / 2:.1f}" y="{ay0 + 41}" text-anchor="middle">τ{sub.format("i")}(t) remaining steps</text>
  <polyline class="hist" points="{pa(histpts)}"/>{dots(histpts[:-1], "dh", (XA, YA))}
  <text class="s" x="{XA(2.9):.1f}" y="{YA(32.5):.1f}" text-anchor="middle">past energy</text>
  <polyline class="latest" points="{pa(latest)}"/>{dots(latest, "dl", (XA, YA))}
  <polyline class="env" points="{pa(envpts)}"/>{dots(envpts[1:], "de", (XA, YA))}
  <text class="s" x="{XA(9.5):.1f}" y="{YA(51):.1f}" text-anchor="end">upper envelope</text>
  <text class="s" x="{XA(9.5):.1f}" y="{YA(51) + 13:.1f}" text-anchor="end">(max. charging)</text>
  <rect x="{XA(10.75) - 3:.1f}" y="{YA(31.5) - 11:.1f}" width="88" height="29" fill="#fff"/>
  <text class="s" x="{XA(10.75):.1f}" y="{YA(31.5):.1f}">latest-start</text>
  <text class="s" x="{XA(10.75):.1f}" y="{YA(31.5) + 13:.1f}">trajectory <tspan class="it">B̂</tspan>{sub.format("i")}</text>
  <line class="lead" x1="{XA(11.3):.1f}" y1="{YA(34.5):.1f}" x2="{XA(10.75):.1f}" y2="{YA(44.5):.1f}"/>
  <text class="s" x="{XA(11.9):.1f}" y="{YA(62.0):.1f}" text-anchor="end">taper</text>
  <line class="lead" x1="{XA(11.95):.1f}" y1="{YA(61.0):.1f}" x2="{XA(12.45):.1f}" y2="{YA(53.4):.1f}"/>
  <line class="step" x1="{XA(tnow):.1f}" y1="{YA(E):.1f}" x2="{XA(tnow + 1) - 3:.1f}" y2="{YA(E + u_dt) + 2:.1f}" marker-end="url(#arr)"/>
  <line class="step" x1="{XA(tnow):.1f}" y1="{YA(E):.1f}" x2="{XA(tnow + 1) - 4:.1f}" y2="{YA(E + l_dt) + 1:.1f}" marker-end="url(#arr)"/>
  <line class="step" style="stroke:#777" x1="{XA(tnow):.1f}" y1="{YA(E):.1f}" x2="{XA(tnow + 1) - 3:.1f}" y2="{YA(E + pmin_dt) - 2:.1f}" marker-end="url(#arrg)"/>
  <circle cx="{XA(tnow):.1f}" cy="{YA(E):.1f}" r="4" fill="#111"/>
  <line x1="{XA(tnow + 1) + 6:.1f}" y1="{YA(E + l_dt):.1f}" x2="{XA(tnow + 1) + 6:.1f}" y2="{YA(E + u_dt):.1f}" stroke="#000" stroke-width="3"/>
  <rect x="{XA(tnow + 1) + 9:.1f}" y="{YA(E + l_dt):.1f}" width="44" height="14" fill="#fff"/>
  <text class="s" x="{XA(tnow + 0.8):.1f}" y="{YA(E + u_dt + 1.6):.1f}" text-anchor="end">+Δ<tspan class="it">t u</tspan>{sub.format("i")}</text>
  <text class="s" x="{XA(tnow + 1) + 11:.1f}" y="{YA(E + l_dt) + 11:.1f}">+Δ<tspan class="it">t l</tspan>{sub.format("i")}</text>
  <text class="s" x="{XA(tnow + 1) + 11:.1f}" y="{YA(E + pmin_dt) + 4:.1f}" fill="#555">+Δ<tspan class="it">t p</tspan>{sub.format("i")}{sup.format("min")}</text>
  </g>'''

# ============================================================== (b) drawing, panel origin (480, 0)
P1, P2 = (-7.2, 13.6), (-5.8, 6.6)
s = 20.0
bx, by = 480 + 40, 40


def XB(a):
    return bx + (a - P1[0]) * s


def YB(b):
    return by + (P2[1] - b) * s


def pb(v):
    return f"{XB(v[0]):.1f},{YB(v[1]):.1f}"


def clip_poly(poly, a, b, c):
    out = []
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = a * P[0] + b * P[1] - c, a * Q[0] + b * Q[1] - c
        if fp <= 0:
            out.append(P)
        if fp * fq < 0:
            t = fp / (fp - fq)
            out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
    return out


box = [(l[0], l[1]), (u[0], l[1]), (u[0], u[1]), (l[0], u[1])]
K = clip_poly(clip_poly(box, 1, 1, Ct), -1, -1, Crev)


def line_sum(c):
    pts = [(a, c - a) for a in P1 if P2[0] <= c - a <= P2[1]] + [(c - b, b) for b in P2 if P1[0] <= c - b <= P1[1]]
    pts = sorted(set(pts))
    return pts[0], pts[-1]


def hatch(c, side, n=12):
    a0, a1 = line_sum(c)
    out = []
    for i in range(1, n):
        t = i / n
        q = (a0[0] + t * (a1[0] - a0[0]), a0[1] + t * (a1[1] - a0[1]))
        e = (q[0] + side * 0.3, q[1] + side * 0.3)
        out.append(f'<line class="hat" x1="{XB(q[0]):.1f}" y1="{YB(q[1]):.1f}" x2="{XB(e[0]):.1f}" y2="{YB(e[1]):.1f}"/>')
    return "".join(out)


cu0, cu1 = line_sum(Ct)
cl0, cl1 = line_sum(-Crev)
Bp = f'''
  <g id="panel-b">
  <text class="pl" x="488" y="20">(b)</text>
  <text class="pt" x="514" y="20">Safety-layer projection (two EVs)</text>
  <line class="axg" x1="{XB(P1[0]):.1f}" y1="{YB(0):.1f}" x2="{XB(P1[1]):.1f}" y2="{YB(0):.1f}" marker-end="url(#arrg)"/>
  <line class="axg" x1="{XB(0):.1f}" y1="{YB(P2[0]):.1f}" x2="{XB(0):.1f}" y2="{YB(P2[1]):.1f}" marker-end="url(#arrg)"/>
  <text class="s it" x="{XB(P1[1]):.1f}" y="{YB(0) + 15:.1f}" text-anchor="end">p{sub.format("1")}</text>
  <text class="s it" x="{XB(0) + 6:.1f}" y="{YB(P2[1]) + 9:.1f}">p{sub.format("2")}</text>
  <polygon class="K" points="{" ".join(pb(v) for v in K)}"/>
  <rect class="box" x="{XB(l[0]):.1f}" y="{YB(u[1]):.1f}" width="{(u[0] - l[0]) * s:.1f}" height="{(u[1] - l[1]) * s:.1f}"/>
  <line class="guide" x1="{XB(u[0]):.1f}" y1="{YB(l[1]):.1f}" x2="{XB(u[0]):.1f}" y2="{YB(P2[0]) - 13:.1f}"/>
  <text class="s it" x="{XB(l[0]) - 4:.1f}" y="{YB(0) + 14:.1f}" text-anchor="end">l{sub.format("1")}</text>
  <text class="s it" x="{XB(u[0]):.1f}" y="{YB(P2[0]) - 1:.1f}" text-anchor="middle">u{sub.format("1")}</text>
  <text class="s it" x="{XB(l[0]) - 4:.1f}" y="{YB(l[1]) + 4:.1f}" text-anchor="end">l{sub.format("2")}</text>
  <text class="s it" x="{XB(l[0]) - 4:.1f}" y="{YB(u[1]) + 4:.1f}" text-anchor="end">u{sub.format("2")}</text>
  <line class="agg" x1="{XB(cu0[0]):.1f}" y1="{YB(cu0[1]):.1f}" x2="{XB(cu1[0]):.1f}" y2="{YB(cu1[1]):.1f}"/>{hatch(Ct, +1)}
  <line class="agg" x1="{XB(cl0[0]):.1f}" y1="{YB(cl0[1]):.1f}" x2="{XB(cl1[0]):.1f}" y2="{YB(cl1[1]):.1f}"/>{hatch(-Crev, -1)}
  <text class="s" x="{XB(-1.0):.1f}" y="{YB(5.95):.1f}" text-anchor="end">Σ<tspan class="it">p</tspan>{sub.format("i")} = C̃</text>
  <text class="s" x="{XB(-7.0):.1f}" y="{YB(-4.9):.1f}">Σ<tspan class="it">p</tspan>{sub.format("i")} = −C{sub.format("rev")}</text>
  <text class="lab h" x="{XB(-2.6):.1f}" y="{YB(2.0):.1f}" text-anchor="middle">𝒦(t)</text>
  <line class="naive" x1="{XB(x[0]):.1f}" y1="{YB(x[1]):.1f}" x2="{XB(clipx[0]) + 6:.1f}" y2="{YB(clipx[1]):.1f}"/>
  <rect x="{XB(clipx[0]) - 4.5:.1f}" y="{YB(clipx[1]) - 4.5:.1f}" width="9" height="9" fill="#fff" stroke="#555" stroke-width="1.3" transform="rotate(45 {XB(clipx[0]):.1f} {YB(clipx[1]):.1f})"/>
  <text class="s" x="{XB(clipx[0]) + 0.5 * s:.1f}" y="{YB(clipx[1]) + 0.8 * s:.1f}">clip(<tspan class="it">x</tspan>, <tspan class="it">l</tspan>, <tspan class="it">u</tspan>)</text>
  <text class="s" x="{XB(clipx[0]) + 0.5 * s:.1f}" y="{YB(clipx[1]) + 0.8 * s + 13:.1f}" fill="#555">infeasible</text>
  <line class="shift" x1="{XB(x[0]) - 4:.1f}" y1="{YB(x[1]) + 4:.1f}" x2="{XB(shifted[0]) + 5:.1f}" y2="{YB(shifted[1]) - 5:.1f}" marker-end="url(#arr)"/>
  <line class="clipa" x1="{XB(shifted[0]) - 5:.1f}" y1="{YB(0):.1f}" x2="{XB(p[0]) + 7:.1f}" y2="{YB(p[1]):.1f}" marker-end="url(#arr)"/>
  <circle cx="{XB(x[0]):.1f}" cy="{YB(x[1]):.1f}" r="4" fill="#111"/>
  <text class="lab it" x="{XB(x[0]) + 7:.1f}" y="{YB(x[1]) - 6:.1f}">x</text>
  <text class="s" x="{XB(11.1):.1f}" y="{YB(1.0):.1f}">−λ*(1, 1)</text>
  <circle cx="{XB(shifted[0]):.1f}" cy="{YB(shifted[1]):.1f}" r="4" fill="#fff" stroke="#111" stroke-width="1.5"/>
  <text class="s" x="{XB(shifted[0]) + 2:.1f}" y="{YB(0) + 17:.1f}"><tspan class="it">x</tspan> − λ*<tspan class="h">1</tspan></text>
  <text class="s" x="{XB(7.5):.1f}" y="{YB(0) - 6:.1f}" text-anchor="middle">clip</text>
  <rect x="{XB(p[0]) - 4.5:.1f}" y="{YB(p[1]) - 4.5:.1f}" width="9" height="9" fill="#111"/>
  <text class="lab h" x="{XB(p[0]) - 7:.1f}" y="{YB(p[1]) + 17:.1f}" text-anchor="end"><tspan class="it">p</tspan>* = Π<tspan class="sub" baseline-shift="sub">t</tspan>(<tspan class="it">x</tspan>)</text>
  </g>'''

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111">
  <title>Per-EV bounds and safety-layer projection</title>
  <!-- Generated by make_concepts.py (main-text Fig. 2, DD29). Numbers: see the script docstring. -->
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker>
    <marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#777"/></marker>
    <pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#bbb" stroke-width="1.1"/></pattern>
    <style>
      .ax {{ stroke: #222; stroke-width: 1.2; fill: none; }}
      .axg {{ stroke: #777; stroke-width: 1; fill: none; }}
      .guide {{ stroke: #999; stroke-width: 1; stroke-dasharray: 2 3; fill: none; }}
      .lead {{ stroke: #666; stroke-width: 0.9; fill: none; }}
      .latest {{ stroke: #000; stroke-width: 2; fill: none; }}
      .env {{ stroke: #222; stroke-width: 1.4; fill: none; stroke-dasharray: 6 3; }}
      .hist {{ stroke: #555; stroke-width: 1.4; fill: none; }}
      .floor {{ stroke: #222; stroke-width: 1.2; fill: none; stroke-dasharray: 2 2; }}
      .dl {{ fill: #000; }} .de {{ fill: #fff; stroke: #222; stroke-width: 1.1; }} .dh {{ fill: #555; }}
      .step {{ stroke: #000; stroke-width: 1.5; fill: none; }}
      .box {{ fill: none; stroke: #222; stroke-width: 1.3; }}
      .K {{ fill: #dedede; stroke: #000; stroke-width: 2; }}
      .agg {{ stroke: #222; stroke-width: 1.3; fill: none; }}
      .hat {{ stroke: #222; stroke-width: 1; }}
      .shift {{ stroke: #222; stroke-width: 1.5; stroke-dasharray: 6 3; fill: none; }}
      .clipa {{ stroke: #222; stroke-width: 1.5; fill: none; }}
      .naive {{ stroke: #777; stroke-width: 1.1; stroke-dasharray: 2 2; fill: none; }}
      .pl {{ font-weight: bold; font-size: 13px; }} .pt {{ font-size: 12px; }}
      .lab {{ font-size: 12px; }} .s {{ font-size: 11px; }} .it {{ font-style: italic; }}
      .sub {{ font-size: 8px; }} .h {{ font-weight: bold; }}
    </style>
  </defs>
{A}
  <line x1="470" y1="34" x2="470" y2="{H - 10}" stroke="#ccc" stroke-width="1"/>
{Bp}
</svg>
'''
(HERE / "fig_concepts.svg").write_text(svg)
print("wrote fig_concepts.svg")
