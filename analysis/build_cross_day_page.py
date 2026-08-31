"""Render the cross-day comparison page (outputs/cross_day_comparison.html)."""
import base64
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
img = base64.b64encode((OUT / "cross_day_comparison.png").read_bytes()).decode()

ROWS = [
    ("Plateau, mean of 5–6 h (L*)", [
        ("coating, within 0.4% gel", "COOH − PEG", "+6.4", "0.39", 0),
        ("coating, within 0.6% gel", "COOH − PEG", "+17.9", "0.22", 0),
        ("gel stiffness, COOH", "0.4% − 0.6%", "+9.1", "0.48", 0),
        ("gel stiffness, PEG", "0.4% − 0.6%", "+20.5", "0.054", 1),
    ]),
    ("Early rate, 0–3 h slope (L*/h)", [
        ("coating, within 0.4% gel", "COOH − PEG", "+0.4", "0.89", 0),
        ("coating, within 0.6% gel", "COOH − PEG", "+4.0", "0.26", 0),
        ("gel stiffness, COOH", "0.4% − 0.6%", "+2.9", "0.33", 0),
        ("gel stiffness, PEG", "0.4% − 0.6%", "+6.5", "0.14", 0),
    ]),
]

def table():
    out = []
    for head, rows in ROWS:
        out.append(f'<tr class="grp"><th colspan="4">{head}</th></tr>')
        for what, contrast, diff, p, near in rows:
            cls = ' class="near"' if near else ""
            out.append(f'<tr><td>{what}</td><td class="mono dim">{contrast}</td>'
                       f'<td class="mono num">{diff}</td><td class="mono num"{cls}>{p}</td></tr>')
    return "\n".join(out)

HTML = f"""<title>Gel Stiffness Across Run Days</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,500;0,600;1,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --page:#F6F5F2; --surface:#FDFDFC; --sunk:#F0EFEB;
  --ink:#1A1A22; --ink2:#5C5C6A; --rule:rgba(26,26,34,.13);
  --accent:#4A3F8C; --caution:#8A4E14;
  --soft:#C0392B; --stiff:#1F6FB2;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --page:#121218; --surface:#1C1C24; --sunk:#191920;
    --ink:#E9E8EE; --ink2:#A29FB0; --rule:rgba(233,232,238,.15);
    --accent:#A497EC; --caution:#D79A5E;
    --soft:#E9705F; --stiff:#5FA8E0;
  }}
}}
:root[data-theme="dark"] {{
  --page:#121218; --surface:#1C1C24; --sunk:#191920;
  --ink:#E9E8EE; --ink2:#A29FB0; --rule:rgba(233,232,238,.15);
  --accent:#A497EC; --caution:#D79A5E;
  --soft:#E9705F; --stiff:#5FA8E0;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--page); color:var(--ink); margin:0;
  padding:clamp(28px,5vw,60px) 20px 84px;
  font:16px/1.62 "IBM Plex Sans", system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing:antialiased; }}
main {{ max-width:1120px; margin:0 auto; display:flex; flex-direction:column; gap:34px; }}
.col {{ max-width:66ch; }}
.eyebrow {{ font-size:11.5px; font-weight:600; letter-spacing:.13em; text-transform:uppercase;
  color:var(--accent); margin:0 0 12px; }}
h1 {{ font-family:Spectral, Georgia, serif; font-weight:600; font-size:clamp(30px,4.4vw,44px);
  line-height:1.14; letter-spacing:-.01em; margin:0 0 14px; text-wrap:balance; }}
.stand {{ font-family:Spectral, Georgia, serif; font-size:19.5px; line-height:1.52; font-style:italic;
  color:var(--ink2); margin:0; text-wrap:balance; }}
h2 {{ font-family:Spectral, Georgia, serif; font-weight:600; font-size:23px; letter-spacing:-.005em;
  margin:0 0 10px; text-wrap:balance; }}
p {{ margin:0 0 14px; }}
p:last-child {{ margin-bottom:0; }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--rule); border-radius:5px;
  padding:14px; }}
figure img {{ width:100%; display:block; }}
figcaption {{ font-size:13.5px; color:var(--ink2); line-height:1.5; margin-top:13px;
  padding-top:12px; border-top:1px solid var(--rule); }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; min-width:520px; font-size:14.5px; }}
caption {{ text-align:left; font-size:13.5px; color:var(--ink2); padding-bottom:10px; }}
th, td {{ padding:8px 16px 8px 0; border-bottom:1px solid var(--rule); text-align:left;
  vertical-align:baseline; }}
tr.grp th {{ font-family:"IBM Plex Sans"; font-size:11.5px; font-weight:600; letter-spacing:.1em;
  text-transform:uppercase; color:var(--accent); padding-top:20px; border-bottom-color:var(--ink2); }}
thead th {{ font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase;
  color:var(--ink2); }}
.mono {{ font-family:"IBM Plex Mono", ui-monospace, monospace; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; padding-right:0; }}
.dim {{ color:var(--ink2); font-size:13px; }}
.near {{ color:var(--caution); font-weight:500; }}
.split {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:26px; }}
.card {{ background:var(--surface); border:1px solid var(--rule); border-radius:5px; padding:20px 22px; }}
.card h3 {{ font-size:12px; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  margin:0 0 12px; color:var(--accent); }}
.card.no h3 {{ color:var(--caution); }}
.card ul {{ margin:0; padding-left:18px; display:flex; flex-direction:column; gap:9px; font-size:15px; }}
.rail {{ border-left:2px solid var(--accent); padding:2px 0 2px 20px; }}
.matched {{ display:grid; grid-template-columns:auto 1fr 1fr; gap:0 22px; font-size:14.5px;
  max-width:560px; }}
.matched div {{ padding:7px 0; border-bottom:1px solid var(--rule); }}
.matched .h {{ font-size:11.5px; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink2); }}
.matched .k {{ color:var(--ink2); }}
.matched .v {{ font-family:"IBM Plex Mono", monospace; }}
.diff {{ color:var(--soft); font-weight:500; }}
.diff2 {{ color:var(--stiff); font-weight:500; }}
footer {{ border-top:1px solid var(--rule); padding-top:22px; font-size:14px; color:var(--ink2); }}
footer a {{ color:var(--accent); text-decoration:none; border-bottom:1px solid var(--rule); }}
footer a:hover, footer a:focus-visible {{ border-bottom-color:var(--accent); }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:3px; }}
</style>
<main>

<header class="col">
  <p class="eyebrow">Nanoparticle transport · cross-day synthesis</p>
  <h1>What the 0.4% and 0.6% runs say when you put them on one axis</h1>
  <p class="stand">Aug 23 and Aug 26 differ in exactly one thing that matters. Read together they
  give a gel-stiffness contrast that neither day can show alone — and a clear view of how far
  three repeats will actually carry it.</p>
</header>

<figure>
  <img src="data:image/png;base64,{img}" alt="Asymmetry versus time for 0.4% and 0.6% agarose, with an Aug 27 penetration-depth panel below">
  <figcaption><strong>Top:</strong> asymmetry — mean nanoparticle darkness on the magnet side of
  the injection gap minus the far side. Bold lines are the mean of three repeats; faint lines are the
  individual repeats; the grey band is the envelope of all four no-magnet controls from both days.
  <strong>Bottom:</strong> Aug 27 is back injection, so it has no far side and no asymmetry — it
  appears as magnet-minus-control penetration depth, a different quantity on a different time span,
  plotted for context only.</figcaption>
</figure>

<section class="col">
  <h2>Why these two days can share an axis</h2>
  <p>Aug 23 and Aug 26 match on everything the measurement depends on. Lining up Aug 23's
  large-magnet arms against Aug 26's <em>non-BSA</em> large-magnet arms leaves agarose concentration
  as the only difference.</p>
  <div class="matched">
    <div class="h"></div><div class="h">Aug 23</div><div class="h">Aug 26</div>
    <div class="k">Injection</div><div class="v">centre gap</div><div class="v">centre gap</div>
    <div class="k">Duration · cadence</div><div class="v">6 h · 13 × 30 min</div><div class="v">6 h · 13 × 30 min</div>
    <div class="k">Coatings</div><div class="v">PEG, COOH</div><div class="v">PEG, COOH</div>
    <div class="k">Magnet</div><div class="v">large</div><div class="v">large</div>
    <div class="k">Serum albumin</div><div class="v">non-BSA</div><div class="v">non-BSA arms only</div>
    <div class="k"><strong>Agarose</strong></div><div class="v diff">0.4%</div><div class="v diff2">0.6%</div>
  </div>
  <p style="margin-top:16px">Both days were recomputed here from a single definition. The per-day
  metric files came from separate ad-hoc scripts and could not be assumed mutually consistent, which
  is the one property a cross-day plot depends on.</p>
</section>

<section>
  <h2 class="col">The contrasts, with their p-values</h2>
  <div class="tablewrap">
  <table>
    <caption>Welch t-tests on per-series values, n=3 per group. Positive means the first
    term transported further.</caption>
    <thead><tr><th>Contrast</th><th>Direction</th><th class="num">Difference</th><th class="num">p</th></tr></thead>
    <tbody>
{table()}
    </tbody>
  </table>
  </div>
</section>

<section class="split">
  <div class="card">
    <h3>What this establishes</h3>
    <ul>
      <li>Every magnet arm separates cleanly from the control envelope, which stays within
      ±4.4 L* for all six hours on both days. The transport itself is not in question.</li>
      <li>Softer gel transports further in both coatings, and the ordering is the same at every
      timepoint after 1 h — no crossing, no reversal.</li>
      <li>The two days agree on shape: a rise over roughly three hours, then a plateau. Stiffer gel
      shifts the plateau down rather than delaying it.</li>
    </ul>
  </div>
  <div class="card no">
    <h3>What it does not</h3>
    <ul>
      <li>None of the four contrasts reaches significance. PEG 0.4% vs 0.6% comes closest at
      <span class="mono">p=0.054</span>; the rest sit between 0.14 and 0.89.</li>
      <li>Repeat scatter is the limit, not measurement noise. One 0.6% COOH repeat plateaus at
      12.8 L* and another at 48.1 — visible as the spread of faint blue lines.</li>
      <li>This is a between-day comparison: different gel batch, different session lighting. The
      stiffness difference is real, but it is not the only thing that differs between the two days.</li>
    </ul>
  </div>
</section>

<section class="col rail">
  <h2>The apparent interaction is the weakest claim here</h2>
  <p>Coating looks irrelevant at 0.4% (COOH and PEG overlap almost exactly) and decisive at 0.6%
  (COOH 32.4 vs PEG 14.4 L*). A tidy mechanism suggests itself — in open gel both coatings
  move freely; tighten the pores and the PEG surface is hindered where COOH is not.</p>
  <p>That story is not supported yet. The within-0.6% coating contrast is <span class="mono">p=0.22</span>,
  and an interaction needs more evidence than either main effect, not less. Treat it as the thing
  worth designing the next run around, not as a result.</p>
</section>

<footer class="col">
  <p>Computed by <span class="mono">cross_day_compare.py</span> from
  <span class="mono">cross_day_asymmetry.csv</span> (405 frames). Per-day heatmaps remain unchanged
  and are the primary record:
  <a href="https://claude.ai/code/artifact/bfbbd924-d55b-411c-b54a-6c53b0541676">Aug&nbsp;23</a>,
  <a href="https://claude.ai/code/artifact/13f18308-9059-4fb3-a44f-672c425ffd74">Aug&nbsp;26</a>,
  <a href="https://claude.ai/code/artifact/b3891b33-43e9-4880-b0cf-80f59790fb3e">Aug&nbsp;27</a>.</p>
</footer>

</main>
"""
(OUT / "cross_day_comparison.html").write_text(HTML)
print("wrote outputs/cross_day_comparison.html", len(HTML))
