"""Render the agarose-concentration comparison page (outputs/cross_day_comparison.html)."""
import base64
from pathlib import Path

OUT = Path(__file__).parent / "outputs"
img = base64.b64encode((OUT / "cross_day_comparison.png").read_bytes()).decode()

# 2x2 factorial, non-BSA large-magnet arms, n=12 series (cross_day_compare.py)
FACTORIAL = [
    ("Agarose 0.4% vs 0.6%", "+14.8 L*", "5.64", "0.045", 2),
    ("Coating COOH vs PEG", "+12.2 L*", "3.82", "0.086", 1),
    ("Coating &times; agarose interaction", "&mdash;", "0.85", "0.384", 0),
]

# plateau (5-6 h mean) per condition cell
CELLS = [
    ("Aug 23 &middot; 0.4% &middot; non-BSA &middot; large", "41.4", "35.0", "+6.4", 1),
    ("Aug 26 &middot; 0.6% &middot; BSA &middot; large", "28.8", "22.7", "+6.1", 1),
    ("Aug 26 &middot; 0.6% &middot; non-BSA &middot; large", "32.4", "14.4", "+17.9", 1),
    ("Aug 23 &middot; 0.4% &middot; non-BSA &middot; small", "7.8", "11.8", "&minus;4.0", 0),
]

BACK = [
    ("Front, magnet &minus; control", "+0.22 mm", "+0.59 mm", "&minus;0.37", "0.61"),
    ("Front, absolute", "3.66 mm", "3.78 mm", "&minus;0.11", "0.76"),
    ("Centroid, magnet &minus; control", "&minus;0.36 mm", "+0.19 mm", "&minus;0.55", "0.17"),
]


def factorial_rows():
    out = []
    for what, eff, F, p, strength in FACTORIAL:
        cls = ' class="sig"' if strength == 2 else (' class="near"' if strength == 1 else "")
        out.append(f'<tr><td>{what}</td><td class="mono num">{eff}</td>'
                   f'<td class="mono num dim">{F}</td><td class="mono num"{cls}>{p}</td></tr>')
    return "\n".join(out)


def cell_rows():
    out = []
    for cell, cooh, peg, diff, ok in CELLS:
        cls = "" if ok else ' class="warn"'
        out.append(f'<tr><td>{cell}</td><td class="mono num">{cooh}</td>'
                   f'<td class="mono num">{peg}</td><td class="mono num"{cls}>{diff}</td>'
                   f'<td{cls}>{"COOH ahead" if ok else "PEG ahead"}</td></tr>')
    return "\n".join(out)


def back_rows():
    return "\n".join(
        f'<tr><td>{q}</td><td class="mono num">{a}</td><td class="mono num">{b}</td>'
        f'<td class="mono num">{d}</td><td class="mono num">{p}</td></tr>'
        for q, a, b, d, p in BACK)


HTML = f"""<title>0.4% vs 0.6% Agarose</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,500;0,600;1,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --page:#F6F5F2; --surface:#FDFDFC;
  --ink:#1A1A22; --ink2:#5C5C6A; --rule:rgba(26,26,34,.13);
  --accent:#4A3F8C; --caution:#8A4E14;
  --soft:#C0392B; --stiff:#1F6FB2;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --page:#121218; --surface:#1C1C24;
    --ink:#E9E8EE; --ink2:#A29FB0; --rule:rgba(233,232,238,.15);
    --accent:#A497EC; --caution:#D79A5E;
    --soft:#E9705F; --stiff:#5FA8E0;
  }}
}}
:root[data-theme="dark"] {{
  --page:#121218; --surface:#1C1C24;
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
table {{ border-collapse:collapse; width:100%; min-width:540px; font-size:14.5px; }}
caption {{ text-align:left; font-size:13.5px; color:var(--ink2); padding-bottom:10px; }}
th, td {{ padding:8px 16px 8px 0; border-bottom:1px solid var(--rule); text-align:left;
  vertical-align:baseline; }}
thead th {{ font-size:12px; font-weight:600; letter-spacing:.05em; text-transform:uppercase;
  color:var(--ink2); }}
.mono {{ font-family:"IBM Plex Mono", ui-monospace, monospace; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.dim {{ color:var(--ink2); }}
.sig {{ color:var(--accent); font-weight:600; }}
.near {{ color:var(--caution); font-weight:500; }}
.warn {{ color:var(--caution); }}
.good {{ color:var(--accent); }}
.split {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:26px; }}
.card {{ background:var(--surface); border:1px solid var(--rule); border-radius:5px; padding:20px 22px; }}
.card h3 {{ font-size:12px; font-weight:600; letter-spacing:.1em; text-transform:uppercase;
  margin:0 0 12px; color:var(--accent); }}
.card.no h3 {{ color:var(--caution); }}
.card ul {{ margin:0; padding-left:18px; display:flex; flex-direction:column; gap:9px; font-size:15px; }}
.rail {{ border-left:2px solid var(--accent); padding:2px 0 2px 20px; }}
.matched {{ display:grid; grid-template-columns:auto 1fr 1fr; gap:0 22px; font-size:14.5px;
  max-width:660px; }}
.matched div {{ padding:7px 0; border-bottom:1px solid var(--rule); }}
.matched .h {{ font-size:11.5px; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
  color:var(--ink2); }}
.matched .k {{ color:var(--ink2); }}
.matched .v {{ font-family:"IBM Plex Mono", monospace; font-size:13.5px; }}
footer {{ border-top:1px solid var(--rule); padding-top:22px; font-size:14px; color:var(--ink2); }}
footer a {{ color:var(--accent); text-decoration:none; border-bottom:1px solid var(--rule); }}
footer a:hover, footer a:focus-visible {{ border-bottom-color:var(--accent); }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:3px; }}
</style>
<main>

<header class="col">
  <p class="eyebrow">Nanoparticle transport &middot; agarose concentration</p>
  <h1>Does stiffer gel slow the particles down?</h1>
  <p class="stand">Yes in the centre-injection runs, where the effect reaches significance once the
  design is analysed as the 2&times;2 it is. The back-injection run cannot answer the question &mdash;
  and the reason it cannot is worth as much as the answer.</p>
</header>

<figure>
  <img src="data:image/png;base64,{img}" alt="Asymmetry versus time for 0.4% and 0.6% agarose in centre injection, and front position versus time in back injection">
  <figcaption><strong>Top &mdash; centre injection (Aug 23 vs Aug 26).</strong> Asymmetry: mean
  nanoparticle darkness on the magnet side of the gap minus the far side. Bold lines are the mean of
  three repeats, faint lines the individual repeats, grey the envelope of all four no-magnet
  controls. <strong>Bottom &mdash; back injection (Aug 27).</strong> Front position: the deepest
  point still carrying 20% of peak darkness, each magnet series minus the mean control of its own
  BSA&times;coating cell, &plusmn;1 SEM across 12 series. Red is 0.4% and blue 0.6% in both panels.</figcaption>
</figure>

<section>
  <h2 class="col">Centre injection, analysed as a factorial rather than as pairwise tests</h2>
  <p class="col">Restricting Aug 26 to its non-BSA arms leaves agarose concentration as the only
  difference from Aug 23: both centre injection, 6 h, 13 timepoints at 30 min, PEG and COOH, large
  magnet, identical warp geometry. Four cells of three repeats is a 2&times;2 design, so it is tested
  as one. That pools the residual variance and gives 8 degrees of freedom instead of the 2&ndash;4
  that separate pairwise t-tests were working with &mdash; which is why the stiffness effect surfaces
  here and did not before.</p>
  <div class="tablewrap">
  <table>
    <caption>Two-way ANOVA on the 5&ndash;6 h plateau of each series. Non-BSA, large magnet,
    n=12 series.</caption>
    <thead><tr><th>Term</th><th class="num">Effect</th><th class="num">F</th><th class="num">p</th></tr></thead>
    <tbody>
{factorial_rows()}
    </tbody>
  </table>
  </div>
  <p class="col" style="margin-top:18px">Within the 0.6% run on its own &mdash; Aug 26, one day, BSA
  as a blocking factor, n=12 &mdash; COOH leads PEG by <span class="mono">+12.0&nbsp;L*</span>,
  <span class="mono">F=4.44, p=0.064</span>, 95% CI <span class="mono">[&minus;0.9, +24.9]</span>.
  The BSA block itself is inert (<span class="mono">p=0.69</span>), which is what justifies pooling
  the two albumin states.</p>
</section>

<section>
  <h2 class="col">Does COOH beat PEG in every cell? Three times out of four</h2>
  <div class="tablewrap">
  <table>
    <caption>Plateau asymmetry (5&ndash;6 h mean, L*) per condition cell, n=3 series each.</caption>
    <thead><tr><th>Cell</th><th class="num">COOH</th><th class="num">PEG</th>
      <th class="num">COOH &minus; PEG</th><th>Direction</th></tr></thead>
    <tbody>
{cell_rows()}
    </tbody>
  </table>
  </div>
  <p class="col" style="margin-top:14px">The exception is the small-magnet arm, and it is the cell
  with almost no transport to compare &mdash; 7.8 and 11.8&nbsp;L* against 22.7 to 41.4 everywhere
  else. That is the same floor problem that makes the back-injection run uninformative, so it is weak
  evidence rather than a clean contradiction. A sign test over four cells has no power to settle it
  either way (<span class="mono">p=0.31</span>).</p>
</section>

<section class="col rail">
  <h2>What survives, and what does not</h2>
  <p><strong>The gel-stiffness effect does reach significance</strong> once the design is analysed as
  the factorial it is: <span class="mono">p=0.045</span>. Softer gel transported further in both
  coatings, at every timepoint after 1 h, with no crossing. But <em>agarose</em> and <em>day</em> are
  the same variable in this comparison &mdash; 0.4% is Aug 23, 0.6% is Aug 26 &mdash; so the p-value
  does not rule out a gel batch or session-lighting difference. No amount of analysis fixes that.</p>
  <p><strong>The coating effect is marginal and directionally consistent:</strong>
  <span class="mono">p=0.086</span> across the factorial, <span class="mono">p=0.064</span> within the
  0.6% run alone, COOH ahead in three of the four cells.</p>
  <p><strong>The claim that PEG is hindered <em>more</em> at 0.6% is not supported.</strong> The
  coating gap does look wider there &mdash; +17.9&nbsp;L* against +6.4 &mdash; but that comparison is
  the interaction term, and it tests at <span class="mono">p=0.38</span>. Describing the pattern is
  fair; asserting that the two concentrations differ in how much coating matters is not.</p>
</section>

<section>
  <h2 class="col">Why the better-designed experiment answered nothing</h2>
  <p class="col">Aug 27 ran both concentrations in one session, balanced across BSA and coating, with
  12 magnet series per concentration instead of 3 &mdash; no between-day confound and four times the
  material. It still finds nothing, and the last row explains why.</p>
  <div class="tablewrap">
  <table>
    <caption>Back injection, magnet series averaged over 12&ndash;21.5 h, each minus the mean control
    of its own BSA&times;coating cell. n=12 series per concentration.</caption>
    <thead><tr><th>Quantity</th><th class="num">0.4%</th><th class="num">0.6%</th>
      <th class="num">Difference</th><th class="num">p</th></tr></thead>
    <tbody>
{back_rows()}
    <tr><td class="dim">Magnet effect itself, both concentrations</td>
      <td class="mono num dim" colspan="2">+0.41 mm (n=24)</td>
      <td class="mono num dim">&mdash;</td><td class="mono num warn">0.26</td></tr>
    </tbody>
  </table>
  </div>
  <p class="col" style="margin-top:16px">On Aug 27 the magnet effect itself is only +0.41&nbsp;mm and
  does not clear significance: the particles barely left the first few millimetres in 21.5 hours.
  Asking whether gel stiffness modulates that means looking for a difference in something that is
  barely present. Centre injection reaches 13&ndash;42&nbsp;L* and plateaus within three hours, an
  order of magnitude more dynamic range. The two geometries are not in conflict &mdash; only one of
  them moved enough particles for the question to be answerable.</p>
</section>

<section class="split">
  <div class="card">
    <h3>Defensible in a write-up</h3>
    <ul>
      <li>Softer gel transported further in both coatings &mdash; significant in the factorial
      (<span class="mono">p=0.045</span>), with the day confound stated alongside it.</li>
      <li>COOH transported further than PEG in three of four cells, and by
      <span class="mono">+12.0&nbsp;L*</span> within the 0.6% run
      (<span class="mono">p=0.064</span>).</li>
      <li>Every magnet arm separates cleanly from the control envelope, which stays within
      &plusmn;4.4&nbsp;L* for six hours across both days.</li>
    </ul>
  </div>
  <div class="card no">
    <h3>Needs the caveat attached</h3>
    <ul>
      <li>&ldquo;PEG is hindered more at 0.6%&rdquo; is an interaction, and it tests at
      <span class="mono">p=0.38</span>. Report the pattern, not the comparison.</li>
      <li>Agarose concentration and run day are perfectly confounded in this comparison.</li>
      <li>The small-magnet cell reverses the coating ordering &mdash; in the one cell where transport
      was near the floor.</li>
      <li>Repeat scatter is the binding limit: one 0.6% COOH repeat plateaus at 12.8&nbsp;L*, another
      at 48.1.</li>
    </ul>
  </div>
</section>

<footer class="col">
  <p>Computed by <span class="mono">cross_day_compare.py</span> and
  <span class="mono">back_depth_compare.py</span> from 405 centre-injection and 572 back-injection
  frames; per-series values in <span class="mono">centre_plateau_by_series.csv</span>. Per-day
  heatmaps are unchanged and remain the primary record:
  <a href="https://claude.ai/code/artifact/bfbbd924-d55b-411c-b54a-6c53b0541676">Aug&nbsp;23</a>,
  <a href="https://claude.ai/code/artifact/13f18308-9059-4fb3-a44f-672c425ffd74">Aug&nbsp;26</a>,
  <a href="https://claude.ai/code/artifact/b3891b33-43e9-4880-b0cf-80f59790fb3e">Aug&nbsp;27</a>.</p>
</footer>

</main>
"""
(OUT / "cross_day_comparison.html").write_text(HTML)
print("wrote outputs/cross_day_comparison.html", len(HTML))
