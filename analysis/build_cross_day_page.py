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

HTML = f"""<title>0.4% vs 0.6% Agarose</title>
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
  max-width:640px; }}
.good {{ color:var(--accent); }}
.warn {{ color:var(--caution); }}
tr.grp2 td {{ padding-top:16px; border-top:1px solid var(--ink2); color:var(--ink2); }}
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
  <p class="eyebrow">Nanoparticle transport · agarose concentration</p>
  <h1>Does stiffer gel slow the particles down?</h1>
  <p class="stand">The question can be asked twice — once in the centre-injection runs and once in
  the back-injection run. The two geometries give different answers, and the reason they differ is
  the most useful thing on this page.</p>
</header>

<figure>
  <img src="data:image/png;base64,{img}" alt="Asymmetry versus time for 0.4% and 0.6% agarose in centre injection, and front position versus time in back injection">
  <figcaption><strong>Top — centre injection (Aug 23 vs Aug 26).</strong> Asymmetry: mean
  nanoparticle darkness on the magnet side of the gap minus the far side. Bold lines are the mean of
  three repeats, faint lines the individual repeats, grey the envelope of all four no-magnet
  controls. <strong>Bottom — back injection (Aug 27).</strong> Front position: the deepest point
  still carrying 20% of peak darkness, each magnet series minus the mean control of its own
  BSA×coating cell. Bands are ±1 SEM across 12 series. Red is 0.4% and blue 0.6% in both panels.</figcaption>
</figure>

<section class="col">
  <h2>Two ways to ask it, and they are not equally good</h2>
  <p>The back-injection comparison is the better experiment by design — both concentrations ran in
  the <em>same session</em>, balanced across BSA and coating, so there is no between-day confound and
  four times the series. The centre-injection comparison is assembled from two different days.</p>
  <div class="matched three">
    <div class="h"></div><div class="h">Centre injection</div><div class="h">Back injection</div>
    <div class="k">Source</div><div class="v">Aug 23 vs Aug 26</div><div class="v good">Aug 27, one day</div>
    <div class="k">Confounding</div><div class="v warn">gel batch, lighting</div><div class="v good">none — same session</div>
    <div class="k">Series per concentration</div><div class="v warn">3</div><div class="v good">12</div>
    <div class="k">Metric</div><div class="v">asymmetry (L*)</div><div class="v">front position (mm)</div>
    <div class="k">Transport observed</div><div class="v good">13–42 L*, plateaus by 3 h</div><div class="v warn">≤4 mm over 21.5 h</div>
    <div class="k"><strong>Result</strong></div><div class="v">0.4% &gt; 0.6%, n.s.</div><div class="v">no difference</div>
  </div>
</section>

<section>
  <h2 class="col">Centre injection — a consistent ordering that misses significance</h2>
  <p class="col">Restricting Aug 26 to its non-BSA arms leaves agarose concentration as the only
  difference from Aug 23: both are centre injection, 6 h, 13 timepoints at 30 min, PEG and COOH,
  large magnet, identical warp geometry. Softer gel transports further in both coatings, with no
  crossing after 1 h.</p>
  <div class="tablewrap">
  <table>
    <caption>Welch t-tests on per-series values, n=3 per group. Positive means the first term
    transported further.</caption>
    <thead><tr><th>Contrast</th><th>Direction</th><th class="num">Difference</th><th class="num">p</th></tr></thead>
    <tbody>
{table()}
    </tbody>
  </table>
  </div>
</section>

<section>
  <h2 class="col">Back injection — better powered, and it finds nothing</h2>
  <p class="col">Front position was chosen over a mass-weighted depth before looking at the
  comparison, because back injection leaves a large reservoir pile against the gap: a magnet arm that
  has drawn particles out of the pile into a long tail scores <em>lower</em> on a centroid than a
  control that kept everything heaped at the boundary. As a check that it measures the right thing,
  the front metric independently recovers this run's known magnet effect, <span class="mono">+0.41 mm</span>
  against a published <span class="mono">+0.44 mm</span>.</p>
  <div class="tablewrap">
  <table>
    <caption>Magnet series averaged over 12–21.5 h, each minus the mean control of its own
    BSA×coating cell. n=12 series per concentration.</caption>
    <thead><tr><th>Quantity</th><th>0.4%</th><th>0.6%</th><th class="num">Difference</th><th class="num">p</th></tr></thead>
    <tbody>
      <tr><td>Front, magnet − control</td><td class="mono">+0.22 mm</td><td class="mono">+0.59 mm</td><td class="mono num">−0.37</td><td class="mono num">0.61</td></tr>
      <tr><td>Front, absolute</td><td class="mono">3.66 mm</td><td class="mono">3.78 mm</td><td class="mono num">−0.11</td><td class="mono num">0.76</td></tr>
      <tr><td>Centroid, magnet − control</td><td class="mono">−0.36 mm</td><td class="mono">+0.19 mm</td><td class="mono num">−0.55</td><td class="mono num">0.17</td></tr>
      <tr class="grp2"><td>Magnet effect itself, both concentrations</td><td class="mono" colspan="2">+0.41 mm (n=24)</td><td class="mono num">—</td><td class="mono num">0.26</td></tr>
    </tbody>
  </table>
  </div>
</section>

<section class="col rail">
  <h2>Why the better experiment is the less informative one</h2>
  <p>The last row of that table is the explanation. On Aug 27 the magnet effect itself is only
  +0.41 mm and does not clear significance — the bulk of the particles never left the first few
  millimetres in 21.5 hours. Asking whether gel stiffness <em>modulates</em> that effect means
  looking for a difference in something that is barely present.</p>
  <p>The centre-injection runs have room to show it: asymmetry climbs to 13–42 L* and plateaus
  within three hours, an order of magnitude more dynamic range. So the honest reading is not that
  the two geometries disagree about gel stiffness. It is that only one of them moved enough
  particles for the question to be answerable, and that one is also the one with n=3 and a
  between-day confound.</p>
</section>

<section class="split">
  <div class="card">
    <h3>Defensible in a write-up</h3>
    <ul>
      <li>Every centre-injection magnet arm separates cleanly from the control envelope, which stays
      within ±4.4 L* for six hours on both days.</li>
      <li>Softer gel transports further in both coatings at every timepoint after 1 h — a consistent
      ordering, reported as such.</li>
      <li>Back injection shows no stiffness difference, in the run that had the power to detect one
      had the transport been there.</li>
    </ul>
  </div>
  <div class="card no">
    <h3>Needs the caveat attached</h3>
    <ul>
      <li>No contrast on this page reaches p&lt;0.05. The closest is PEG 0.4% vs 0.6% at
      <span class="mono">p=0.054</span>.</li>
      <li>Repeat scatter is the limit: one 0.6% COOH repeat plateaus at 12.8 L*, another at 48.1.</li>
      <li>The apparent coating×stiffness interaction — coating irrelevant at 0.4%, decisive at 0.6%
      — is the weakest claim here at <span class="mono">p=0.22</span>. An interaction needs more
      evidence than a main effect, not less.</li>
    </ul>
  </div>
</section>

<footer class="col">
  <p>Computed by <span class="mono">cross_day_compare.py</span> and
  <span class="mono">back_depth_compare.py</span> from 405 centre-injection and 572 back-injection
  frames. Per-day heatmaps are unchanged and remain the primary record:
  <a href="https://claude.ai/code/artifact/bfbbd924-d55b-411c-b54a-6c53b0541676">Aug&nbsp;23</a>,
  <a href="https://claude.ai/code/artifact/13f18308-9059-4fb3-a44f-672c425ffd74">Aug&nbsp;26</a>,
  <a href="https://claude.ai/code/artifact/b3891b33-43e9-4880-b0cf-80f59790fb3e">Aug&nbsp;27</a>.</p>
</footer>

</main>
"""
(OUT / "cross_day_comparison.html").write_text(HTML)
print("wrote outputs/cross_day_comparison.html", len(HTML))
