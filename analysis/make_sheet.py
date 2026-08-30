"""Generate the verification contact sheet (HTML artifact) for one experiment day.

One row per series: representative photo, tally crop, label chips, timepoint
coverage strip, kymograph. Groups by condition. Embeds images as data URIs.
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

S = Path("/private/tmp/claude-501/-Users-steven-NP-Experiments/d38dc98f-cc81-4670-b3a2-b557500370b1/scratchpad")
OUT = S / "verification_sheet.html"
TIMEPOINTS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 21.5]


def b64(path, width):
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    s = width / w
    img = cv2.resize(img, (width, int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def coverage_strip(present, multi):
    cells = []
    for t in TIMEPOINTS:
        label = f"{t:g}"
        if t in multi:
            cls, txt = "multi", f"{label}&times;{multi[t]}"
        elif t in present:
            cls, txt = "ok", label
        else:
            cls, txt = "miss", label
        cells.append(f'<span class="tp {cls}">{txt}</span>')
    return "".join(cells)


def chip(text, cls=""):
    return f'<span class="chip {cls}">{text}</span>'


def main():
    photos = pd.read_csv("photos_final.csv", dtype={"agarose_f": str})
    photos["capture_time"] = pd.to_datetime(photos["capture_time"])
    n_photos = len(photos)
    n_series = photos.series.nunique()
    n_lowconf = int((photos.v_conf == "low").sum())

    groups_html = []
    issues = []
    cond_cols = ["agarose_f", "bsa_f", "coating_f"]
    for cond, cg in photos.groupby(cond_cols):
        ag, bsa, coat = cond
        rows_html = []
        for (arm, rep), g in cg.groupby(["arm", "tally_f"]):
            g = g.sort_values("capture_time")
            sname = f"{ag}_{bsa}_{coat}_{arm}_r{rep}"
            present = sorted(g.timepoint_hr.unique())
            counts = g.groupby("timepoint_hr").size()
            expected2 = {0.0} if arm == "magnet" else set()
            multi = {
                t: c
                for t, c in counts.items()
                if c > (2 if t in expected2 else 1)
            }
            missing = [t for t in TIMEPOINTS if t not in present]
            if missing:
                issues.append(f"{sname}: missing {', '.join(f'{t:g}h' for t in missing)}")
            if multi:
                issues.append(f"{sname}: extra photos at {', '.join(f'{t:g}h' for t in multi)}")
            # representative: latest timepoint with sticker likely visible
            rep_row = g.iloc[len(g) // 2]
            frame_b64 = b64(S / "prep" / "frames" / f"{int(rep_row.idx):04d}.jpg", 300)
            tally_b64 = b64(S / "prep" / "tally" / f"{int(rep_row.idx):04d}.jpg", 150)
            fig = S / "figs" / f"{sname}.png"
            fig_b64 = b64(fig, 500) if fig.exists() else ""
            armcls = "control" if arm == "control" else "magnet"
            flags = ""
            if (g.v_conf == "low").any():
                flags += chip(f"{int((g.v_conf=='low').sum())} low-conf reads", "warn")
            if g.get("tally_changed", pd.Series(False, index=g.index)).any():
                flags += chip("tally corrected", "warn")
            rows_html.append(f"""
<div class="series">
  <div class="thumbs">
    <img class="photo" src="{frame_b64}" alt="representative photo {sname}">
    <img class="tally" src="{tally_b64}" alt="tally crop">
  </div>
  <div class="meta">
    <div class="sname">{sname}</div>
    <div class="chips">{chip(arm.upper(), armcls)}{chip(f"repeat {rep}")}{chip(f"{len(g)} photos")}{flags}</div>
    <div class="strip">{coverage_strip(present, multi)}</div>
  </div>
  <div class="kymo">{f'<img src="{fig_b64}" alt="kymograph {sname}">' if fig_b64 else '<span class="nofig">no measurement</span>'}</div>
</div>""")
        groups_html.append(f"""
<section>
  <h2>{ag}% agarose &middot; {bsa} &middot; {coat}</h2>
  {''.join(rows_html)}
</section>""")

    issues_html = "".join(f"<li>{i}</li>" for i in issues) or "<li>none</li>"
    html = f"""<title>Aug 27 Series Audit</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#FAFBFC; --card:#FFFFFF; --ink:#1B2430; --muted:#5B6572;
  --line:#E3E7EC; --accent:#1E7A5A; --accent-soft:#E4F2EC;
  --warn:#B45309; --warn-soft:#FCF0E1; --miss:#B3423A; --miss-soft:#F9E9E7;
  --ok-soft:#EDF3F0;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#12161C; --card:#1A2029; --ink:#E8ECF1; --muted:#98A2AE;
    --line:#2A323D; --accent:#4CC094; --accent-soft:#1D3A2F;
    --warn:#E5A05A; --warn-soft:#3A2B18; --miss:#E07B72; --miss-soft:#3C2220;
    --ok-soft:#212B27;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#12161C; --card:#1A2029; --ink:#E8ECF1; --muted:#98A2AE;
  --line:#2A323D; --accent:#4CC094; --accent-soft:#1D3A2F;
  --warn:#E5A05A; --warn-soft:#3A2B18; --miss:#E07B72; --miss-soft:#3C2220;
  --ok-soft:#212B27;
}}
body {{ background:var(--paper); color:var(--ink); font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif;
  margin:0; padding:32px 24px 80px; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:26px; font-weight:600; margin:0 0 4px; text-wrap:balance; }}
.sub {{ color:var(--muted); margin:0 0 24px; }}
.stats {{ display:flex; gap:24px; flex-wrap:wrap; margin-bottom:8px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:10px 18px; }}
.stat b {{ display:block; font:500 22px/1.2 "IBM Plex Mono",monospace; color:var(--accent); }}
.stat span {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
details.issues {{ margin:16px 0 8px; background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px 18px; }}
details.issues summary {{ cursor:pointer; font-weight:600; color:var(--warn); }}
details.issues li {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; margin:4px 0; }}
h2 {{ font-size:15px; font-weight:600; letter-spacing:.04em; text-transform:uppercase;
  color:var(--accent); border-bottom:2px solid var(--accent); padding-bottom:6px; margin:40px 0 12px; }}
.series {{ display:grid; grid-template-columns:340px 1fr 500px; gap:18px; align-items:center;
  background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin-bottom:10px; }}
.thumbs {{ display:flex; gap:8px; align-items:center; }}
.thumbs img {{ border-radius:4px; display:block; }}
.photo {{ width:210px; }}
.tally {{ width:110px; }}
.sname {{ font:500 13.5px "IBM Plex Mono",monospace; margin-bottom:6px; }}
.chips {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }}
.chip {{ font:500 11px "IBM Plex Mono",monospace; padding:2px 8px; border-radius:99px;
  background:var(--ok-soft); border:1px solid var(--line); }}
.chip.magnet {{ background:var(--accent-soft); color:var(--accent); border-color:transparent; }}
.chip.control {{ background:var(--ok-soft); color:var(--muted); }}
.chip.warn {{ background:var(--warn-soft); color:var(--warn); border-color:transparent; }}
.strip {{ display:flex; gap:3px; flex-wrap:wrap; }}
.tp {{ font:500 10.5px "IBM Plex Mono",monospace; min-width:30px; text-align:center;
  padding:3px 4px; border-radius:3px; }}
.tp.ok {{ background:var(--ok-soft); color:var(--ink); }}
.tp.miss {{ background:var(--miss-soft); color:var(--miss); text-decoration:line-through; }}
.tp.multi {{ background:var(--warn-soft); color:var(--warn); }}
.kymo img {{ width:100%; border-radius:4px; }}
.nofig {{ color:var(--muted); font-size:12px; }}
@media (max-width:1000px) {{ .series {{ grid-template-columns:1fr; }} .kymo img {{ max-width:520px; }} }}
</style>
<main>
<h1>Aug 27 Series Audit</h1>
<p class="sub">21.5 h back-injection run &middot; automated sorting from stickers, tally marks and EXIF times &middot; check each row's labels against its photo</p>
<div class="stats">
  <div class="stat"><b>{n_photos}</b><span>unique photos</span></div>
  <div class="stat"><b>{n_series}</b><span>series</span></div>
  <div class="stat"><b>{n_lowconf}</b><span>low-conf reads</span></div>
  <div class="stat"><b>{len(issues)}</b><span>coverage issues</span></div>
</div>
<details class="issues"><summary>Coverage issues ({len(issues)})</summary><ul>{issues_html}</ul></details>
{''.join(groups_html)}
</main>
"""
    OUT.write_text(html)
    print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
