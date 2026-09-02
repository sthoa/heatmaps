"""Statistics for the centre-injection 0.4% vs 0.6% comparison.

Reads centre_plateau_by_series.csv (one plateau value per series, produced by
cross_day_compare.py) and prints every number quoted on the comparison and
methods pages, so the p-values are reproducible from a committed script rather
than from an interactive session.

  1. 2x2 factorial (agarose x coating) on the non-BSA large-magnet arms, n=12.
  2. Coating effect within the 0.6% run alone (Aug 26), BSA as a block, n=12.
  3. Cell-by-cell direction of COOH vs PEG.

Unit of replication is the SERIES (one physical sample), never the photograph.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

S = pd.read_csv(Path(__file__).parent / "centre_plateau_by_series.csv")

print("=== the 12 plateau values (5-6 h mean asymmetry, L*) ===")
nb = S[(S.bsa == "non-BSA") & (S.arm == "large")]
for (ag, co), g in nb.groupby(["agarose", "coating"]):
    print(f"  {ag} {co:5s} " + ", ".join(f"{v:.1f}" for v in g.plateau) + f"   mean {g.plateau.mean():.1f}")

print("\n=== 1. two-way ANOVA, non-BSA large arms (n=12) ===")
m = smf.ols("plateau ~ C(coating) * C(agarose)", data=nb).fit()
print(sm.stats.anova_lm(m, typ=2).round(4).to_string())
mar = nb.groupby("agarose").plateau.mean()
print(f"  agarose effect (0.4% - 0.6%): {mar['0.4%'] - mar['0.6%']:+.1f} L*")
mar = nb.groupby("coating").plateau.mean()
print(f"  coating effect (COOH - PEG):  {mar['COOH'] - mar['PEG']:+.1f} L*")

print("\n=== 2. coating within 0.6% (Aug 26 only), BSA as a block (n=12) ===")
s6 = S[S.agarose == "0.6%"]
m6 = smf.ols("plateau ~ C(coating) + C(bsa)", data=s6).fit()
print(sm.stats.anova_lm(m6, typ=2).round(4).to_string())
eff = -m6.params["C(coating)[T.PEG]"]
ci = m6.conf_int().loc["C(coating)[T.PEG]"]
print(f"  COOH - PEG = {eff:+.1f} L*, 95% CI [{-ci[1]:+.1f}, {-ci[0]:+.1f}]")

print("\n=== 3. COOH vs PEG per cell ===")
for (day, ag, bsa, arm), g in S.groupby(["day", "agarose", "bsa", "arm"]):
    c = g[g.coating == "COOH"].plateau.mean(); p = g[g.coating == "PEG"].plateau.mean()
    print(f"  {day} {ag} {bsa:8s} {arm:6s}  COOH {c:5.1f}  PEG {p:5.1f}  diff {c - p:+5.1f}  {'COOH ahead' if c > p else 'PEG ahead'}")
