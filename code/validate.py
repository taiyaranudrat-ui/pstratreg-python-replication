"""Validate the Python port against R pstratreg on identical imputed data."""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, sys.argv[2])
import pstrat as ps

vdir = sys.argv[1]
r = pd.read_csv(f"{vdir}/r_estimates.csv")
rs = pd.read_csv(f"{vdir}/r_subgroups.csv")
rows = []
for sex in ["Men", "Women"]:
    d = pd.read_csv(f"{vdir}/imputed_{sex}.csv")
    t0 = time.time()
    f = ps.Fitted(d)
    est = ps.all_estimates(f)
    el = time.time() - t0
    for _, x in r[r.sex == sex].iterrows():
        key = f"{x.estimand}|mono={int(x.monotonicity)}|md={int(x.mean_dominance)}"
        rows.append(dict(sex=sex, mono=x.monotonicity, md=x.mean_dominance,
                         estimand=x.estimand, R=x.estimate, Python=est[key]))
    for _, x in rs[rs.sex == sex].iterrows():
        rows.append(dict(sex=sex, mono="", md="", estimand="subgroup: " + x.wage_baseline_categories,
                         R=x.effect_s, Python=est[f"sub_ascoded|{x.wage_baseline_categories}"]))
    print(f"{sex}: all estimates from one imputed dataset in {el:.2f}s; NaN bound rows: "
          f"{[ps.bounds(f, m, md)['n_nan'] for m in (0, 1) for md in (0, 1)]}")
out = pd.DataFrame(rows)
out["abs_diff"] = (out.R - out.Python).abs()
pd.set_option("display.width", 200)
print(out.to_string(float_format=lambda v: f"{v:.6g}"))
print("max abs diff:", out.abs_diff.max())
out.to_csv(f"{vdir}/validation_python_vs_r.csv", index=False)
