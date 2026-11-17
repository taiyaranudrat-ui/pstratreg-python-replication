"""Port of sim_normality_violated.R using exact integration instead of 1e5
Monte Carlo draws: the bound is the mean of the top (or bottom) share
`keep` of a standardized error distribution = average of its quantile
function over [1-keep, 1] (or [0, keep])."""
import numpy as np
import pandas as pd
from scipy import integrate, stats

dists = {
    "normal": stats.norm.ppf,
    "uniform": lambda p: p * 2 * np.sqrt(3) - np.sqrt(3),
    "t": lambda p: stats.t.ppf(p, 3) * np.sqrt(1 / 3),
    "expo": lambda p: stats.expon.ppf(p) - 1,
}


def mean_q(q, a, b):
    return integrate.quad(q, a, b, limit=200)[0] / (b - a)


rows = []
for keep in np.round(np.arange(0.5, 1.0001, 0.025), 3):
    for name, q in dists.items():
        rows.append(dict(prop_keep=keep, name=name,
                         upper=mean_q(q, 1 - keep, 1), lower=mean_q(q, 0, keep)))
sim = pd.DataFrame(rows)
if __name__ == "__main__":
    import sys
    for k in (0.6, 0.95):
        print(sim[sim.prop_keep == k].round(2).to_string(index=False))
    norm_ = sim[sim.name == "normal"].set_index("prop_keep")
    bad = sim[sim.name != "normal"].join(norm_[["upper", "lower"]], on="prop_keep", rsuffix="_normal")
    bad["outside"] = (bad.upper > bad.upper_normal + 1e-12) | (bad.lower < bad.lower_normal - 1e-12)
    print(bad.groupby("name").outside.mean().rename("share of keep-values where true interval exceeds normal-assumed interval"))
    sim.to_csv(sys.argv[1], index=False)
