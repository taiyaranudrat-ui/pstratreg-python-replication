"""Compare the Python replication with the paper's shipped estimates.
usage: summarize.py <draws.csv> <paper_intermediate_dir> <outdir>"""
import sys
import numpy as np
import pandas as pd
import pyreadr

draws = pd.read_csv(sys.argv[1])
pdir, out = sys.argv[2], sys.argv[3]
paper = pyreadr.read_r(f"{pdir}/estimates.RDS")[None]
sub_paper = {"Women": pyreadr.read_r(f"{pdir}/motherhood_within_subgroups.RDS")[None],
             "Men": pyreadr.read_r(f"{pdir}/fatherhood_within_subgroups.RDS")[None]}
std_paper = pyreadr.read_r(f"{pdir}/standard_practice_mothers.RDS")[None]


def summ(sex, col, normal_ci=False):
    p = draws[(draws.sex == sex) & (draws.kind == "point")][col]
    r = draws[(draws.sex == sex) & (draws.kind == "row")][col].dropna()
    c = draws[(draws.sex == sex) & (draws.kind == "cluster")][col].dropna()
    est = p.mean()
    res = dict(py_est=est, py_imp_sd=p.std(), py_se_row=r.std(), py_se_cluster=c.std())
    if normal_ci:
        res.update(py_lo_cluster=est - 1.96 * c.std(), py_hi_cluster=est + 1.96 * c.std())
    else:
        res.update(py_lo_row=r.quantile(.025), py_hi_row=r.quantile(.975),
                   py_lo_cluster=c.quantile(.025), py_hi_cluster=c.quantile(.975))
    return res


rows = []
for _, x in paper.iterrows():
    col = f"{x.estimand}|mono={int(x.monotonicity)}|md={int(x.mean_dominance)}"
    r = dict(sex=x.sex, monotonicity=bool(x.monotonicity), mean_dominance=bool(x.mean_dominance),
             estimand=x.estimand, paper_est=x.estimate, paper_se=x.se, paper_lo=x["ci.min"], paper_hi=x["ci.max"])
    r.update(summ(x.sex, col))
    if x.estimand.startswith("effect_y") and not x.monotonicity:
        r["py_est_incl_zero_rows"] = draws[(draws.sex == x.sex) & (draws.kind == "point")][col + "|incl0"].mean()
    rows.append(r)
main = pd.DataFrame(rows)
main["diff"] = main.py_est - main.paper_est
main["se_ratio_cluster_vs_row"] = main.py_se_cluster / main.py_se_row
main.to_csv(f"{out}/main_estimates_comparison.csv", index=False)

srows = []
for sex in ["Women", "Men"]:
    for _, x in sub_paper[sex].iterrows():
        g = x.wage_baseline_categories
        r = dict(sex=sex, subgroup=g, paper_est=x.effect_s, paper_se=x.se, paper_lo=x["ci.min"], paper_hi=x["ci.max"])
        a = summ(sex, f"sub_ascoded|{g}", True)
        b = summ(sex, f"sub_baseline|{g}", True)
        r.update({"py_ascoded_" + k: v for k, v in a.items()})
        r.update({"py_corrected_" + k: v for k, v in b.items()})
        srows.append(r)
subs = pd.DataFrame(srows)
subs.to_csv(f"{out}/subgroup_comparison.csv", index=False)

sp = summ("Women", "standard_practice")
sp.update(paper_est=std_paper.estimate[0], paper_se=std_paper.se[0], paper_lo=std_paper["ci.min"][0], paper_hi=std_paper["ci.max"][0])
pd.DataFrame([sp]).to_csv(f"{out}/standard_practice_comparison.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
f = lambda v: f"{v:.4f}"
print(main[["sex", "monotonicity", "mean_dominance", "estimand", "paper_est", "py_est", "diff", "py_imp_sd",
            "paper_se", "py_se_row", "py_se_cluster", "paper_lo", "paper_hi", "py_lo_cluster", "py_hi_cluster",
            "py_est_incl_zero_rows"]].to_string(float_format=f))
print(subs[["sex", "subgroup", "paper_est", "py_ascoded_py_est", "paper_se", "py_ascoded_py_se_row",
            "py_ascoded_py_se_cluster", "py_corrected_py_est", "py_corrected_py_se_cluster"]].to_string(float_format=f))
print(pd.DataFrame([sp]).to_string(float_format=f))
print("max |diff| main:", main["diff"].abs().max())
print("n draws:", draws.groupby(["sex", "kind"]).size().to_dict())
