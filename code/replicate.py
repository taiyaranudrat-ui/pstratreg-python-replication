"""Full Python replication of analyze_motherhood.R + standard_practice_comparison.R.

usage: replicate.py <motherhood.RDS> <outdir> <reps> <workers> [seed]

For each sex, and each bootstrap replicate, one resample + one imputation is
shared across all settings (the R code draws a fresh resample per setting;
marginal intervals are equivalent in distribution).
Two bootstraps are run:
  row     - resample rows, which is what the R code actually does
            (the person-level resample is overwritten by sample_frac())
  cluster - resample people with all their rows, which is what the R code's
            comment says it intends
"""
import sys, time, os
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pstrat as ps

D = None


def _init(path):
    global D
    d = ps.load_prepared(path)
    D = {s: d[d.sex == s].reset_index(drop=True) for s in ["Men", "Women"]}


def one(args):
    sex, kind, seed = args
    rng = np.random.default_rng(seed)
    d = D[sex]
    if kind == "point":
        ds = d
    elif kind == "row":
        ds = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
    else:
        ids = d["PUBID"].unique()
        pick = rng.choice(ids, len(ids), replace=True)
        groups = d.groupby("PUBID").indices
        ds = d.iloc[np.concatenate([groups[i] for i in pick])].reset_index(drop=True)
    try:
        di = ps.impute(ds, rng, ref=d)
        f = ps.Fitted(di)
        est = ps.all_estimates(f)
        for md in (0, 1):
            b = ps.bounds(f, False, md, include_zero_rows=True)
            est[f"effect_y_lower|mono=0|md={md}|incl0"] = b["effect_y_lower"]
            est[f"effect_y_upper|mono=0|md={md}|incl0"] = b["effect_y_upper"]
        est["failed"] = ""
    except Exception as e:  # record, don't crash the run
        est = {"failed": f"{type(e).__name__}: {e}"}
    est.update(sex=sex, kind=kind, seed=seed)
    return est


if __name__ == "__main__":
    path, outdir, reps, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    base = int(sys.argv[5]) if len(sys.argv) > 5 else 14850
    os.makedirs(outdir, exist_ok=True)
    ss = np.random.SeedSequence(base)
    seeds = iter(int(s.generate_state(1)[0]) for s in ss.spawn(10 ** 5))
    jobs = []
    for sex in ["Men", "Women"]:
        jobs += [(sex, "point", next(seeds)) for _ in range(20)]  # 20 imputations of full data
        for kind in ["row", "cluster"]:
            jobs += [(sex, kind, next(seeds)) for _ in range(reps)]
    t0 = time.time()
    rows = []
    with Pool(workers, initializer=_init, initargs=(path,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{outdir}/draws.csv", index=False)
    print(f"finished {len(jobs)} jobs in {time.time() - t0:.0f}s")
