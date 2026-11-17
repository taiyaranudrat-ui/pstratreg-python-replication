# Python replication of *Non-existent outcomes in research on inequality*

This is an independent Python replication of Lundberg & Cho, *"Non-existent outcomes in research on inequality: A causal approach"*. It re-implements the authors' R package [`pstratreg`](https://github.com/ilundberg/pstratreg) and the motherhood/fatherhood analysis from their replication package, then compares every published estimate.

**Result: the paper replicates.** Every one of the 30 published point estimates is reproduced to within 0.004 log points. Standard errors and 95% intervals agree closely. Along the way we found one coding error in the subgroup figure and a few smaller issues ([below](#issues-found-in-the-original-code)).

---

## The question

How does parenthood affect hourly wages? Wages exist only for people who work, and motherhood lowers employment by about 14 percentage points. So "mothers vs. non-mothers among the employed" compares two different groups of people.

The paper targets a well-defined group: people who would be **employed whether or not they became parents** (the "always-employed" principal stratum). That group's membership is not observed, so the effect is **set-identified**. The method returns a lower and upper bound, which narrow as you add assumptions:

| Assumption | Meaning here |
|---|---|
| **Monotonicity** | Motherhood never *causes* a woman to be employed (S¹ ≤ S⁰). Not used for men. |
| **Mean dominance** | Always-employed people earn at least as much as people whose employment depends on parenthood. |

Bounds come from parametric models: a logit for employment, a linear model for log wage, and a Gamma variance-function regression. A normal error assumption places the "dropped" survivors in the tails.

## Headline results

Effect of parenthood on log hourly wage among the always-employed (ATT). The paper's numbers come from the authors' shipped `estimates.RDS`. The Python numbers are the mean of 20 imputations, with percentile intervals from a 1,000-rep person-level bootstrap.

| Group | Assumptions | Paper bounds | Python bounds | Paper 95% CI | Python 95% CI |
|---|---|---|---|---|---|
| Mothers | none | [−0.387, 0.377] | [−0.387, 0.378] | [−0.428, 0.420] | [−0.427, 0.423] |
| Mothers | mean dominance | [−0.234, 0.152] | [−0.235, 0.151] | [−0.266, 0.183] | [−0.267, 0.185] |
| Mothers | monotonicity | [−0.124, 0.117] | [−0.125, 0.116] | [−0.153, 0.142] | [−0.151, 0.146] |
| Mothers | both | **[−0.123, −0.003]** | **[−0.125, −0.004]** | [−0.154, 0.023] | [−0.151, 0.026] |
| Fathers | none | [−0.197, 0.380] | [−0.198, 0.384] | [−0.230, 0.419] | [−0.232, 0.422] |
| Fathers | mean dominance | [−0.052, 0.243] | [−0.052, 0.240] | [−0.077, 0.271] | [−0.082, 0.274] |

Effect of parenthood on employment (ATT):

| | Paper | Python |
|---|---|---|
| Mothers | −0.140 (SE 0.011) | −0.139 (SE 0.011) |
| Fathers | +0.004 (SE 0.009) | +0.004 (SE 0.009) |

"Standard practice" (regress wages among the employed), mothers: paper −0.003, Python −0.004.

Full tables: [`results/summary.txt`](results/summary.txt) and [`results/main_estimates_comparison.csv`](results/main_estimates_comparison.csv).

## How the replication was validated

We checked in three stages, each a stricter test than the last:

1. **Data construction (R, unchanged).** We re-ran `prepare_data_motherhood.R` on the raw NLSY97 extract. Its output is **identical** to the shipped `motherhood.RDS`: 50,267 person-pair rows, `all.equal` returns `TRUE`, and the ID lists are byte-identical.
2. **Estimator port.** Python and R's `pstratreg` were run on the *same* imputed datasets for all 6 settings plus the subgroup models. The maximum absolute difference across all 38 quantities is **4.2 × 10⁻⁶**. That difference is GLM convergence tolerance; most quantities agree to 10⁻¹². See [`validation/`](validation/).
3. **Full independent pipeline.** Python's own imputation, 20 point imputations, and 1,000 bootstrap reps each for two bootstrap schemes. The maximum difference from the paper's point estimates is **0.0036**. That's inside the spread you get from imputation draws alone (SD up to 0.0027): the paper uses a single imputation (`m = 1`), so its point estimates carry that much noise too.

The normal-error simulation (`sim_normality_violated.R`) was ported using exact integration in place of 10⁵ Monte Carlo draws. It reproduces the paper's printed table, apart from Monte Carlo noise in two cells that the original rounds to 2 decimals.

## Issues found in the original code

| # | Issue | Effect on published results |
|---|---|---|
| 1 | **Subgroup figure groups on a post-treatment variable.** `analyze_motherhood.R` builds "baseline outcome" groups with `!employed`, which is employment *after* the birth. It should be `employed_baseline`. 29% of mothers and 23% of fathers change group when this is fixed. | Women: same pattern, a bit larger (e.g. "Not employed" −0.159 → −0.170, with SE nearly doubling to 0.026). **Men: changes the story.** Fathers not employed at baseline show **+0.071 (SE 0.025)**, against +0.014 published, and the "Below $15" group flips sign (+0.020 → −0.018). |
| 2 | **Bootstrap is row-level, not person-level.** The code builds a person-level cluster resample, then overwrites it with `sample_frac(data, replace = T)`. People contribute up to 17 rows (mean 7.1). | Small. Person-level SEs for the wage bounds are on average 5% larger (at most 13% larger). Conclusions are unchanged. |
| 3 | **Some people are silently dropped from the bounds.** When predicted s¹ + s⁰ ≤ 1 (about 9–10% of treated rows), the row-level bound is 0/0. R's `weighted.mean`/`arrange` behaviour drops these rows instead of letting them widen the bound. | Negligible: bounds widen by ≤ 0.003 if they are included at a feasible value. |
| 4 | **Different employment models.** Subgroup employment effects use `glm()` with no `family`, which is a linear probability model. The overall effect uses a logit inside `pstratreg`. | Cosmetic; estimates are close. |
| 5 | **Runtime.** `pstratreg`'s bound aggregation is an O(n²) loop: 20–35 s per fit, which is most of the paper's 3.4 h on 32 cores. | None. The Python port computes the same quantity exactly with cumulative sums (~0.4 s per fit). The full analysis takes **6 minutes on 4 cores**. |

## Repository layout

```
code/
  pstrat.py          # pstratreg port, Amelia-style EM imputation, all estimators
  replicate.py       # full pipeline: point estimates + row and cluster bootstraps
  summarize.py       # compares Python output with the paper's .RDS results
  validate.py        # Python vs R pstratreg on identical imputed data
  sim_normality.py   # port of sim_normality_violated.R (exact integration)
  r_reference.R      # produces the R side of the validation
results/
  summary.txt                        # printed comparison tables
  main_estimates_comparison.csv      # 30 estimands: paper vs Python, SEs, CIs
  subgroup_comparison.csv            # as-coded and corrected subgroups
  standard_practice_comparison.csv
  draws.csv                          # every imputation and bootstrap draw
  sim_normality.csv
validation/
  validation_python_vs_r.csv         # stage-2 check (max diff 4.2e-6)
  r_estimates.csv, r_subgroups.csv
```

## Running it

You need the authors' replication package for the data. NLSY97 data are not redistributed here.

```bash
pip install numpy pandas scipy pyreadr
```

1. Build `intermediate/motherhood.RDS` with the authors' `prepare_data_motherhood.R`, or use the copy shipped in their package.
2. Run the replication (point estimates plus 1,000 reps of each bootstrap, per sex):

```bash
OMP_NUM_THREADS=1 python3 code/replicate.py path/to/motherhood.RDS results 1000 4
```

3. Compare with the paper:

```bash
python3 code/summarize.py results/draws.csv path/to/authors/intermediate results
```

4. Optional: validate against R (needs R with `Amelia`, `dplyr`, `tidyr` and the bundled `pstratreg`):

```bash
Rscript code/r_reference.R path/to/motherhood.RDS validation
```

```bash
python3 code/validate.py validation code
```

Set `OMP_NUM_THREADS=1`. Otherwise every worker spins up its own BLAS threads and the run slows down several-fold.

## Implementation notes

- **Interactions.** `treated*(X)` models are fully interacted, so they are fit as separate regressions per treatment arm. `poly(age, 2)` is replaced by centred age and age², which spans the same column space, so predictions are identical.
- **GLMs.** These use IRLS with R's starting values and prior-weight conventions. The logit link clamps the linear predictor at ±30 as R does; without this, near-separation in a few bootstrap samples produced probabilities of exactly 1.
- **Imputation.** Amelia (`m = 1`, `boot.type = "none"`) is approximated by EM for a multivariate normal followed by a conditional draw. Ordinal `educ` is rounded, nominal variables are drawn from imputed dummies, and numeric variables are clamped to the original data's range. Only about 6% of rows have any imputed covariate. This is the one component that is not an exact port, which is why stage 3 is compared at the level of imputation noise.
- **Bootstrap sharing.** Each bootstrap replicate shares one resample and one imputation across all assumption settings for that sex. The R code draws separately per setting; the marginal intervals have the same distribution.
- **numpy workaround.** With numpy 2.5.3 in our environment, `np.nanmin` sometimes returned NaN on resampled columns. All NaN-aware reductions use explicit masks instead.

## Credit

Method, data construction, and original code: Ian Lundberg and Soonhong Cho. This repository only re-implements and checks their work. Please cite the original paper.
