"""Python port of the pstratreg estimator (Lundberg & Cho) and the
motherhood-analysis pipeline from analyze_motherhood.R.

Notes on equivalence with the R code:
* `treated*(X)` models are fully interacted, so they equal separate fits
  within each treatment arm; we fit per arm.
* `poly(age_1, 2)` spans the same column space as (age_1, age_1^2), so
  fitted values / predictions are identical.
* GLMs are fit by IRLS with prior weights exactly as R's glm().
* The greedy bound-aggregation loop in pstratreg() (O(n^2) in R) is computed
  exactly with cumulative sums.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

RACE = ["Hispanic", "Non-Hispanic Black", "Non-Hispanic Non-Black"]
MARITAL = ["cohabiting", "married", "no_partner"]


# ----------------------------------------------------------------- design
def covariates(d):
    """Covariate matrix (no intercept, no treatment) matching the R formula
    race + poly(age_1,2) + educ + marital + fulltime + log(tenure+1)
    + log(experience+1) + wage_baseline + employed_baseline."""
    cols = []
    for lv in RACE[1:]:
        cols.append((d["race"].values == lv).astype(float))
    a = d["age_1"].values.astype(float)
    a0 = a - 25.0  # centring only improves conditioning; same column space
    cols += [a0, a0 ** 2]
    e = d["educ"].values.astype(int)
    for lv in (2, 3, 4):
        cols.append((e == lv).astype(float))
    for lv in MARITAL[1:]:
        cols.append((d["marital"].values == lv).astype(float))
    cols.append(d["fulltime"].values.astype(float))
    cols.append(np.log(d["tenure"].values + 1))
    cols.append(np.log(d["experience"].values + 1))
    cols.append(d["wage_baseline"].values.astype(float))
    cols.append(d["employed_baseline"].values.astype(float))
    return np.column_stack(cols)


def with_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


# ------------------------------------------------------------------- fits
def wls(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def logit_irls(X, y, w, tol=1e-10, maxit=100):
    # R's binomial mustart: (w*y + 0.5)/(w + 1)
    mu = (w * y + 0.5) / (w + 1)
    eta = np.log(mu / (1 - mu))
    beta = None
    for _ in range(maxit):
        v = mu * (1 - mu)
        z = eta + (y - mu) / v
        b_new = wls(X, z, w * v)
        eta = X @ b_new
        # R's binomial()$linkinv clamps mu to [eps, 1 - eps]
        mu = np.clip(1 / (1 + np.exp(-np.clip(eta, -700, 700))), 2.22e-16, 1 - 2.22e-16)
        if beta is not None and np.max(np.abs(b_new - beta)) < tol * (1 + np.max(np.abs(beta))):
            beta = b_new
            break
        beta = b_new
    return beta


def gamma_log_irls(X, y, w, tol=1e-10, maxit=200):
    # R's Gamma mustart is y; with log link the working weight is w (constant)
    eta = np.log(y)
    beta = None
    for _ in range(maxit):
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        b_new = wls(X, z, w)
        eta = X @ b_new
        if beta is not None and np.max(np.abs(b_new - beta)) < tol * (1 + np.max(np.abs(beta))):
            beta = b_new
            break
        beta = b_new
    return beta


def expit(x):
    # R's logit_linkinv thresholds eta at +/-30, so mu never rounds to 0 or 1
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------------- shared fitting
class Fitted:
    """All model fits for one (imputed) dataset; shared by every setting."""

    def __init__(self, d):
        self.d = d
        X = with_intercept(covariates(d))
        t = d["treated"].values.astype(bool)
        w = d["w"].values.astype(float)
        s = d["employed"].values.astype(float)
        y = d["wage"].values.astype(float)
        valid = ~np.isnan(y)
        self.t, self.w = t, w

        # employment (logit) and wage (gaussian) models, per arm = full interaction
        self.s1 = expit(X @ logit_irls(X[t], s[t], w[t]))
        self.s0 = expit(X @ logit_irls(X[~t], s[~t], w[~t]))
        b1 = wls(X[t & valid], y[t & valid], w[t & valid])
        b0 = wls(X[~t & valid], y[~t & valid], w[~t & valid])
        self.y1 = X @ b1
        self.y0 = X @ b0
        yhat_obs = np.where(t, self.y1, self.y0)

        # variance-function regression (Gamma, log link), no interaction
        Xv = np.column_stack([X[:, :1], t.astype(float), X[:, 1:]])
        sq = 0.001 + (y - yhat_obs) ** 2
        bv = gamma_log_irls(Xv[valid], sq[valid], w[valid])
        # NB: as in pstratreg, sd is predicted at the *observed* treatment
        self.sd = np.sqrt(np.exp(Xv @ bv))

        # LPM for employment (the paper's employment_only branch uses glm()
        # with default gaussian family)
        l1 = wls(X[t], s[t], w[t])
        l0 = wls(X[~t], s[~t], w[~t])
        self.lpm_effect = X @ l1 - X @ l0

        # "standard practice": wage regression among employed, ATT among employed treated
        self.std_effect = self.y1 - self.y0


# ------------------------------------------------------------- bounds
def trunc_mean(p, sd, upper):
    """Mean of top (upper) or bottom share p of N(0, sd^2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if upper:
            a = norm.ppf(1 - p)
            return sd * norm.pdf(a) / (1 - norm.cdf(a))
        b = norm.ppf(p)
        return sd * (-norm.pdf(b)) / norm.cdf(b)


def greedy_bound(eff, wl, wu, lower):
    """Exact vectorised version of the while-loop in pstratreg().

    R behaviour reproduced: rows whose row-level bound is NaN (p_always_lower
    = 0 gives 0/0 in the truncated-normal mean) have weight_lower = 0, so
    weighted.mean() drops them from the initial estimate, and arrange() sorts
    NaN last so the loop stops before reaching them. They are therefore
    excluded. (If the loop ever reached them, R would error.)"""
    nan = np.isnan(eff)
    if nan.any():
        assert np.all(wl[nan] == 0)
        eff, wl, wu = eff[~nan], wl[~nan], wu[~nan]
    order = np.argsort(eff if lower else -eff, kind="stable")
    e, l, u = eff[order], wl[order], wu[order]
    num = np.sum(l * e) + np.concatenate([[0.0], np.cumsum((u - l) * e)])
    den = np.sum(l) + np.concatenate([[0.0], np.cumsum(u - l)])
    seq = num / den
    step = np.diff(seq)
    stop = np.nonzero(step > 0)[0] if lower else np.nonzero(step < 0)[0]
    if not len(stop) and nan.any():
        return np.nan  # R would hit a NaN row here and stop with an error
    return seq[stop[0]] if len(stop) else seq[-1]


def bounds(f, monotonicity_negative=False, mean_dominance=False, include_zero_rows=False):
    """include_zero_rows=True is a sensitivity analysis, not the paper's
    method: rows with p_always_lower = 0 get their row-level bound evaluated
    at the feasible share p_always_upper instead of being dropped."""
    """ATT-scale bounds among treated, as returned by the paper's estimator()."""
    s1, s0, sd, t, w = f.s1.copy(), f.s0.copy(), f.sd, f.t, f.w
    if monotonicity_negative:
        viol = s1 > s0
        mid = 0.5 * (s1 + s0)
        s1t = np.where(viol, mid, s1)
        s0t = np.where(viol, mid, s0)
        p_lo = p_hi = s1t
        pl0 = s1t / s0t
        pl1 = s1t / s1t
    else:
        p_lo = np.maximum(s1 + s0 - 1, 0)
        p_hi = np.minimum(s1, s0)
        p_eval = np.where((p_lo == 0) & include_zero_rows, p_hi, p_lo)
        # guard against 1 + 4e-16 from floating point (p_lo <= min(s1, s0) exactly)
        pl0, pl1 = np.minimum(p_eval / s0, 1), np.minimum(p_eval / s1, 1)
    y0l = f.y0 + trunc_mean(pl0, sd, False)
    y0u = f.y0 + trunc_mean(pl0, sd, True)
    y1l = f.y1 + trunc_mean(pl1, sd, False)
    y1u = f.y1 + trunc_mean(pl1, sd, True)
    if mean_dominance:  # y1_positive and y0_positive, as in the paper
        y1l, y0l = f.y1, f.y0
    eff_lo = y1l - y0u
    eff_hi = y1u - y0l

    tr = t
    out = {
        "s0": np.average(f.s0[tr], weights=w[tr]),
        "s1": np.average(f.s1[tr], weights=w[tr]),
    }
    out["effect_s"] = np.average((f.s1 - f.s0)[tr], weights=w[tr])
    if monotonicity_negative:
        ww = w[tr] * p_lo[tr]
        out["effect_y_lower"] = np.average(eff_lo[tr], weights=ww)
        out["effect_y_upper"] = np.average(eff_hi[tr], weights=ww)
    else:
        out["effect_y_lower"] = greedy_bound(eff_lo[tr], (p_lo * w)[tr], (p_hi * w)[tr], True)
        out["effect_y_upper"] = greedy_bound(eff_hi[tr], (p_lo * w)[tr], (p_hi * w)[tr], False)
    out["n_nan"] = int(np.isnan(eff_lo[tr]).sum() + np.isnan(eff_hi[tr]).sum())
    return out


SUBGROUPS = ["Not employed", "Below $15", "$15 to $20", "Over $20"]


def subgroup_codes(d, use_baseline):
    emp = d["employed_baseline"] if use_baseline else d["employed"]
    wb = d["wage_baseline"].values
    return np.select(
        [~emp.values.astype(bool), wb < np.log(15), wb < np.log(20)], [0, 1, 2], default=3
    )


def all_estimates(f):
    """Everything the paper computes from one imputed dataset for one sex."""
    d, t, w = f.d, f.t, f.w
    res = {}
    for mono in (False, True):
        for md in (False, True):
            b = bounds(f, mono, md)
            for k in ("s0", "s1", "effect_s", "effect_y_lower", "effect_y_upper"):
                res[f"{k}|mono={int(mono)}|md={int(md)}"] = b[k]
    for tag, base in (("ascoded", False), ("baseline", True)):
        g = subgroup_codes(d, base)
        for i, name in enumerate(SUBGROUPS):
            m = t & (g == i)
            res[f"sub_{tag}|{name}"] = np.average(f.lpm_effect[m], weights=w[m]) if m.any() else np.nan
    emp_t = t & d["employed"].values.astype(bool)
    res["standard_practice"] = np.average(f.std_effect[emp_t], weights=w[emp_t])
    return res


# ------------------------------------------------------------ imputation
IMP_NUM = ["age_1", "treated", "employed", "wage", "employed_baseline",
           "wage_baseline", "educ", "tenure", "experience"]


def _to_matrix(d):
    cols, names = [], []
    for c in IMP_NUM:
        cols.append(d[c].astype(float).to_numpy(copy=True))
        names.append(c)
    for lv in RACE[1:]:
        cols.append((d["race"] == lv).astype(float).values); names.append("race=" + lv)
    for lv in MARITAL[1:]:
        x = (d["marital"] == lv).astype(float).to_numpy(copy=True)
        x[d["marital"].isna().values] = np.nan
        cols.append(x); names.append("marital=" + lv)
    x = d["fulltime"].astype(float).to_numpy(copy=True)
    cols.append(x); names.append("fulltime")
    return np.column_stack(cols), names


def em_mvn(Y, maxit=1000, tol=1e-6, ridge=1e-8):
    """EM for a multivariate normal with missing values (Amelia's core,
    without the bootstrap since the paper uses boot.type = "none")."""
    n, p = Y.shape
    M = np.isnan(Y)
    mu = _colstat(Y, np.mean)
    Z = np.where(M, mu, Y)
    S = np.cov(Z, rowvar=False) + ridge * np.eye(p)
    patterns = {}
    for i, row in enumerate(M):
        patterns.setdefault(row.tobytes(), []).append(i)
    pats = [(np.frombuffer(k, dtype=bool), np.array(v)) for k, v in patterns.items()]
    for _ in range(maxit):
        T1 = np.zeros(p)
        T2 = np.zeros((p, p))
        for m, idx in pats:
            Yi = Y[np.ix_(idx, ~m)]
            if not m.any():
                T1 += Yi.sum(0)
                T2 += Yi.T @ Yi
                continue
            o = ~m
            Soo = S[np.ix_(o, o)]
            Smo = S[np.ix_(m, o)]
            B = np.linalg.solve(Soo, Smo.T).T
            C = S[np.ix_(m, m)] - B @ Smo.T
            Yhat = np.empty((len(idx), p))
            Yhat[:, o] = Yi
            Yhat[:, m] = mu[m] + (Yi - mu[o]) @ B.T
            T1 += Yhat.sum(0)
            T2 += Yhat.T @ Yhat
            T2[np.ix_(m, m)] += len(idx) * C
        mu_new = T1 / n
        S_new = T2 / n - np.outer(mu_new, mu_new) + ridge * np.eye(p)
        delta = max(np.max(np.abs(mu_new - mu)), np.max(np.abs(S_new - S)))
        mu, S = mu_new, S_new
        if delta < tol:
            break
    return mu, S, pats


# numpy's nanmin/nanmax returned NaN on some resampled columns in this
# environment (with only ~6% missing), so reduce over explicit masks.
def _colstat(Y, fn):
    return np.array([fn(c[~np.isnan(c)]) for c in Y.T])


def _range(v):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return v.min(), v.max()


def impute(d, rng, ref=None):
    """Single stochastic imputation (m = 1) following Amelia's conventions:
    draws from the conditional normal at the EM estimates; ordinal educ is
    rounded; nominal dummies are turned into category draws; numeric
    variables are clamped to the observed range of `ref` (the original,
    pre-bootstrap data, as in the R code); wage is restored."""
    Y, names = _to_matrix(d)
    sc_mu = _colstat(Y, np.mean)
    sc_sd = _colstat(Y, np.std)
    sc_sd[sc_sd == 0] = 1
    Ys = (Y - sc_mu) / sc_sd
    mu, S, pats = em_mvn(Ys)
    Z = Ys.copy()
    for m, idx in pats:
        if not m.any():
            continue
        o = ~m
        B = np.linalg.solve(S[np.ix_(o, o)], S[np.ix_(m, o)].T).T
        C = S[np.ix_(m, m)] - B @ S[np.ix_(m, o)].T
        L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
        cm = mu[m] + (Ys[np.ix_(idx, o)] - mu[o]) @ B.T
        Z[np.ix_(idx, m)] = cm + rng.standard_normal((len(idx), m.sum())) @ L.T
    Z = Z * sc_sd + sc_mu
    out = d.copy()
    col = {n: i for i, n in enumerate(names)}
    for c in ["wage_baseline", "tenure", "experience"]:
        miss = d[c].isna().values
        if miss.any():
            lo, hi = _range((d if ref is None else ref)[c].values)
            out.loc[miss, c] = np.clip(Z[miss, col[c]], lo, hi)
    miss = d["educ"].isna().values
    if miss.any():
        out.loc[miss, "educ"] = np.clip(np.round(Z[miss, col["educ"]]), 1, 4)
    miss = d["marital"].isna().values
    if miss.any():
        p = np.clip(Z[miss][:, [col["marital=married"], col["marital=no_partner"]]], 0, 1)
        p = np.column_stack([np.clip(1 - p.sum(1), 0, 1), p])
        p = p / p.sum(1, keepdims=True)
        u = rng.random(miss.sum())
        k = (u[:, None] > np.cumsum(p, 1)).sum(1)
        out.loc[miss, "marital"] = np.array(MARITAL)[np.minimum(k, 2)]
    miss = d["fulltime"].isna().values
    if miss.any():
        p = np.clip(Z[miss, col["fulltime"]], 0, 1)
        out.loc[miss, "fulltime"] = rng.random(miss.sum()) < p
    out["educ"] = out["educ"].astype(int)
    out["fulltime"] = out["fulltime"].astype(bool)
    return out


# ------------------------------------------------------------- data I/O
def load_prepared(path):
    import pyreadr
    d = pyreadr.read_r(path)[None]
    d["educ"] = d["educ"].cat.codes.replace(-1, np.nan).astype(float) + 1
    d["race"] = d["race"].astype(str)
    d["marital"] = d["marital"].astype(object)
    for c in ["treated", "employed", "employed_baseline"]:
        d[c] = d[c].astype(bool)
    d["fulltime"] = d["fulltime"].astype(object).where(d["fulltime"].notna(), np.nan)
    return d
