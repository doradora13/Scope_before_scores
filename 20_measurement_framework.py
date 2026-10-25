# -*- coding: utf-8 -*-
"""
20_measurement_framework.py - testable implications of the measurement framework (Section 3; Supplementary Section S14).

Measurement model (reported emissions = complete-scope emissions x reported share):
    ln E_i = ln E*_i - w_i,   w_i = -ln pi_i >= 0,   pi_i = (emissions of the reported sectors) / (emissions if all six sectors were reported)
    If ln E*_i = x_i'b + a_c + v_i + u_i, then ln E_i = x_i'b + a_c + v_i + u_i - w_i  (the two-tier structure follows from the measurement model)

Proposition 1 (coverage contrasts in levels): under Assumptions 1 and 3 (ignorable coverage, transferable donor shares),
    E[ln E | x, r, c, S] = linear part + mu - w~(S, c).
    Hence, in a regression on the same regressors, the coefficient of a coverage indicator = -(its coefficient when w~ is regressed on them).
    Tests (1a) compute a 'mechanical shortfall' w_hat for each inventory from the sector shares of the six-sector inventories (donors)
              (donors from the same country if it has at least 20, otherwise all donors; w_hat = mean over donors of
              -ln(sum of the reported-sector shares)). The coverage coefficients of model (3) (Table 2) are compared with the
              sign-reversed coefficients from regressing w_hat on the same regressors (the mechanical prediction).
              (Model (3) also contains the core-five indicator, so the reference cell for the sector-count coefficients is
              'five sectors, core five incomplete', 1,139 inventories.)
         (1b) with w_hat in the regression its coefficient should be -1. Joint test of whether the sector-count indicators still
              matter once w_hat is included (NUTS3-clustered Wald). Also: the coefficient in the subsample with four or more
              sectors, the share of the identifying variation of w_hat that comes from one- and two-sector inventories, and the
              interval from a two-stage bootstrap that also redraws the donors (NUTS3 regions within country + donors by country,
              B_LAMBDA replications).
         (1c) coefficients of the reported-sector sets with at least 20 inventories vs the mechanical shortfall (Fig. 2a).
    Corollary 1 (benchmarks): w >= 0 implies ln E <= ln E*, so quantile benchmarks from coverage-mixed peer groups are biased downward.
         Test: tau = 0.25 benchmark from complete reporters only (as in stage 10) vs a benchmark from all inventories (no coverage controls).
Proposition 2 (residual asymmetry): if (v, u, w) are mutually independent within a coverage class, kappa3(eps|c) = kappa3(u|c) - kappa3(w|c).
    Tests (2a) exact decomposition of the third moment of the model (3) residuals by class (with the class dummies in the model, the
              within-class residual means are 0 and the shares sum to 1).
         (2b) NUTS3-cluster bootstrap confidence intervals for the skewness of each class (NUTS3 regions resampled with
              replacement within country, model re-estimated).
         (2c) residual skewness implied by the fitted exponential two-tier models (c) and (d) (does a model with distributional
              assumptions reproduce the observed skewness?).
         (2d) sector coverage of the inventories removed by trimming the lowest 1%.
         (2e) residual skewness when coverage is controlled by reported-sector sets or by w_hat instead of the number of sectors.
Proposition 3 (changes: scope and retained-sector terms): for baseline b, monitoring m and retained sectors C, exactly
         Dln E = Dln E_C + ln(1 - baseline share of the removed sectors) - ln(1 - monitoring share of the added sectors).
    Tests (3a) the scope term (scope = dT - dR) equals ln(1 - s) in pairs that remove one sector and add none (identity check).
         (3b) exact decomposition of the excess of the all-pairs mean over the constant-coverage mean ('inflation') into the
              scope term and the retained-sector term.
         (3c) how close the scope-corrected estimate (mean retained-sector change over all pairs) is to the constant-coverage mean.
         (3b)-(3c) come with NUTS3-cluster bootstrap confidence intervals (resampling within country).
         The scope-term identity is checked in every pair with rho- (baseline share of the removed sectors) and rho+
         (monitoring share of the added sectors) computed from the sector data.
         For historical reasons the retained-sector term is named revision_* in variable names and output files. It combines
         possible revisions with differences in underlying trends and is not an identified revision effect.
         Because elapsed years differ across categories, the null hypothesis is assessed on the annualised scale.
Inputs : work/data.pkl, work/repeated.pkl, work/twotier.pkl
Outputs: work/framework.pkl, tables/Framework_Tables.xlsx (Tables S14-S18), tables/framework_numbers.csv,
         figures/framework/Figure_M1 (Fig. 2; Propositions 1-2), Figure_M2 (Fig. 4; Proposition 3)
"""
from lib import *

B, SEED = 999, 20          # bootstrap replications and random seed (keep unchanged to reproduce the manuscript values)
B_LAMBDA = 499             # replications of the two-stage bootstrap (clusters + donors) for the w_hat coefficient
MIN_DONORS = 20            # if a country has fewer six-sector donors than this, the pooled donors are used
MIN_SET = 20               # minimum number of inventories for a reported-sector set to enter Fig. 2a and the set regression separately
FIG_FW = os.path.join(OUT_DIR, "figures", "framework")
os.makedirs(FIG_FW, exist_ok=True)
ABBR = ["Mu", "Te", "Re", "Ma", "Tr", "Wa"]   # sector abbreviations (SECTOR_NAMES order)

banner("20 measurement framework: testable implications of reported = complete-scope x coverage share")
main = load_obj("data.pkl")["main"].reset_index(drop=True)
rep = load_obj("repeated.pkl")
tt = load_obj("twotier.pkl")
y = main.lnE.values
cl = main.nuts3.values
cty = main.country_code.values
sets = main.sector_set.values
sc = main.sc.values
complete = ((main.sc >= 5) & (main.core5_complete_flag == 1)).values
nums = []                  # numbers for the manuscript (name, value, note)


def note(name, value, txt=""):
    nums.append(dict(name=name, value=float(value), note=txt))


def ols_V(yv, X, cluster):
    """OLS coefficients and cluster-robust covariance (same small-sample correction as lib.ols)."""
    X = np.asarray(X, float); n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ yv; e = yv - X @ b
    uniq, inv = np.unique(cluster, return_inverse=True); G = len(uniq)
    M = np.zeros((G, k)); np.add.at(M, inv, X * e[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX_inv @ (M.T @ M) @ XtX_inv
    return b, V, e


def set_label(s):
    """Label of a reported-sector set: the missing sectors ("no ...") if four or more are reported, otherwise the reported sectors."""
    rep_ = [a for a, c in zip(ABBR, s) if c == "1"]; mis = [a for a, c in zip(ABBR, s) if c == "0"]
    if len(rep_) >= 4:
        return "no " + ", ".join(mis)
    return " ".join(rep_) + (" only" if len(rep_) == 1 else "")


def m3_share(e, mask):
    return float(np.sum(e[mask] ** 3) / np.sum(e ** 3))


# =====================================================================================
# Proposition 1: mechanical coverage shortfall w_hat
# =====================================================================================
E6 = main[SECTOR_COLS].values.astype(float)
six = sc == 6
share6 = E6[six] / E6[six].sum(1, keepdims=True)
cty6 = cty[six]
donor_n = pd.Series(cty6).value_counts()
print(f"donors (6-sector inventories): {six.sum()}; countries with >= {MIN_DONORS} donors:",
      {c: int(n) for c, n in donor_n.items() if n >= MIN_DONORS})


def mech_w(setstr, c, pooled=False):
    mask = np.array([ch == "1" for ch in setstr])
    if mask.all():
        return 0.0
    D = share6 if (pooled or donor_n.get(c, 0) < MIN_DONORS) else share6[cty6 == c]
    return float(np.mean(-np.log(np.clip(D[:, mask].sum(1), 1e-6, None))))


combo = pd.DataFrame({"s": sets, "c": cty}).drop_duplicates()
wmap = {(r.s, r.c): mech_w(r.s, r.c) for r in combo.itertuples()}
wmap_pool = {(r.s, r.c): mech_w(r.s, r.c, pooled=True) for r in combo.itertuples()}
main["w_hat"] = [wmap[(s, c)] for s, c in zip(sets, cty)]
main["w_hat_pooled"] = [wmap_pool[(s, c)] for s, c in zip(sets, cty)]
w_hat = main.w_hat.values

# (1a) coverage coefficients of model (3) vs the mechanical prediction = -(coefficients from regressing w_hat on the same regressors)  [Proposition 1]
X3, n3 = design(main, BASE + Q, fe="country_code")
b3, V3, e3 = ols_V(y, X3, cl)
se3 = np.sqrt(np.diag(V3))
bW, _, _ = ols_V(w_hat, X3, cl)
COV_TERMS = ["sc1", "sc2", "sc3", "sc4", "sc6", "core5_complete_flag"]
rows = []
for c_ in COV_TERMS:
    j = n3.index(c_)
    rows.append(dict(term=c_, n=int(main[c_].sum()), coef_model3=b3[j], coef_se=se3[j], mechanical=-bW[j],
                     mechanical_share=(-bW[j] / b3[j]) if abs(b3[j]) > 1e-12 else np.nan))
t_count = pd.DataFrame(rows)
core5 = main.core5_complete_flag.values == 1
ref0, ref1 = (sc == 5) & ~core5, (sc == 5) & core5
t_class = pd.DataFrame([dict(sectors=k, n=int((sc == k).sum()), mean_w_hat=w_hat[sc == k].mean(),
                             mean_w_hat_pooled=main.w_hat_pooled.values[sc == k].mean(),
                             geo_mean_coverage=np.exp(-w_hat[sc == k].mean())) for k in range(1, 7)])
print("\n(1a) coverage coefficients of model (3) vs the mechanical prediction -(coefficients of w_hat on the same regressors):")
print(t_count.round(3).to_string(index=False))
print(f"     reference cells: five sectors without core-five completeness n={ref0.sum()} (mean w_hat {w_hat[ref0].mean():.3f}); "
      f"five sectors with core five n={ref1.sum()} (mean w_hat {w_hat[ref1].mean():.3f}); complete reporters mean w_hat {w_hat[complete].mean():.3f}")
print(t_class.round(3).to_string(index=False))
for r in t_count.itertuples():
    note(f"P1 coef {r.term}", r.coef_model3); note(f"P1 coef se {r.term}", r.coef_se)
    note(f"P1 mech {r.term}", r.mechanical); note(f"P1 mech share {r.term}", r.mechanical_share)
for r in t_class.itertuples():
    note(f"P1 mean w_hat sc{r.sectors}", r.mean_w_hat); note(f"P1 geo coverage sc{r.sectors}", r.geo_mean_coverage)
note("P1 ref0 n", ref0.sum()); note("P1 ref0 mean w_hat", w_hat[ref0].mean())
note("P1 ref1 n", ref1.sum()); note("P1 ref1 mean w_hat", w_hat[ref1].mean()); note("P1 complete mean w_hat", w_hat[complete].mean())

# (1b) w_hat in the regression: is its coefficient -1?
Qnc = ["core5_complete_flag", "co2eq", "lca", "mon"]
XA, nA = design(main.assign(w_hat=w_hat), BASE + Qnc + ["w_hat"], fe="country_code")
bA, VA, eA = ols_V(y, XA, cl)
iA = nA.index("w_hat")
lam, lam_se = bA[iA], np.sqrt(VA[iA, iA])
XB, nB = design(main.assign(w_hat=w_hat), BASE + Q + ["w_hat"], fe="country_code")
bB, VB, eB = ols_V(y, XB, cl)
idx_sc = [nB.index(c) for c in ["sc1", "sc2", "sc3", "sc4", "sc6"]]
Wd = float(bB[idx_sc] @ np.linalg.solve(VB[np.ix_(idx_sc, idx_sc)], bB[idx_sc]))
p_Wd = 1 - stats.chi2.cdf(Wd, len(idx_sc))
lamB, lamB_se = bB[nB.index("w_hat")], np.sqrt(VB[nB.index("w_hat"), nB.index("w_hat")])
XAp, nAp = design(main, BASE + Qnc + ["w_hat_pooled"], fe="country_code")
bAp, VAp, _ = ols_V(y, XAp, cl)
lam_p = bAp[nAp.index("w_hat_pooled")]; lam_p_se = np.sqrt(VAp[nAp.index("w_hat_pooled")] [nAp.index("w_hat_pooled")])
t_reg = pd.DataFrame([
    dict(model="w_hat replaces sector-count indicators", coef_w_hat=lam, se=lam_se, t_vs_minus1=(lam + 1) / lam_se,
         wald_sector_counts=np.nan, p_wald=np.nan, r2=1 - eA @ eA / np.sum((y - y.mean()) ** 2), skewness=stats.skew(eA)),
    dict(model="w_hat + sector-count indicators", coef_w_hat=lamB, se=lamB_se, t_vs_minus1=(lamB + 1) / lamB_se,
         wald_sector_counts=Wd, p_wald=p_Wd, r2=1 - eB @ eB / np.sum((y - y.mean()) ** 2), skewness=stats.skew(eB)),
    dict(model="pooled donors: w_hat replaces sector counts", coef_w_hat=lam_p, se=lam_p_se, t_vs_minus1=(lam_p + 1) / lam_p_se,
         wald_sector_counts=np.nan, p_wald=np.nan, r2=np.nan, skewness=np.nan)])
seB = np.sqrt(np.diag(VB))
sc_given_w = {c: (bB[nB.index(c)], seB[nB.index(c)]) for c in ["sc1", "sc2", "sc3", "sc4", "sc6"]}
print("\n(1b) regressions with the mechanical shortfall (NUTS3-clustered):")
print(t_reg.round(3).to_string(index=False))
note("P1 lambda (w_hat replaces counts)", lam); note("P1 lambda se", lam_se); note("P1 t(lambda=-1)", (lam + 1) / lam_se)
note("P1 lambda with counts", lamB); note("P1 lambda with counts se", lamB_se)
note("P1 Wald sector counts given w_hat", Wd); note("P1 p Wald", p_Wd)
note("P1 lambda pooled donors", lam_p); note("P1 lambda pooled donors se", lam_p_se)
print("     sector-count coefficients given w_hat:", "  ".join(f"{c}={v[0]:.3f} (t={v[0] / v[1]:.1f})" for c, v in sc_given_w.items()))
for c, v in sc_given_w.items():
    note(f"P1 {c} given w_hat", v[0]); note(f"P1 {c} given w_hat t", v[0] / v[1])
# subsample with four or more sectors; source of the identifying variation of w_hat
m4 = sc >= 4
X4s, n4s = design(main[m4].assign(w_hat=w_hat[m4]), BASE + Qnc + ["w_hat"], fe="country_code")
b4s, V4s, _ = ols_V(y[m4], X4s, cl[m4])
lam4, lam4_se = b4s[n4s.index("w_hat")], np.sqrt(V4s[n4s.index("w_hat"), n4s.index("w_hat")])
Xo = np.delete(XA, iA, axis=1)
r_w = w_hat - Xo @ np.linalg.lstsq(Xo, w_hat, rcond=None)[0]
lev12 = float(np.sum(r_w[sc <= 2] ** 2) / np.sum(r_w ** 2))
print(f"     four or more sectors: coefficient on w_hat {lam4:.3f} (SE {lam4_se:.3f}); "
      f"share of the identifying variation of w_hat from one- and two-sector inventories {lev12 * 100:.1f}%")
note("P1 lambda ge4", lam4); note("P1 lambda ge4 se", lam4_se); note("P1 leverage share 1-2 sectors", lev12)
# two-stage bootstrap: NUTS3 regions resampled within country + donors resampled within country (includes the sampling error of the generated regressor)
rngL = np.random.default_rng(SEED + 3)
clu_by_cty_L = {c: np.unique(cl[cty == c]) for c in np.unique(cty)}
rows_by_clu_L = {k: np.where(cl == k)[0] for k in np.unique(cl)}
donor_idx = {c: np.where(cty6 == c)[0] for c in donor_n.index}
combo_masks = [(r.s, r.c, np.array([ch == "1" for ch in r.s])) for r in combo.itertuples()]
key_idx = pd.Series(range(len(combo_masks)), index=pd.MultiIndex.from_tuples([(a, b_) for a, b_, _ in combo_masks]))
inv_key = key_idx.reindex(pd.MultiIndex.from_arrays([sets, cty])).values
lam_bs = []
for b_ in range(B_LAMBDA):
    di = np.concatenate([rngL.choice(ix, len(ix), replace=True) for ix in donor_idx.values()])
    sh_b, ct_b = share6[di], cty6[di]
    vals = np.empty(len(combo_masks))
    for j, (s_, c_, msk) in enumerate(combo_masks):
        if msk.all():
            vals[j] = 0.0
            continue
        D = sh_b if donor_n.get(c_, 0) < MIN_DONORS else sh_b[ct_b == c_]
        vals[j] = np.mean(-np.log(np.clip(D[:, msk].sum(1), 1e-6, None)))
    wb = vals[inv_key]
    idx = np.concatenate([rows_by_clu_L[k] for c, ks in clu_by_cty_L.items() for k in rngL.choice(ks, len(ks), replace=True)])
    Xb = XA[idx].copy(); Xb[:, iA] = wb[idx]
    lam_bs.append(np.linalg.lstsq(Xb, y[idx], rcond=None)[0][iA])
lam_bs = np.array(lam_bs)
print(f"     two-stage bootstrap ({B_LAMBDA}; NUTS3 clusters and donors): SE {lam_bs.std():.3f}, "
      f"95% interval [{np.percentile(lam_bs, 2.5):.3f}, {np.percentile(lam_bs, 97.5):.3f}]")
note("P1 lambda boot se", lam_bs.std()); note("P1 lambda boot lo", np.percentile(lam_bs, 2.5)); note("P1 lambda boot hi", np.percentile(lam_bs, 97.5))

# (1c) reported-sector sets (reference = all six sectors)
vc = pd.Series(sets).value_counts()
big = [s for s in vc.index if vc[s] >= MIN_SET and s != "111111"]
setcat = np.where(np.isin(sets, big + ["111111"]), sets, "other")
Dset = pd.get_dummies(pd.Series(setcat)).astype(float)
Dset = Dset[[s for s in big] + (["other"] if (setcat == "other").any() else [])]
XS0, nS0 = design(main, BASE + ["co2eq", "lca", "mon"], fe="country_code")
XS = np.column_stack([XS0, Dset.values]); nS = nS0 + list(Dset.columns)
bS, VS, eS = ols_V(y, XS, cl)
seS = np.sqrt(np.diag(VS))
t_sets = pd.DataFrame([dict(set=s, label=set_label(s), n=int(vc[s]), coef=bS[nS.index(s)], se=seS[nS.index(s)],
                            mechanical=-w_hat[sets == s].mean(), gap=bS[nS.index(s)] + w_hat[sets == s].mean())
                       for s in big]).sort_values("n", ascending=False).reset_index(drop=True)
wls = np.average(t_sets.coef - t_sets.mechanical, weights=t_sets.n)
slope = np.polyfit(t_sets.mechanical, t_sets.coef, 1, w=np.sqrt(t_sets.n))
corr = np.corrcoef(t_sets.mechanical, t_sets.coef)[0, 1]
print(f"\n(1c) reported-set coefficients (ref. all six sectors), sets with n >= {MIN_SET}: {len(t_sets)} sets, "
      f"{int(t_sets.n.sum())} inventories")
print(t_sets.round(3).to_string(index=False))
print(f"     weighted mean (coef - mechanical) = {wls:.3f}; slope of coef on mechanical = {slope[0]:.3f}; correlation = {corr:.3f}")
note("P1 sets n", len(t_sets)); note("P1 sets slope", slope[0]); note("P1 sets corr", corr); note("P1 sets mean gap", wls)

# Corollary 1: benchmarks from coverage-mixed peer groups
hq = main[complete].reset_index(drop=True)
qcols = BASE + [c for c in ["sc6", "co2eq", "lca", "mon"] if hq[c].nunique() > 1]
XQ, _ = design(hq, qcols, fe="country_code")
q_scr = XQ @ qreg(hq.lnE.values, XQ, 0.25)
XM, _ = design(main, BASE + ["co2eq", "lca", "mon"], fe="country_code")
q_mix_all = XM @ qreg(y, XM, 0.25)
q_mix = q_mix_all[complete]
Ehq = hq.total_emissions_reported_tonnes.values
yhq = hq.lnE.values


def agg_gap(q, yy, EE):
    return 1 - np.sum(EE * np.exp(np.minimum(q - yy, 0))) / EE.sum()


below_mix = y <= q_mix_all
inc_share_sample = float((~complete).mean())
inc_share_below = float((~complete)[below_mix].mean())
le2_share_sample = float((sc <= 2).mean()); le2_share_below = float((sc <= 2)[below_mix].mean())
gap_s = np.clip(1 - np.exp(q_scr - yhq), 0, None); gap_m = np.clip(1 - np.exp(q_mix - yhq), 0, None)
flag_s = gap_s > np.quantile(gap_s, 0.75); flag_m = gap_m > np.quantile(gap_m, 0.75)
t_bench = pd.DataFrame([
    dict(benchmark="complete reporters only (Section 5.6)", above_share_complete=float(np.mean(yhq > q_scr)),
         aggregate_gap_complete=agg_gap(q_scr, yhq, Ehq), mean_shift_vs_screened=0.0),
    dict(benchmark="all inventories, no coverage controls", above_share_complete=float(np.mean(yhq > q_mix)),
         aggregate_gap_complete=agg_gap(q_mix, yhq, Ehq), mean_shift_vs_screened=float(np.mean(q_mix - q_scr)))])
print("\nCorollary 1: tau = 0.25 benchmarks for the 3,108 complete reporters")
print(t_bench.round(3).to_string(index=False))
print(f"     at/below the mixed benchmark: incomplete inventories {inc_share_below * 100:.1f}% (sample {inc_share_sample * 100:.1f}%), "
      f"1-2 sectors {le2_share_below * 100:.1f}% (sample {le2_share_sample * 100:.1f}%); "
      f"top-quartile flag overlap {np.mean(flag_s & flag_m) / np.mean(flag_s) * 100:.1f}%")
note("C1 above share screened", t_bench.above_share_complete[0]); note("C1 above share mixed", t_bench.above_share_complete[1])
note("C1 agg gap screened", t_bench.aggregate_gap_complete[0]); note("C1 agg gap mixed", t_bench.aggregate_gap_complete[1])
note("C1 mean benchmark shift", t_bench.mean_shift_vs_screened[1])
note("C1 incomplete share at/below mixed", inc_share_below); note("C1 incomplete share sample", inc_share_sample)
note("C1 le2 share at/below mixed", le2_share_below); note("C1 le2 share sample", le2_share_sample)
note("C1 flag overlap", np.mean(flag_s & flag_m) / np.mean(flag_s))
cross = float(np.mean(q_mix > q_scr))
print(f"     the mixed benchmark lies above the complete-reporter benchmark for {cross * 100:.1f}% of complete reporters (fits cross)")
note("C1 crossing share", cross)

# =====================================================================================
# Proposition 2: third-moment decomposition and bootstrap
# =====================================================================================
groups = {"all": np.ones(len(main), bool), "complete reporters": complete, "incomplete": ~complete}
for c_ in range(1, 7):
    groups[f"{c_} sectors"] = sc == c_
groups["complete, Spain"] = complete & (cty == "es")
groups["complete, Italy"] = complete & (cty == "it")
# The shares summing to 1 is an identity for any partition; the substantive check is a zero within-class residual mean (third moment = central moment):
# true for the sector-count classes (they have indicators) and for complete/incomplete (the 'complete' indicator equals the core-five indicator).
assert np.array_equal(complete, main.core5_complete_flag.values == 1)
for g_, m in groups.items():   # check zero within-class residual means (complete reporters in Spain and Italy are excluded: not a class partition)
    if g_ not in ("complete, Spain", "complete, Italy"):
        assert abs(e3[m].mean()) < 1e-8, g_

rng = np.random.default_rng(SEED)
clu_by_cty = {c: np.unique(cl[cty == c]) for c in np.unique(cty)}
rows_by_clu = {k: np.where(cl == k)[0] for k in np.unique(cl)}
bs = {g_: [] for g_ in groups}
bs_share = {g_: [] for g_ in groups}
for b_ in range(B):
    idx = np.concatenate([rows_by_clu[k] for c, ks in clu_by_cty.items() for k in rng.choice(ks, len(ks), replace=True)])
    Xb, yb = X3[idx], y[idx]
    bb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
    eb = yb - Xb @ bb
    tot3 = np.sum(eb ** 3)
    for g_, m in groups.items():
        mb = m[idx]
        bs[g_].append(stats.skew(eb[mb]) if mb.sum() >= 3 else np.nan)
        bs_share[g_].append(np.sum(eb[mb] ** 3) / tot3)
rows = []
for g_, m in groups.items():
    s_ = np.array(bs[g_]); sh = np.array(bs_share[g_])
    rows.append(dict(group=g_, n=int(m.sum()), share_inventories=float(m.mean()), skewness=stats.skew(e3[m]),
                     ci_lo=np.nanpercentile(s_, 2.5), ci_hi=np.nanpercentile(s_, 97.5),
                     p_boot_skew_ge0=float(np.mean(s_ >= 0)), m3_share=m3_share(e3, m), m3_share_lo=np.nanpercentile(sh, 2.5),
                     m3_share_hi=np.nanpercentile(sh, 97.5), sd=float(e3[m].std())))
t_m3 = pd.DataFrame(rows)
print(f"\n(2a,b) residual skewness and contribution to the pooled third moment (model 3; {B} NUTS3-cluster bootstrap replications):")
print(t_m3.round(3).to_string(index=False))
le2 = (sc <= 2)
m3t_c = skew_tests(e3[complete])["m3t"]
e6 = e3[sc == 6]; e6_drop = np.delete(e6, np.argmin(e6))
print(f"     complete reporters: iid M3T = {m3t_c:.2f}; six-sector class skewness {stats.skew(e6):.2f}, "
      f"without its lowest residual ({e6.min():.2f}) {stats.skew(e6_drop):.2f}")
note("P2 M3T complete iid", m3t_c); note("P2 skew six", stats.skew(e6)); note("P2 skew six drop min", stats.skew(e6_drop)); note("P2 six min resid", e6.min())
note("P2 skew all", stats.skew(e3)); note("P2 m3 share 1-2 sectors", m3_share(e3, le2)); note("P2 share inventories 1-2 sectors", le2.mean())
for r in t_m3.itertuples():
    note(f"P2 skew {r.group}", r.skewness); note(f"P2 skew {r.group} lo", r.ci_lo); note(f"P2 skew {r.group} hi", r.ci_hi)
    note(f"P2 m3 share {r.group}", r.m3_share)
le2_bs = np.array(bs_share["1 sectors"]) + np.array(bs_share["2 sectors"])   # sum of the one- and two-sector shares within the same bootstrap replication
note("P2 m3 share 1-2 sectors lo", np.percentile(le2_bs, 2.5)); note("P2 m3 share 1-2 sectors hi", np.percentile(le2_bs, 97.5))
print(f"     1-2 sector inventories: {le2.sum()} ({le2.mean() * 100:.1f}%) contribute {m3_share(e3, le2) * 100:.1f}% of the third moment "
      f"[{np.percentile(le2_bs, 2.5) * 100:.1f}, {np.percentile(le2_bs, 97.5) * 100:.1f}]")

# (2c) skewness implied by the fitted exponential two-tier models: kappa3 = 2 su^3 - 2 sw^3, var = sv^2 + su^2 + sw^2
oc = tt["out"]["c"]
k3c = 2 * oc["su"] ** 3 - 2 * oc["sw"] ** 3; v2c = oc["sv"] ** 2 + oc["su"] ** 2 + oc["sw"] ** 2
skew_c = k3c / v2c ** 1.5
od = tt["out"]["d"]
Zw = main[ZW_COLS].values.astype(float)
sw_i = np.exp(np.log(od["sw"]) + Zw @ od["dw"])
k3d = np.mean(2 * od["su"] ** 3 - 2 * sw_i ** 3); v2d = np.mean(od["sv"] ** 2 + od["su"] ** 2 + sw_i ** 2)
skew_d = k3d / v2d ** 1.5
t_tt = pd.DataFrame([dict(model="two-tier (c): constant scales", sigma_v=oc["sv"], sigma_u=oc["su"], sigma_w=oc["sw"], implied_skewness=skew_c),
                     dict(model="two-tier (d): sigma_w by coverage", sigma_v=od["sv"], sigma_u=od["su"], sigma_w=float(np.mean(sw_i)),
                          implied_skewness=skew_d),
                     dict(model="observed (model 3 residuals)", sigma_v=np.nan, sigma_u=np.nan, sigma_w=np.nan, implied_skewness=stats.skew(e3))])
print("\n(2c) skewness implied by the fitted exponential two-tier models:")
print(t_tt.round(3).to_string(index=False))
note("P2 two-tier c implied skew", skew_c); note("P2 two-tier d implied skew", skew_d)

# (2d) inventories removed by trimming the lowest 1%
cut = np.quantile(e3, 0.01)
low = e3 <= cut
t_trim = pd.DataFrame([dict(sectors=c_, share_in_lowest_1pct=float((sc[low] == c_).mean()), share_in_sample=float((sc == c_).mean()))
                       for c_ in range(1, 7)])
t_trim.loc[len(t_trim)] = dict(sectors=0, share_in_lowest_1pct=float(complete[low].mean()), share_in_sample=float(complete.mean()))
print(f"\n(2d) composition of the lowest 1% residuals ({low.sum()} inventories) by number of sectors (0 = complete reporters):")
print(t_trim.round(3).to_string(index=False))
note("P2 lowest1pct n", low.sum()); note("P2 lowest1pct share le2", float((sc[low] <= 2).mean()))
note("P2 lowest1pct share le3", float((sc[low] <= 3).mean())); note("P2 lowest1pct share complete", float(complete[low].mean()))

# (2e) residual skewness under alternative coverage controls
X0, _ = design(main, BASE + ["co2eq", "lca", "mon"], fe="country_code")
e0 = ols(y, X0)["e"]
t_alt = pd.DataFrame([dict(coverage_control="none", skewness=stats.skew(e0), m3t=skew_tests(e0)["m3t"]),
                      dict(coverage_control="sector-count indicators (model 3)", skewness=stats.skew(e3), m3t=skew_tests(e3)["m3t"]),
                      dict(coverage_control="mechanical shortfall w_hat", skewness=stats.skew(eA), m3t=skew_tests(eA)["m3t"]),
                      dict(coverage_control=f"reported-set indicators (n >= {MIN_SET}; others pooled)", skewness=stats.skew(eS),
                           m3t=skew_tests(eS)["m3t"])])
Dfull = pd.get_dummies(pd.Series(sets), drop_first=True).astype(float)
eF = ols(y, np.column_stack([X0, Dfull.values]))["e"]
t_alt.loc[len(t_alt)] = dict(coverage_control="all reported-set indicators", skewness=stats.skew(eF), m3t=skew_tests(eF)["m3t"])
print("\n(2e) residual skewness under alternative coverage controls:")
print(t_alt.round(3).to_string(index=False))
for r in t_alt.itertuples():
    note(f"P2 skew control {r.coverage_control}", r.skewness)

# =====================================================================================
# Proposition 3: exact decomposition of changes
# =====================================================================================
d = rep["pairs"].copy().reset_index(drop=True)
assert d.dR.notna().all(), "every pair should share at least one sector"
d["scope"] = d.dT - d.dR
s1 = rep["single"]
d1 = d[(d.n_rem == 1) & (d.n_add == 0)]
ok = s1["share"].values < 1 - 1e-6
err = np.max(np.abs(d1.scope.values[ok] - s1.mech.values[ok]))
print(f"\n(3a) single removals: max |scope - ln(1 - s)| = {err:.2e} over {ok.sum()} pairs (identity)")
assert err < 1e-8
cc = d.cat == "same coverage"
assert np.max(np.abs(d.scope[cc])) < 1e-10
# identity check in all pairs: scope term = ln(1 - rho-) - ln(1 - rho+), with rho computed from the sector data
bm = main[main.GCoM_ID.isin(d.id)]
B_ = bm[bm.mon == 0].set_index("GCoM_ID").reindex(d.id); M_ = bm[bm.mon == 1].set_index("GCoM_ID").reindex(d.id)
sb = np.array([[ch == "1" for ch in s_] for s_ in B_.sector_set]); sm = np.array([[ch == "1" for ch in s_] for s_ in M_.sector_set])
Eb = np.nan_to_num(B_[SECTOR_COLS].values.astype(float)); Em = np.nan_to_num(M_[SECTOR_COLS].values.astype(float))
rho_m = (Eb * (sb & ~sm)).sum(1) / B_.total_emissions_reported_tonnes.values
rho_p = (Em * (sm & ~sb)).sum(1) / M_.total_emissions_reported_tonnes.values
err_all = np.max(np.abs(d.scope.values - (np.log(1 - rho_m) - np.log(1 - rho_p))))
print(f"     all {len(d)} pairs: max |scope - [ln(1 - rho-) - ln(1 - rho+)]| = {err_all:.2e}")
assert err_all < 1e-8
note("P3 identity max error", err_all)
CATS = ["addition only", "removal only", "both"]


def decompose(dd, ann=False):
    f = (100 / dd.dy) if ann else 1.0
    T, R_, S_ = dd.dT * f, dd.dR * f, dd.scope * f
    ccm = dd.cat.values == "same coverage"
    mean_cc = T[ccm].mean(); N = len(dd)
    out = dict(mean_all=T.mean(), mean_cc=mean_cc, inflation=T.mean() - mean_cc, common_sector_all=R_.mean())
    out["scope_part"] = sum(S_[dd.cat.values == c].sum() for c in CATS) / N
    out["revision_part"] = sum((R_[dd.cat.values == c] - mean_cc).sum() for c in CATS) / N
    for c in CATS:
        m = dd.cat.values == c
        out[f"scope_{c}"] = S_[m].sum() / N
        out[f"revision_{c}"] = (R_[m] - mean_cc).sum() / N
    return out


res = {"log": decompose(d), "annual": decompose(d, ann=True)}
for k_, r in res.items():
    assert abs(r["scope_part"] + r["revision_part"] - r["inflation"]) < 1e-10
rngp = np.random.default_rng(SEED + 2)
pc_by_cty = {c: np.unique(d.nuts3[d.cty == c]) for c in d.cty.unique()}
prow = {k: np.where(d.nuts3.values == k)[0] for k in d.nuts3.unique()}
bsd = {"log": [], "annual": []}
for b_ in range(B):
    idx = np.concatenate([prow[k] for c, ks in pc_by_cty.items() for k in rngp.choice(ks, len(ks), replace=True)])
    db = d.iloc[idx]
    bsd["log"].append(decompose(db)); bsd["annual"].append(decompose(db, ann=True))
rows = []
for k_ in ("log", "annual"):
    bdf = pd.DataFrame(bsd[k_])
    for item in ["mean_all", "mean_cc", "inflation", "scope_part", "revision_part", "common_sector_all"] + \
                [f"{p}_{c}" for p in ("scope", "revision") for c in CATS]:
        rows.append(dict(scale=k_, item=item, value=res[k_][item], ci_lo=bdf[item].quantile(0.025), ci_hi=bdf[item].quantile(0.975)))
t_inf = pd.DataFrame(rows)
print("\n(3b,c) decomposition of the pooled-mean change relative to constant coverage (NUTS3-cluster bootstrap 95% CI):")
print(t_inf.round(4).to_string(index=False))
for r in t_inf.itertuples():
    note(f"P3 {r.scale} {r.item}", r.value); note(f"P3 {r.scale} {r.item} lo", r.ci_lo); note(f"P3 {r.scale} {r.item} hi", r.ci_hi)
for k_ in ("log", "annual"):
    r = res[k_]
    print(f"   [{k_}] inflation {r['inflation']:.3f}: scope {r['scope_part']:.3f} ({r['scope_part'] / r['inflation'] * 100:.0f}%), "
          f"revision {r['revision_part']:.3f} ({r['revision_part'] / r['inflation'] * 100:.0f}%); "
          f"scope-corrected pooled mean {r['common_sector_all']:.3f} vs constant coverage {r['mean_cc']:.3f}")
    note(f"P3 {k_} revision share of inflation", r["revision_part"] / r["inflation"])

t_cat = d.groupby("cat").agg(n=("dT", "size"), observed=("dT", "mean"), common_sectors=("dR", "mean"), scope=("scope", "mean"),
                             median_years=("dy", "median")).reindex(["same coverage"] + CATS).reset_index()
t_cat["excess_common_vs_constant"] = t_cat.common_sectors - t_cat.observed[0]
t_cat["mean_years"] = d.groupby("cat").dy.mean().reindex(t_cat.cat).values
print("\nmean change by coverage-change category (observed = common sectors + scope):")
print(t_cat.round(3).to_string(index=False))
for r in t_cat.itertuples():
    note(f"P3 cat {r.cat} observed", r.observed); note(f"P3 cat {r.cat} common", r.common_sectors); note(f"P3 cat {r.cat} scope", r.scope)
    note(f"P3 cat {r.cat} mean years", r.mean_years)

# =====================================================================================
# Figure_M1 = Fig. 2 (Propositions 1-2), Figure_M2 = Fig. 4 (Proposition 3)
# =====================================================================================
fig, ax = plt.subplots(1, 3, figsize=(FIG_W, 2.85), gridspec_kw=dict(width_ratios=[1.05, 1.1, 0.95]))
a = ax[0]
lim = [min(t_sets.mechanical.min(), t_sets.coef.min()) - 0.25, 0.35]
a.plot(lim, lim, color="0.6", lw=0.8, ls="--", zorder=0)
a.errorbar(t_sets.mechanical, t_sets.coef, yerr=1.96 * t_sets.se, fmt="none", ecolor="0.7", lw=0.7, zorder=1)
a.scatter(t_sets.mechanical, t_sets.coef, s=8 + 60 * np.sqrt(t_sets.n / t_sets.n.max()), color="#1f77b4", zorder=2)
OFFS = {"111110": (4, 4), "111011": (6, -12), "111010": (5, -4), "101010": (5, -4), "001010": (5, 2),
        "000010": (5, -2), "101000": (4, -10), "111101": (5, -4)}
for r in t_sets.itertuples():
    if r.set in OFFS:
        dx, dy = OFFS[r.set]
        a.annotate(r.label, (r.mechanical, r.coef), fontsize=5.5, xytext=(dx, dy), textcoords="offset points",
                   ha="right" if dx < 0 else "left")
a.set_xlim(lim); a.set_ylim(lim)
a.set_xlabel("share-implied coverage effect, $-\\bar{w}$")
a.set_ylabel("estimated reported-set effect")
a.set_title("(a) Levels: reported sets", fontsize=8)
b_ = ax[1]
order = ["1 sectors", "2 sectors", "3 sectors", "4 sectors", "5 sectors", "6 sectors", "incomplete", "complete reporters"]
lab = {"1 sectors": "1 sector", "complete reporters": "complete"}
tm = t_m3.set_index("group").loc[order]
yy = np.arange(len(order))[::-1]
b_.errorbar(tm.skewness, yy, xerr=[tm.skewness - tm.ci_lo, tm.ci_hi - tm.skewness], fmt="o", ms=3, color="#d62728", ecolor="0.5", lw=0.8)
b_.axvline(0, color="k", lw=0.7)
b_.set_yticks(yy); b_.set_yticklabels([f"{lab.get(g, g)} ({int(tm.n[g])})" for g in order])
b_.set_xlabel("residual skewness (95% cluster-bootstrap CI)")
b_.set_title("(b) Residual skewness", fontsize=8)
c_ = ax[2]
cats_ = [f"{k} sectors" for k in range(1, 7)]
w_ = 0.38
xx = np.arange(6)
c_.bar(xx - w_ / 2, [t_m3.set_index("group").share_inventories[g] for g in cats_], w_, color="0.65", label="share of inventories")
c_.bar(xx + w_ / 2, [t_m3.set_index("group").m3_share[g] for g in cats_], w_, color="#d62728", label="share of third moment")
c_.set_xticks(xx); c_.set_xticklabels([str(k) for k in range(1, 7)])
c_.set_xlabel("number of reported sectors"); c_.set_ylabel("share")
c_.axhline(0, color="k", lw=0.5)
c_.set_ylim(-0.02, 0.72)
c_.legend(frameon=False, loc="upper right")
c_.set_title("(c) Third moment by coverage", fontsize=8)
plt.tight_layout()
save_figure(fig, FIG_FW, "Figure_M1", tight=True)

fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
a = ax[0]
cats4 = ["same coverage"] + CATS
tc = t_cat.set_index("cat").loc[cats4]
xx = np.arange(4)
a.bar(xx - 0.2, tc.common_sectors, 0.38, color="#1f77b4", label="sectors in both inventories")
a.bar(xx + 0.2, tc.scope, 0.38, color="#ff7f0e", label="scope term")
a.scatter(xx, tc.observed, marker="D", color="k", s=14, zorder=3, label="observed total")
a.axhline(tc.observed["same coverage"], color="0.4", ls="--", lw=0.8, label="constant-coverage mean")
a.axhline(0, color="k", lw=0.6)
a.set_xticks(xx); a.set_xticklabels([f"{c}\n(n={int(tc.n[c])})" for c in cats4], fontsize=6.5)
a.set_ylabel("mean change in ln reported emissions")
a.legend(frameon=False, fontsize=6, loc="lower left")
a.set_title("(a) Observed change = common sectors + scope", fontsize=8)
b_ = ax[1]
ra = res["annual"]
steps_ = [("constant\ncoverage", ra["mean_cc"], "0.5"), ("+ scope\nterms", ra["scope_part"], "#ff7f0e"),
          ("+ retained-\nsector terms", ra["revision_part"], "#1f77b4"), ("pooled\nall pairs", ra["mean_all"], "k")]
cum = 0.0
for j, (nm, v, col) in enumerate(steps_):
    if j in (0, 3):                       # first (constant coverage) and last (all pairs) bars are drawn from 0
        b_.bar(j, v, 0.6, color=col); ymid = v / 2
        if j == 0: cum = v
        b_.text(j, ymid, f"{v:+.2f}", ha="center", va="center", fontsize=6.5, color="white")
    else:                                 # scope and retained-sector (revision_part) terms are stacked on the previous bar
        b_.bar(j, v, 0.6, bottom=cum, color=col); ymid = cum + v / 2; cum += v
        b_.text(j, ymid, f"{v:+.2f}", ha="center", va="center", fontsize=6.5, color="k")
b_.axhline(0, color="k", lw=0.6)
b_.set_xticks(range(4)); b_.set_xticklabels([s[0] for s in steps_], fontsize=6.5)
b_.set_ylabel("mean annualised change (% per year)")
b_.set_title("(b) Pooled versus constant-coverage mean", fontsize=8)
plt.tight_layout()
save_figure(fig, FIG_FW, "Figure_M2", tight=True)

# =====================================================================================
# save outputs
# =====================================================================================
save_obj("framework.pkl", dict(t_count=t_count, t_class=t_class, lam_bs=lam_bs, t_reg=t_reg, t_sets=t_sets, t_bench=t_bench, t_m3=t_m3, t_tt=t_tt, t_trim=t_trim,
                               t_alt=t_alt, t_inf=t_inf, t_cat=t_cat, w_hat=w_hat, nums=pd.DataFrame(nums)))
with pd.ExcelWriter(os.path.join(TAB_DIR, "Framework_Tables.xlsx")) as xw:
    for nm, df in [("P1_coefficients", t_count), ("P1_classes", t_class), ("P1_regressions", t_reg), ("P1_sets", t_sets), ("C1_benchmarks", t_bench),
                   ("P2_third_moment", t_m3), ("P2_two_tier_implied", t_tt), ("P2_trim", t_trim), ("P2_alt_controls", t_alt),
                   ("P3_categories", t_cat), ("P3_inflation", t_inf)]:
        df.to_excel(xw, sheet_name=nm, index=False)
pd.DataFrame(nums).to_csv(os.path.join(TAB_DIR, "framework_numbers.csv"), index=False)
print("saved: work/framework.pkl, tables/Framework_Tables.xlsx, tables/framework_numbers.csv, figures/framework/Figure_M1, Figure_M2")
