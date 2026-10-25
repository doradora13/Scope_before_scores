# -*- coding: utf-8 -*-
"""
22_robustness_checks.py - robustness checks (Supplementary Section S16): numbers and LaTeX for Tables S20-S22.

(R1) Regional sensitivity of the Proposition 3 decomposition
     - decomposition without Andalucia (ES61) (log and annualised) with NUTS3-cluster bootstrap intervals (resampling within countries, B replications)
     - leave-one-NUTS2-out decomposition: retained-sector share after dropping each NUTS2 region with at least MIN_PAIRS pairs
(R2) Table 3 Panel B (regression of the retained-sector change on category indicators + elapsed years + country effects, NUTS3 clusters) without ES61,
     with NUTS2 instead of country fixed effects (comparison with constant-coverage pairs of the same region), and for ES61 only;
     Table 3 Panel C (observed vs share-implied change for the 329 single-sector removals) split into ES61 and the other regions
(R2d) Removal events by sector (Table S6) and how many of them are in ES61
(R3) Inference unit: bootstrap intervals resampling NUTS2 regions within countries (difference, scope term, retained-sector term, share; all pairs)
(R4) Annualisation sensitivity: excluding one-year pairs, excluding pairs of 3 years or less
(R5) Donor robustness for Proposition 1: NUTS2 composition of the donors, w_hat coefficient without both the ES61 inventories and the ES61 donors,
     and in the same sample the skewness of complete reporters and the third-moment share of one- and two-sector inventories
(R6) Source of the identifying variation for Proposition 1: share of the residualised variation of w_hat by reported-sector set; coefficients without the most influential sets
(R7) Unit of the benchmark flags: municipalities and monitoring inventories behind the 3,108 complete-reporter inventories and the 518 flags
(R8) Monitoring inventories without a pair (reported for the same year as the baseline)
(R9) Residual skewness of Eq. (10) by specification (Table 2 model (1), the two specifications of the stage-21 simulation, Table S16 Panel C, Table 2 model (3))
Input : work/data.pkl, work/repeated.pkl, work/framework.pkl, work/benchmarks.pkl
Output: work/robustness.pkl, tables/Robustness_Tables.xlsx, tables/robustness_numbers.csv, tables/Tables_S20_S22.tex (LaTeX for Tables S20-S22)
"""
from lib import *

B, SEED = 999, 22          # bootstrap replications and random seed
MIN_DONORS = 20            # same donor rule as stage 20 (pooled donors when a country has fewer six-sector donors than this)
MIN_PAIRS = 20             # minimum number of pairs for a region in the leave-one-NUTS2-out check
FOCUS = "ES61"             # Andalucia
CATS = ["addition only", "removal only", "both"]

banner("22 robustness checks: regional concentration, inference unit, annualisation, donors and leverage")
main = load_obj("data.pkl")["main"].reset_index(drop=True)
rep = load_obj("repeated.pkl")
fw = load_obj("framework.pkl")
bm = load_obj("benchmarks.pkl")
nums = []


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


# =====================================================================================
# (R1)-(R4) Proposition 3: scope term and retained-sector term
# =====================================================================================
d = rep["pairs"].copy().reset_index(drop=True)
d["scope"] = d.dT - d.dR
d["nuts2"] = d.nuts3.astype(str).str[:4]


def decompose(dd, ann=True):
    f = (100 / dd.dy) if ann else 1.0
    T, R_, S_ = dd.dT * f, dd.dR * f, dd.scope * f
    ccm = dd.cat.values == "same coverage"
    mean_cc = T[ccm].mean(); N = len(dd)
    scope = sum(S_[dd.cat.values == c].sum() for c in CATS) / N
    ret = sum((R_[dd.cat.values == c] - mean_cc).sum() for c in CATS) / N
    return dict(pairs=N, changed=int((~ccm).sum()), mean_all=T.mean(), mean_cc=mean_cc, difference=T.mean() - mean_cc,
                scope_part=scope, retained_part=ret, retained_share=ret / (T.mean() - mean_cc),
                scope_corrected_mean=R_.mean(),
                retained_removal=(R_[dd.cat.values == "removal only"] - mean_cc).sum() / N)


def boot(dd, unit="nuts3", ann=True, seed=SEED):
    """Bootstrap that resamples regions (unit) with replacement within countries."""
    rng = np.random.default_rng(seed)
    by_c = {c: np.unique(dd[unit][dd.cty == c]) for c in dd.cty.unique()}
    rows = {k: np.where(dd[unit].values == k)[0] for k in dd[unit].unique()}
    out = []
    for _ in range(B):
        idx = np.concatenate([rows[k] for c, ks in by_c.items() for k in rng.choice(ks, len(ks), replace=True)])
        out.append(decompose(dd.iloc[idx], ann))
    return pd.DataFrame(out)


es = (d.nuts2 == FOCUS).values
chg = (d.cat != "same coverage").values
print(f"pairs {len(d)}; {FOCUS}: {es.sum()} pairs ({es.mean():.1%}), {(es & chg).sum()} of {chg.sum()} changed-coverage pairs, "
      f"{(es & (d.cat == 'removal only').values).sum()} of {(d.cat == 'removal only').sum()} removal-only pairs")
note("R1 pairs", len(d)); note("R1 ES61 pairs", es.sum()); note("R1 changed pairs", chg.sum()); note("R1 ES61 changed pairs", (es & chg).sum())
note("R1 removal pairs", (d.cat == "removal only").sum()); note("R1 ES61 removal pairs", (es & (d.cat == "removal only").values).sum())

rows = []
for lab, dd in [("all pairs", d), (f"without {FOCUS}", d[~es].reset_index(drop=True)), (f"{FOCUS} only", d[es].reset_index(drop=True))]:
    for scale, ann in [("annual", True), ("log", False)]:
        r = decompose(dd, ann)
        bt = boot(dd, "nuts3", ann)
        for k in ["mean_all", "mean_cc", "difference", "scope_part", "retained_part", "retained_share", "scope_corrected_mean", "retained_removal"]:
            rows.append(dict(sample=lab, scale=scale, item=k, value=r[k], ci_lo=bt[k].quantile(0.025), ci_hi=bt[k].quantile(0.975),
                             p_ge0=float((bt[k] >= 0).mean()) if k == "retained_part" else np.nan, pairs=r["pairs"], changed=r["changed"]))
t_es = pd.DataFrame(rows)
print("\n(R1) decomposition with and without ES61 (NUTS3-cluster bootstrap 95% CI):")
print(t_es[t_es.item.isin(["difference", "scope_part", "retained_part", "retained_share"])].round(3).to_string(index=False))
for r in t_es.itertuples():
    note(f"R1 {r.sample} {r.scale} {r.item}", r.value); note(f"R1 {r.sample} {r.scale} {r.item} lo", r.ci_lo)
    note(f"R1 {r.sample} {r.scale} {r.item} hi", r.ci_hi)

vc2 = d.nuts2.value_counts()
regions = vc2[vc2 >= MIN_PAIRS].index
t_loo = pd.DataFrame([dict(nuts2=k, pairs=int(vc2[k]), changed=int(((d.nuts2 == k) & chg).sum()),
                           **{f"{s}_{x}": decompose(d[d.nuts2 != k], ann)[x] for s, ann in [("annual", True), ("log", False)]
                              for x in ["difference", "retained_part", "retained_share"]})
                      for k in regions]).sort_values("annual_retained_share").reset_index(drop=True)
others = t_loo[t_loo.nuts2 != FOCUS]
print(f"\n(R1) leave-one-NUTS2-out (regions with >= {MIN_PAIRS} pairs, {len(t_loo)} regions): retained share (annual) without "
      f"{FOCUS} {t_loo.loc[t_loo.nuts2 == FOCUS, 'annual_retained_share'].iloc[0]:.0%}; other regions "
      f"{others.annual_retained_share.min():.0%}-{others.annual_retained_share.max():.0%}")
note("R1 loo regions", len(t_loo)); note("R1 loo other min share", others.annual_retained_share.min())
note("R1 loo other max share", others.annual_retained_share.max())

# (R2) Table 3 Panel B without ES61; (R2b) NUTS2 instead of country fixed effects (comparison with constant-coverage pairs of the same region), ES61 only
rows = []
for lab, dd, fe in [("all pairs", d, "cty"), (f"without {FOCUS}", d[~es], "cty"), ("all pairs, NUTS2 effects", d, "nuts2"),
                    (f"without {FOCUS}, NUTS2 effects", d[~es], "nuts2"), (f"{FOCUS} only", d[es], None)]:
    dd = dd.dropna(subset=["dR"]).copy()
    for c in CATS:
        dd[c] = (dd.cat == c).astype(int)
    X, nm = design(dd, CATS + ["dy"], fe=fe)
    with np.errstate(invalid="ignore"):   # standard errors of the fixed-effect columns are not used
        o = ols(dd.dR.values, X, cluster=dd.nuts3.values)
    for v in CATS + ["dy"]:
        rows.append(dict(sample=lab, effects={"cty": "country", "nuts2": "NUTS2", None: "none"}[fe], var=v, coef=o["b"][nm.index(v)],
                         se=o["se"][nm.index(v)], t=o["t"][nm.index(v)], n=o["n"], r2=o["r2"]))
t_pb = pd.DataFrame(rows)
print("\n(R2) Table 3 Panel B (retained-sector change relative to constant coverage): without ES61, NUTS2 effects, ES61 only:")
print(t_pb.round(3).to_string(index=False))
for r in t_pb.itertuples():
    note(f"R2 {r.sample} {r.var} coef", r.coef); note(f"R2 {r.sample} {r.var} t", r.t)

# (R2c) Table 3 Panel C (single-sector removals: observed vs share-implied change), split into ES61 and the other regions
s1 = rep["single"].reset_index(drop=True)
d1 = d[(d.n_rem == 1) & (d.n_add == 0)].reset_index(drop=True)
ok1 = s1["share"].values < 1 - 1e-6
assert len(s1) == len(d1) and np.max(np.abs(d1.scope.values[ok1] - s1.mech.values[ok1])) < 1e-8, "single removals do not align with pairs"
s1["region"] = np.where(d1.nuts2.values == FOCUS, FOCUS, "other regions")
rows = []
for reg, g in [("all", s1), (f"without {FOCUS}", s1[s1.region != FOCUS]), (f"{FOCUS} only", s1[s1.region == FOCUS])]:
    for sec, gg in list(g.groupby("sector")) + [("all single removals", g)]:
        rows.append(dict(sample=reg, sector=sec, pairs=len(gg), share=gg.share.mean(), observed=gg.obs.mean(), implied=gg.mech.mean(),
                         ratio=gg.obs.mean() / gg.mech.mean()))
t_single = pd.DataFrame(rows)
print("\n(R2c) Table 3 Panel C by region (single-sector removals):")
print(t_single[t_single.sector.isin(["waste", "transport", "all single removals"])].round(3).to_string(index=False))
for r in t_single.itertuples():
    note(f"R2c {r.sample} {r.sector} pairs", r.pairs); note(f"R2c {r.sample} {r.sector} observed", r.observed)
    note(f"R2c {r.sample} {r.sector} implied", r.implied); note(f"R2c {r.sample} {r.sector} ratio", r.ratio)

# (R2d) removal events by sector (Table S6) and those in ES61
ev = pd.DataFrame([(sec, nm2 == FOCUS) for r_, nm2 in zip(d["rem"], d.nuts2) for sec in [x.strip() for x in r_.split(",") if x.strip()]],
                  columns=["sector", "es"])
t_rem = ev.groupby("sector").agg(removals=("es", "size"), in_focus=("es", "sum")).sort_values("removals", ascending=False).reset_index()
print("\n(R2d) removal events by sector (Table S6) and how many are in", FOCUS)
print(t_rem.to_string(index=False))
for r in t_rem.itertuples():
    note(f"R2d removals {r.sector}", r.removals); note(f"R2d removals {r.sector} in ES61", r.in_focus)

# (R3) NUTS2-cluster bootstrap (all pairs)
bt2 = boot(d, "nuts2", True, seed=SEED + 1)
r3 = dict(lo=bt2.retained_part.quantile(0.025), hi=bt2.retained_part.quantile(0.975), p_ge0=float((bt2.retained_part >= 0).mean()),
          diff_lo=bt2.difference.quantile(0.025), diff_hi=bt2.difference.quantile(0.975),
          scope_lo=bt2.scope_part.quantile(0.025), scope_hi=bt2.scope_part.quantile(0.975),
          share_lo=bt2.retained_share.quantile(0.025), share_hi=bt2.retained_share.quantile(0.975))
print(f"\n(R3) NUTS2-cluster bootstrap, all pairs: retained term [{r3['lo']:.2f}, {r3['hi']:.2f}] (share of replications >= 0: "
      f"{r3['p_ge0']:.1%}); difference [{r3['diff_lo']:.2f}, {r3['diff_hi']:.2f}]")
for k, v in r3.items():
    note(f"R3 nuts2 {k}", v)

# (R4) sensitivity to elapsed years
rows = []
for lab, dd in [("all pairs", d), ("without 1-year pairs", d[d.dy > 1]), ("without pairs of 3 years or less", d[d.dy > 3])]:
    r = decompose(dd)
    rows.append(dict(sample=lab, **{k: r[k] for k in ["pairs", "changed", "mean_all", "mean_cc", "difference", "scope_part", "retained_part",
                                                      "retained_share"]}))
t_gap = pd.DataFrame(rows)
print("\n(R4) annualisation sensitivity (minimum elapsed years):")
print(t_gap.round(3).to_string(index=False))
note("R4 one-year pairs", (d.dy == 1).sum())
for r in t_gap.itertuples():
    for k in ["pairs", "mean_all", "mean_cc", "difference", "retained_part", "retained_share"]:
        note(f"R4 {r.sample} {k}", getattr(r, k))

# =====================================================================================
# (R5)-(R6) Proposition 1: donors and identifying variation
# =====================================================================================
y = main.lnE.values; cl = main.nuts3.values; cty = main.country_code.values; sets = main.sector_set.values; sc = main.sc.values
n2 = main.nuts3.astype(str).str[:4].values
six = sc == 6
E6 = main[SECTOR_COLS].values.astype(float)
donors = pd.DataFrame(dict(country=cty[six], nuts2=n2[six]))
dn2 = donors.groupby(["country", "nuts2"]).size().sort_values(ascending=False).rename("donors").reset_index()
print(f"\n(R5) donors: {six.sum()}; in {FOCUS}: {(donors.nuts2 == FOCUS).sum()} ({(donors.nuts2 == FOCUS).mean():.0%}); "
      f"Spanish donors {int((donors.country == 'es').sum())}, of which {FOCUS} {int(((donors.country == 'es') & (donors.nuts2 == FOCUS)).sum())}")
note("R5 donors", six.sum()); note("R5 ES61 donors", (donors.nuts2 == FOCUS).sum()); note("R5 Spanish donors", (donors.country == "es").sum())

Qnc = ["core5_complete_flag", "co2eq", "lca", "mon"]
w_full = np.asarray(fw["w_hat"], float)


def w_hat_from(keep_donor):
    """w_hat using as donors only the six-sector inventories where keep_donor is True (same rule as stage 20)."""
    don = six & keep_donor
    sh = E6[don] / E6[don].sum(1, keepdims=True); cd = cty[don]; dn = pd.Series(cd).value_counts()
    cache = {}
    out = np.empty(len(main))
    for i, (s, c) in enumerate(zip(sets, cty)):
        msk = np.array([ch == "1" for ch in s])
        if msk.all():
            out[i] = 0.0
            continue
        key = (s, c if dn.get(c, 0) >= MIN_DONORS else "pool")
        if key not in cache:
            D = sh if key[1] == "pool" else sh[cd == c]
            cache[key] = float(np.mean(-np.log(np.clip(D[:, msk].sum(1), 1e-6, None))))
        out[i] = cache[key]
    return out


def lam_fit(keep, w):
    m = main[keep].reset_index(drop=True)
    X, nm = design(m.assign(w_hat=w[keep]), BASE + Qnc + ["w_hat"], fe="country_code")
    b, V, _ = ols_V(y[keep], X, cl[keep])
    j = nm.index("w_hat")
    return b[j], np.sqrt(V[j, j]), int(keep.sum())


rows = []
allk = np.ones(len(main), bool)
lam0 = lam_fit(allk, w_full)
rows.append(dict(spec="all inventories (Table S14, Panel C)", coef=lam0[0], se=lam0[1], n=lam0[2]))
noes = n2 != FOCUS
w_noes = w_hat_from(noes)
r_ = lam_fit(noes, w_noes)
rows.append(dict(spec=f"without {FOCUS} inventories and donors", coef=r_[0], se=r_[1], n=r_[2]))
for lab, keep in [("without one-sector municipal-only inventories", sets != "100000"),
                  ("without municipal-only and municipal+waste inventories", ~np.isin(sets, ["100000", "100001"])),
                  ("three or more sectors", sc >= 3), ("four or more sectors (Section 6.1)", sc >= 4)]:
    r_ = lam_fit(keep, w_full)
    rows.append(dict(spec=lab, coef=r_[0], se=r_[1], n=r_[2]))
t_lam = pd.DataFrame(rows)
print("\n(R5-R6) coefficient on w_hat (NUTS3-clustered):")
print(t_lam.round(3).to_string(index=False))
for i, r in enumerate(t_lam.itertuples()):
    note(f"R5 lambda {i} {r.spec}", r.coef); note(f"R5 lambda {i} se", r.se); note(f"R5 lambda {i} n", r.n)

# Proposition 2 quantities (model (3) residuals): full sample and sample without ES61
def p2_of(keep):
    m_ = main[keep].reset_index(drop=True)
    X3, _ = design(m_, BASE + Q, fe="country_code")
    e3 = y[keep] - X3 @ np.linalg.lstsq(X3, y[keep], rcond=None)[0]
    comp = ((m_.sc >= 5) & (m_.core5_complete_flag == 1)).values
    le2 = m_.sc.values <= 2
    return dict(n=int(keep.sum()), skew_all=stats.skew(e3), skew_complete=stats.skew(e3[comp]), skew_incomplete=stats.skew(e3[~comp]),
                m3_share_le2=float(np.sum(e3[le2] ** 3) / np.sum(e3 ** 3)), share_le2=float(le2.mean()))


p2 = pd.DataFrame([dict(sample="all inventories", **p2_of(allk)), dict(sample=f"without {FOCUS}", **p2_of(noes))])
print("     model (3) residual asymmetry, all inventories and without", FOCUS)
print(p2.round(3).to_string(index=False))
for r in p2.itertuples():
    for k in ["skew_all", "skew_complete", "skew_incomplete", "m3_share_le2", "share_le2"]:
        note(f"R5 {r.sample} {k}", getattr(r, k))

# (R9) residual skewness of Eq. (10) by specification: Table 2 model (1) -2.94, Table S16 Panel C 'None' -3.23, and the pooled specification of the stage-21 simulation (structural covariates + country effects)
rows = []
for lab, cols, fe in [("structural covariates only (Table 2, model 1)", BASE, None),
                      ("structural covariates + country effects (simulation pooled specification)", BASE, "country_code"),
                      ("+ reporting characteristics, no coverage indicators (Table S16, Panel C, None)", BASE + ["co2eq", "lca", "mon"], "country_code"),
                      ("structural + coverage indicators + country effects (simulation controlled specification)",
                       BASE + ["sc1", "sc2", "sc3", "sc4", "sc6", "core5_complete_flag"], "country_code"),
                      ("+ sector-count and core-five indicators (Table 2, model 3)", BASE + Q, "country_code")]:
    Xs, _ = design(main, cols, fe=fe)
    es_ = y - Xs @ np.linalg.lstsq(Xs, y, rcond=None)[0]
    rows.append(dict(specification=lab, skewness=stats.skew(es_)))
t_skew = pd.DataFrame(rows)
print("\n(R9) residual skewness of Eq. (10) by specification:")
print(t_skew.round(3).to_string(index=False))
for i, r in enumerate(t_skew.itertuples()):
    note(f"R9 skewness spec {i} {r.specification}", r.skewness)

# (R6) source of the identifying variation: sum of squares of w_hat residualised on the other regressors, split by reported-sector set
Xo, _ = design(main, BASE + Qnc, fe="country_code")
r_w = w_full - Xo @ np.linalg.lstsq(Xo, w_full, rcond=None)[0]
s2 = r_w ** 2
t_lev = (pd.DataFrame(dict(set=sets, sectors=sc, s2=s2)).groupby(["set", "sectors"])
         .agg(inventories=("s2", "size"), share=("s2", lambda x: x.sum() / s2.sum())).reset_index()
         .sort_values("share", ascending=False).reset_index(drop=True))
t_lev["cum_share"] = t_lev.share.cumsum()
print("\n(R6) share of the residualised variation of w_hat by reported-sector set (top 6):")
print(t_lev.head(6).round(3).to_string(index=False))
note("R6 share 1-2 sectors", s2[sc <= 2].sum() / s2.sum())
for i, r in t_lev.head(3).iterrows():
    note(f"R6 set {r.set} share", r.share); note(f"R6 set {r.set} n", r.inventories)

# =====================================================================================
# (R7) unit of the benchmark flags, (R8) monitoring inventories without a pair
# =====================================================================================
hq = main[(main.sc >= 5) & (main.core5_complete_flag == 1)].reset_index(drop=True)   # same benchmark sample as stage 10
ptop = np.asarray(bm["ptop"]); flag = ptop >= 0.9
assert len(ptop) == len(hq) == bm["n_hq"] and int(flag.sum()) == bm["stable"], "benchmark sample does not match step 10"
r7 = dict(complete_inventories=len(hq), complete_municipalities=hq.GCoM_ID.nunique(), complete_meis=int((hq.mon == 1).sum()),
          flagged_inventories=int(flag.sum()), flagged_municipalities=hq.GCoM_ID[flag].nunique(), flagged_meis=int((hq.mon[flag] == 1).sum()))
print(f"\n(R7) benchmark sample: {r7['complete_inventories']} inventories of {r7['complete_municipalities']} municipalities "
      f"({r7['complete_meis']} monitoring inventories); flagged: {r7['flagged_inventories']} inventories of "
      f"{r7['flagged_municipalities']} municipalities ({r7['flagged_meis']} monitoring inventories)")
for k, v in r7.items():
    note(f"R7 {k}", v)
mei = main[main.mon == 1]
unp = mei[~mei.GCoM_ID.isin(d.id)]
for gid in unp.GCoM_ID:
    yrs = main.loc[main.GCoM_ID == gid, ["mon", "inventory_year"]].sort_values("mon").inventory_year.tolist()
    print(f"(R8) monitoring inventory without a pair: {gid}, BEI/MEI years {yrs} (elapsed time 0, excluded from changes)")
note("R8 MEIs", len(mei)); note("R8 unpaired MEIs", len(unp))

save_obj("robustness.pkl", dict(t_es=t_es, t_loo=t_loo, t_pb=t_pb, t_single=t_single, t_rem=t_rem, t_skew=t_skew, r3=r3, t_gap=t_gap, dn2=dn2, t_lam=t_lam, p2=p2, t_lev=t_lev,
                                r7=r7, nums=pd.DataFrame(nums)))
with pd.ExcelWriter(os.path.join(TAB_DIR, "Robustness_Tables.xlsx")) as xw:
    t_es.to_excel(xw, sheet_name="R1_ES61", index=False)
    t_loo.to_excel(xw, sheet_name="R1_leave_one_NUTS2_out", index=False)
    t_pb.to_excel(xw, sheet_name="R2_Table3_PanelB", index=False)
    t_single.to_excel(xw, sheet_name="R2c_Table3_PanelC_region", index=False)
    t_rem.to_excel(xw, sheet_name="R2d_removals_by_sector", index=False)
    pd.DataFrame([r3]).to_excel(xw, sheet_name="R3_NUTS2_cluster", index=False)
    t_gap.to_excel(xw, sheet_name="R4_elapsed_years", index=False)
    dn2.to_excel(xw, sheet_name="R5_donors_by_NUTS2", index=False)
    t_lam.to_excel(xw, sheet_name="R5_R6_lambda", index=False)
    t_lev.to_excel(xw, sheet_name="R6_leverage_by_set", index=False)
    p2.to_excel(xw, sheet_name="R5_asymmetry_without_ES61", index=False)
    t_skew.to_excel(xw, sheet_name="R9_skewness_by_spec", index=False)
    pd.DataFrame([r7]).to_excel(xw, sheet_name="R7_flags", index=False)
pd.DataFrame(nums).to_csv(os.path.join(TAB_DIR, "robustness_numbers.csv"), index=False, encoding="utf-8-sig")
print("saved: work/robustness.pkl, tables/Robustness_Tables.xlsx, tables/robustness_numbers.csv")


# =====================================================================================
# LaTeX tables: Table S20 (sensitivity of the Proposition 3 decomposition), S21 (Table 3 Panels B and C by region), S22 (robustness for Propositions 1 and 2)
# All numbers are formatted directly from the results above; none is entered by hand.
# =====================================================================================
def fx(x, dgt=2):
    v = round(float(x), dgt)
    v = 0.0 if v == 0 else v
    t_ = f"{abs(v):,.{dgt}f}"
    return ("$-$" + t_) if v < 0 else t_


def fci(lo, hi, dgt=2):
    return f"[{fx(lo, dgt)}, {fx(hi, dgt)}]"


def fpc(x, dgt=0):
    return f"{100 * float(x):.{dgt}f}\\%"


def fn(x):
    return f"{int(x):,}"


def fpi(lo, hi):   # percentage interval (integers; negative values use the LaTeX minus sign)
    return f"[{fx(100 * float(lo), 0)}, {fx(100 * float(hi), 0)}]"


ABBR = ["Mu", "Te", "Re", "Ma", "Tr", "Wa"]


def set_label(s_):
    rep_ = [a_ for a_, c_ in zip(ABBR, s_) if c_ == "1"]; mis = [a_ for a_, c_ in zip(ABBR, s_) if c_ == "0"]
    if len(rep_) >= 4:
        return "no " + ", ".join(mis)
    return " ".join(rep_) + (" only" if len(rep_) == 1 else "")


def es_row(sample, scale):
    g = t_es[(t_es["sample"] == sample) & (t_es.scale == scale)].set_index("item")
    return g


FOC_NAME = "Andaluc\\'ia (ES61)"
L = []
L.append(r"\begin{table}[width=\linewidth,pos=htbp]")
L.append(r"\caption{Regional and elapsed-time sensitivity of the scope and retained-sector terms (Proposition~\ref{prop:change})}\label{tab:S20}")
L.append(r"\footnotesize")
L.append(r"\setlength{\tabcolsep}{4pt}")
L.append(r"\begin{tabular*}{\tblwidth}{@{}LRRRRRR@{}}")
L.append(r"\toprule")
hdr = r"Sample & Pairs & Changed & Difference & Scope & Retained & Retained \\" + "\n" + r" & & coverage & & term & sectors & share \\"
for scale, title, dg in [("annual", r"Panel A. Pooled mean relative to constant coverage, \% per year (Eq.~\ref{eq:inflannual})", 2),
                         ("log", r"Panel B. The same decomposition in log points (Eq.~\ref{eq:infl})", 3)]:
    if scale == "log":
        L.append(r"\midrule")
    L.append(r"\multicolumn{7}{@{}l}{\textit{" + title + r"}}\\")
    L.append(hdr)
    L.append(r"\midrule")
    samples = [("all pairs", "All pairs (Table~\\ref{tab:S17})"), (f"without {FOCUS}", f"Without {FOCUS}"), (f"{FOCUS} only", f"{FOCUS} only")]
    for key, lab in samples:
        g = es_row(key, scale)
        L.append(f"{lab} & {fn(g.loc['difference', 'pairs'])} & {fn(g.loc['difference', 'changed'])} & {fx(g.loc['difference', 'value'], dg)} & "
                 f"{fx(g.loc['scope_part', 'value'], dg)} & {fx(g.loc['retained_part', 'value'], dg)} & {fpc(g.loc['retained_share', 'value'])} \\\\")
        L.append(f" & & & {fci(g.loc['difference', 'ci_lo'], g.loc['difference', 'ci_hi'], dg)} & {fci(g.loc['scope_part', 'ci_lo'], g.loc['scope_part', 'ci_hi'], dg)} & "
                 f"{fci(g.loc['retained_part', 'ci_lo'], g.loc['retained_part', 'ci_hi'], dg)} & "
                 f"{fpi(g.loc['retained_share', 'ci_lo'], g.loc['retained_share', 'ci_hi'])} \\\\")
    if scale == "annual":
        L.append(f"All pairs, NUTS2 resampling & & & {fci(r3['diff_lo'], r3['diff_hi'])} & {fci(r3['scope_lo'], r3['scope_hi'])} & "
                 f"{fci(r3['lo'], r3['hi'])} & {fpi(r3['share_lo'], r3['share_hi'])} \\\\")
        for r in t_gap.itertuples():
            if r.sample == "all pairs":
                continue
            lab = {"without 1-year pairs": "Without one-year pairs", "without pairs of 3 years or less": "Without pairs of $\\leq$3 years"}[r.sample]
            L.append(f"{lab} & {fn(r.pairs)} & {fn(r.changed)} & {fx(r.difference)} & {fx(r.scope_part)} & {fx(r.retained_part)} & {fpc(r.retained_share)} \\\\")
L.append(r"\bottomrule")
L.append(r"\end{tabular*}")
g_no = es_row(f"without {FOCUS}", "annual")
L.append(r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize Notes: ES61: Andaluc\'ia. Changed coverage: pairs with additions, removals or both. Difference = scope term + "
         r"retained-sector term; retained share = retained-sector term/difference. Intervals: percentiles of " + f"{B}" + r" bootstrap replications that resample "
         r"NUTS3 regions within countries (NUTS2 regions in the row so labelled). " + f"{FOCUS} supplies {fn((d.nuts2 == FOCUS).sum())} of the "
         f"{fn(len(d))} pairs, {fn((es & chg).sum())} of the {fn(chg.sum())} changed-coverage pairs, {fn((es & (d.cat == 'removal only').values).sum())} of the "
         f"{fn((d.cat == 'removal only').sum())} removal-only pairs and {fn((es & ~chg).sum())} of the {fn((~chg).sum())} constant-coverage pairs. "
         f"Without it, {100 * g_no.loc['retained_part', 'p_ge0']:.1f}\\% of replications give a non-negative annualised retained-sector term. "
         f"Omitting instead any one of the other {len(others)} NUTS2 regions with at least {MIN_PAIRS} pairs gives annualised retained shares of "
         f"{100 * others.annual_retained_share.min():.0f}--{100 * others.annual_retained_share.max():.0f}\\%. One-year pairs: {int((d.dy == 1).sum())}. "
         f"{FOCUS}-only rows rest on {d.nuts3[es].nunique()} NUTS3 regions, so their intervals are imprecise."
         r"\end{minipage}")
L.append(r"\end{table}")
L.append("")

L.append(r"\begin{table}[width=\linewidth,pos=htbp]")
L.append(r"\caption{Retained-sector contrasts and single-sector removals by region (Table~\ref{tab:pairs}, Panels B and C)}\label{tab:S21}")
L.append(r"\footnotesize")
L.append(r"\setlength{\tabcolsep}{3pt}")
L.append(r"\begin{tabular*}{\tblwidth}{@{}LRRRRR@{}}")
L.append(r"\toprule")
L.append(r"\multicolumn{6}{@{}l}{\textit{Panel A. Change in sectors reported in both inventories, relative to constant coverage (ln points; $t$ in parentheses)}}\\")
L.append(r"Effects; sample & Pairs & Addition only & Removal only & Both & Years \\")
L.append(r"\midrule")
for key, lab in [("all pairs", "Country; all pairs (Table~\\ref{tab:pairs})"), (f"without {FOCUS}", f"Country; without {FOCUS}"),
                 ("all pairs, NUTS2 effects", "NUTS2; all pairs"), (f"without {FOCUS}, NUTS2 effects", f"NUTS2; without {FOCUS}"),
                 (f"{FOCUS} only", f"None; {FOCUS} only")]:
    g = t_pb[t_pb["sample"] == key].set_index("var")
    L.append(f"{lab} & {fn(g.n.iloc[0])} & " + " & ".join(fx(g.loc[v, 'coef'], 3) for v in CATS + ["dy"]) + r" \\")
    L.append(" & & " + " & ".join(f"({fx(g.loc[v, 't'], 2)})" for v in CATS + ["dy"]) + r" \\")
L.append(r"\midrule")
L.append(r"\multicolumn{6}{@{}l}{\textit{Panel B. Single-sector removals: observed versus share-implied change (ln points)}}\\")
L.append(r"Removed sector; sample & Pairs & Share & Observed & Implied & Ratio \\")
L.append(r"\midrule")
for sec, lab in [("all single removals", "All single removals"), ("waste", "Waste"), ("transport", "Transport")]:
    for smp, sl in [("all", lab), (f"without {FOCUS}", r"\quad without " + FOCUS), (f"{FOCUS} only", r"\quad " + FOCUS + " only")]:
        g = t_single[(t_single["sample"] == smp) & (t_single.sector == sec)].iloc[0]
        L.append(f"{sl} & {fn(g.pairs)} & {fpc(g.share, 1)} & {fx(g.observed, 3)} & {fx(g.implied, 3)} & {fx(g.ratio, 2)} \\\\")
L.append(r"\bottomrule")
L.append(r"\end{tabular*}")
rm = t_rem.set_index("sector")
L.append(r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize Notes: ES61: Andaluc\'ia. Panel A: regressions of the change in ln emissions of the sectors reported in both "
         r"inventories on coverage-change indicators (reference: constant coverage) and years between inventories (Years); standard errors clustered by NUTS3 region. NUTS2 effects "
         r"compare changed-coverage pairs with constant-coverage pairs of the same region. " + f"The {FOCUS}-only row rests on {d.nuts3[es].nunique()} NUTS3 clusters "
         f"and {int((es & (d.cat == 'addition only').values).sum())} addition-only pairs. " + r"Panel B: as Table~\ref{tab:pairs}, Panel C; the observed change also "
         r"contains the change in retained sectors. Removal events by sector (Table~\ref{tab:S6}): " +
         "; ".join(f"{sec} {fn(rm.loc[sec, 'removals'])} ({fn(rm.loc[sec, 'in_focus'])} in ES61)" for sec in ["waste", "transport", "manufacturing"]) +
         r".\end{minipage}")
L.append(r"\end{table}")
L.append("")

L.append(r"\begin{table}[width=\linewidth,pos=htbp]")
L.append(r"\caption{Robustness of the share-implied shortfall coefficient and of residual asymmetry (Propositions~\ref{prop:levels} and \ref{prop:asym})}\label{tab:S22}")
L.append(r"\footnotesize")
L.append(r"\begin{tabular*}{\tblwidth}{@{}LRRRR@{}}")
L.append(r"\toprule")
L.append(r"\multicolumn{5}{@{}l}{\textit{Panel A. Coefficient on $\hat w_i$ replacing the sector-count indicators (NUTS3-clustered standard errors)}}\\")
L.append(r"Sample & Inventories & Coef. & SE & $t$ ($H_0$: $-1$) \\")
L.append(r"\midrule")
lam_lab = {"all inventories (Table S14, Panel C)": "All inventories (Table~\\ref{tab:S14})",
           f"without {FOCUS} inventories and donors": "Without " + FOC_NAME + " inventories and donors",
           "without one-sector municipal-only inventories": "Without municipal-only inventories",
           "without municipal-only and municipal+waste inventories": "Without municipal-only and municipal--waste inventories",
           "three or more sectors": "Three or more sectors", "four or more sectors (Section 6.1)": "Four or more sectors (Table~\\ref{tab:S14})"}
for r in t_lam.itertuples():
    L.append(f"{lam_lab[r.spec]} & {fn(r.n)} & {fx(r.coef, 3)} & {fx(r.se, 3)} & {fx((r.coef + 1) / r.se, 2)} \\\\")
L.append(r"\midrule")
L.append(r"\multicolumn{5}{@{}l}{\textit{Panel B. Share of the residualised variation of $\hat w_i$ by reported-sector set (largest six)}}\\")
L.append(r"Reported sectors & Sectors & Inventories & Share & Cumulative \\")
L.append(r"\midrule")
for r in t_lev.head(6).itertuples():
    L.append(f"{set_label(r.set)} & {r.sectors} & {fn(r.inventories)} & {fpc(r.share, 1)} & {fpc(r.cum_share, 1)} \\\\")
L.append(r"\midrule")
L.append(r"\multicolumn{5}{@{}l}{\textit{Panel C. Residual asymmetry, Eq.~(\ref{eq:model}), model (3) of Table~\ref{tab:model}}}\\")
L.append(r"Quantity & All inventories & Without " + FOC_NAME + r" & & \\")
L.append(r"\midrule")
pa, pb = p2.iloc[0], p2.iloc[1]
L.append(f"Inventories & {fn(pa.n)} & {fn(pb.n)} & & \\\\")
L.append(f"Skewness, complete reporters & {fx(pa.skew_complete)} & {fx(pb.skew_complete)} & & \\\\")
L.append(f"Skewness, incomplete reporters & {fx(pa.skew_incomplete)} & {fx(pb.skew_incomplete)} & & \\\\")
L.append(f"One- and two-sector inventories: share of the third moment & {fpc(pa.m3_share_le2, 1)} & {fpc(pb.m3_share_le2, 1)} & & \\\\")
L.append(f"One- and two-sector inventories: share of inventories & {fpc(pa.share_le2, 1)} & {fpc(pb.share_le2, 1)} & & \\\\")
L.append(r"\bottomrule")
L.append(r"\end{tabular*}")
n_es_don = int((donors.nuts2 == FOCUS).sum()); n_sp = int((donors.country == "es").sum()); n_it = int((donors.country == "it").sum())
L.append(r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize Notes: six-sector donors: " + f"{fn(six.sum())} (Spain {fn(n_sp)}, of which {fn(n_es_don)} "
         f"in Andaluc\\'ia; Italy {fn(n_it)}). Panel A, row without " + FOC_NAME + f": its {fn((~noes).sum())} inventories are removed and $\\hat w_i$ is "
         f"recomputed from the remaining {fn(six.sum() - n_es_don)} donors; Spain then has {fn(n_sp - n_es_don)} donors, still above the minimum of {MIN_DONORS}. "
         r"Panel B: squared residuals of $\hat w_i$ after projection on the other regressors of Panel A, summed by set; Mu, Te, Re, Ma, Tr, Wa: municipal, "
         r"tertiary, residential, manufacturing, transport, waste; `no' lists the sectors not reported. Panel C: model (3) re-estimated without " + FOC_NAME +
         r"'s inventories (donors play no part in model (3)); complete reporters report the five core sectors.\end{minipage}")
L.append(r"\end{table}")
tex_path = os.path.join(TAB_DIR, "Tables_S20_S22.tex")
with open(tex_path, "w", encoding="utf-8") as fh:
    fh.write("% Tables S20-S22 generated by 22_robustness_checks.py (cas-sc LaTeX)\n" + "\n".join(L) + "\n")
print("saved:", tex_path)
