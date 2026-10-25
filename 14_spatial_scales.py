# -*- coding: utf-8 -*-
"""
14_spatial_scales.py - at which spatial scale does unexplained variation cluster: administrative units and placebo
partitions.

Sample: as in stage 09 (baseline inventories, integer-degree coordinates excluded, n = 5,582). Fixed effects for each
unit are added to the structural and reporting covariates of the conditional model, Eq. (10).
(1) Country -> NUTS1 -> NUTS2 -> NUTS3 fixed effects: residual Moran's I (k = 5, 10, 20) and the share of the
    country-FE residual variance absorbed
(2) Placebo partitions: within each country, random spatial clusters with as many groups as the real units.
      k-means  : k-means on coordinates (compact clusters)
      voronoi  : nearest-seed partition around randomly drawn seed municipalities (clusters of uneven size)
    If the real administrative units absorb more variance than the placebo partitions, the excess is the
    'boundary-aligned' component. If the placebo partitions reduce Moran's I just as much, that reduction alone
    does not show an administrative-unit effect.
(3) (optional) if statsmodels is installed: nested random-intercept variance components of the country-FE residuals
    (NUTS1 > NUTS2 > NUTS3)
Input : work/extra.pkl
Output: work/scales.pkl, figures/revision/Figure_R1 (Fig. 6)
Manuscript: Table S10, Fig. 6, Section 6.6, Supplementary Section S10
"""
from lib import *

NREP = 200          # placebo partitions drawn per comparison
SEED = 2026
banner("14 spatial scales: country / NUTS1 / NUTS2 / NUTS3 and placebo partitions")
mx = load_obj("extra.pkl")["main"]
sp = mx[(mx.mon == 0) & (mx.int_coords == 0)].copy().reset_index(drop=True)
qv = [c for c in Q if sp[c].nunique() > 1]
y = sp.lnE.values
Xb = np.column_stack([np.ones(len(sp))] + [sp[c].values.astype(float) for c in BASE + qv])
W = {k: knn_weights(sp.latitude.values, sp.longitude.values, k=k)[0] for k in (5, 10, 20)}
e_cty, _ = fe_resid(y, Xb, sp.country_code.values)
v0 = e_cty.var()
n = len(sp); tss = np.sum((y - y.mean()) ** 2)

rows = []
for lv, lab in [("country_code", "country"), ("N1", "NUTS1"), ("N2", "NUTS2"), ("N3", "NUTS3")]:
    e, p = fe_resid(y, Xb, sp[lv].values)
    rows.append(dict(level=lab, groups=sp[lv].nunique(), adjR2=1 - (e @ e / (n - p)) / (tss / (n - 1)),
                     absorbed=1 - e.var() / v0, I5=moran_value(e, W[5]), I10=moran_value(e, W[10]), I20=moran_value(e, W[20])))
scales = pd.DataFrame(rows)
print(f"spatial sample n = {n}")
print(scales.round(4).to_string(index=False))

# ---- placebo partitions
XY = xy_km(sp.latitude.values, sp.longitude.values)
cty = sp.country_code.values


def placebo_labels(sub_idx, level, kind, rng):
    lab = np.empty(len(sub_idx), dtype=object)
    s_cty = cty[sub_idx]
    for c in np.unique(s_cty):
        loc = np.where(s_cty == c)[0]
        g = sp.loc[sub_idx[loc], level].nunique()
        if g <= 1 or len(loc) <= g:
            lab[loc] = c
            continue
        P = XY[sub_idx[loc]]
        part = kmeans_np(P, g, rng) if kind == "kmeans" else voronoi_np(P, g, rng)
        lab[loc] = [f"{c}_{j}" for j in part]
    return lab


def compare(sub, level, kinds, label, with_moran):
    idx = np.where(sub)[0]
    yy, XX = y[idx], Xb[idx]
    XX = XX[:, np.r_[0, 1 + np.where(XX[:, 1:].std(0) > 0)[0]]]
    e0, _ = fe_resid(yy, XX, cty[idx])
    er, _ = fe_resid(yy, XX, sp[level].values[idx])
    real = 1 - er.var() / e0.var()
    Wm = knn_weights(sp.latitude.values[idx], sp.longitude.values[idx], k=10)[0] if with_moran else None
    out = []
    for kind in kinds:
        rng = np.random.default_rng(SEED)
        vals, Is = [], []
        for r in range(NREP):
            lab = placebo_labels(idx, level, kind, rng)
            ep, _ = fe_resid(yy, XX, lab)
            vals.append(1 - ep.var() / e0.var())
            if with_moran:
                Is.append(moran_value(ep, Wm))
        vals = np.array(vals)
        rec = dict(sample=label, level={"N2": "NUTS2", "N3": "NUTS3"}[level], placebo=kind, groups=int(pd.Series(sp[level].values[idx]).nunique()),
                   real_absorbed=real, placebo_mean=vals.mean(), placebo_lo=np.quantile(vals, .025), placebo_hi=np.quantile(vals, .975),
                   excess=real - vals.mean(), excess_share_of_real=(real - vals.mean()) / real, p_placebo_ge_real=float(np.mean(vals >= real)))
        if with_moran:
            rec.update(real_I10=moran_value(er, Wm), placebo_I10_mean=float(np.mean(Is)), placebo_I10_lo=float(np.quantile(Is, .025)),
                       placebo_I10_hi=float(np.quantile(Is, .975)))
        out.append(rec)
        print(f"{label:6s} {rec['level']} {kind:7s}: real {real:.3f} | placebo {rec['placebo_mean']:.3f} [{rec['placebo_lo']:.3f}, {rec['placebo_hi']:.3f}] "
              f"| excess {rec['excess']:+.3f} | P(placebo >= real) {rec['p_placebo_ge_real']:.2f}"
              + (f" | I10 real {rec['real_I10']:.3f} vs placebo {rec['placebo_I10_mean']:.3f} [{rec['placebo_I10_lo']:.3f}, {rec['placebo_I10_hi']:.3f}]" if with_moran else ""), flush=True)
    return out


allm = np.ones(n, bool)
pl = []
pl += compare(allm, "N3", ["kmeans", "voronoi"], "all", True)
pl += compare(allm, "N2", ["kmeans"], "all", True)
for c in ["it", "es", "be"]:
    for lev in ["N2", "N3"]:
        pl += compare(cty == c, lev, ["kmeans"], c, False)
placebo = pd.DataFrame(pl)

# ---- (optional) nested random-intercept variance components
vdec = None
try:
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")
    dd = pd.DataFrame(dict(e=e_cty, n1=sp.N1.values, n2=sp.N2.values, n3=sp.N3.values))
    fit = smf.mixedlm("e ~ 1", dd, groups=dd["n1"], vc_formula={"n2": "0 + C(n2)", "n3": "0 + C(n3)"}, re_formula="1").fit(reml=True, method=["lbfgs"])
    vc = dict(zip(fit.model.exog_vc.names, fit.vcomp)); v1 = float(fit.cov_re.iloc[0, 0]); ve = float(fit.scale)
    tot = v1 + sum(vc.values()) + ve
    vdec = dict(NUTS1=v1 / tot, NUTS2=vc["n2"] / tot, NUTS3=vc["n3"] / tot, municipal=ve / tot)
    print("nested random-intercept variance shares of country-FE residuals (REML):", {k: round(v, 3) for k, v in vdec.items()})
except ImportError:
    print("statsmodels not installed: nested variance components skipped (optional)")
save_obj("scales.pkl", dict(scales=scales, placebo=placebo, vdec=vdec, n=n))

# ---- Fig. 6 (figures/revision/Figure_R1)
FIG_REV = os.path.join(OUT_DIR, "figures", "revision"); os.makedirs(FIG_REV, exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(FIG_W, 2.7))
xl = scales.level.tolist()
ax[0].plot(xl, scales.I10, "o-", color="#4c78a8", label="k = 10")
ax[0].plot(xl, scales.I20, "s--", color="#9ecae9", label="k = 20")
q = placebo[(placebo["sample"] == "all") & (placebo.level == "NUTS3")]
for _, r in q.iterrows():
    ax[0].errorbar([3.15 if r.placebo == "kmeans" else 3.3], [r.placebo_I10_mean], yerr=[[r.placebo_I10_mean - r.placebo_I10_lo], [r.placebo_I10_hi - r.placebo_I10_mean]],
                   fmt="D" if r.placebo == "kmeans" else "^", color="#d9822b" if r.placebo == "kmeans" else "#59a14f", ms=4, capsize=2, label=f"placebo ({r.placebo})")
ax[0].axhline(0, color="k", lw=0.5); ax[0].set_ylabel("Moran's I of residuals"); ax[0].set_title("(a) Moran's I by fixed-effect unit")
ax[0].legend(frameon=False, fontsize=6)
ax[1].bar(xl, scales.absorbed * 100, color="#4c78a8")
for _, r in placebo[(placebo["sample"] == "all")].iterrows():
    xpos = 3 if r.level == "NUTS3" else 2
    off = -0.25 if r.placebo == "kmeans" else 0.25
    ax[1].errorbar([xpos + off], [r.placebo_mean * 100], yerr=[[(r.placebo_mean - r.placebo_lo) * 100], [(r.placebo_hi - r.placebo_mean) * 100]],
                   fmt="D" if r.placebo == "kmeans" else "^", color="#d9822b" if r.placebo == "kmeans" else "#59a14f", ms=4, capsize=2)
ax[1].set_ylabel("% of country-FE residual variance"); ax[1].set_title("(b) Variance absorbed vs placebo")
cc_ = placebo[placebo["sample"].isin(["it", "es", "be"])].copy()
labs = [f"{r['sample']} {r['level']}" for _, r in cc_.iterrows()]
ax[2].barh(labs, cc_.excess * 100, color=np.where(cc_.p_placebo_ge_real < 0.05, "#4c78a8", "#c7c7c7"))
ax[2].axvline(0, color="k", lw=0.5); ax[2].set_xlabel("excess over k-means placebo (pp)\ngrey: p \u2265 0.05"); ax[2].set_title("(c) Excess over placebo")
plt.tight_layout()
save_figure(fig, FIG_REV, "Figure_R1")
print("saved: work/scales.pkl, figures/revision/Figure_R1")
