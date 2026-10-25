# -*- coding: utf-8 -*-
"""
15_borders_fua.py - do residuals jump at administrative boundaries, and do they align with functional urban areas (FUAs)
or with administrative units?

(1) Boundary-pair test: for every pair of municipalities in the same country within 20 km, the semivariance
    0.5 (e_i - e_j)^2 of the country-FE residuals. Pairs are classed as same NUTS3 / across a NUTS3 boundary only
    (same NUTS2) / across a NUTS2 boundary and compared with controls for distance band (4 km) and country.
    Standard errors: dyadic-robust (allowing for correlation between pairs that share a municipality), plus
    NUTS2-cluster bootstrap intervals.
(2) Functional urban areas: for municipalities inside the Eurostat GISCO 2018 FUA boundaries (about 30% of the
    sample), NUTS3 fixed effects are compared with FUA fixed effects. Partial F tests ask whether one adds explanatory
    power given the other. FUAs are defined by commuting flows, so they serve as a proxy for the spatial extent of flows.
(3) Joint-inventory check: number of joint-SECAP (option 2) signatories, inventories with identical per-capita
    emissions (4 significant digits) within a NUTS3 region, and Moran's I after excluding them.
Input : work/extra.pkl, FUA_GEOJSON (top of lib.py; downloaded automatically if missing)
Output: work/borders_fua.pkl, figures/revision/Figure_R2 (Fig. S10)
Manuscript: Table S11, Fig. S10, Section 6.6, Supplementary Section S11
"""
from lib import *

NB = 500           # NUTS2-cluster bootstrap replications
SEED = 11
RMAX = 20.0        # maximum pair distance (km)
banner("15 boundary discontinuity, functional urban areas, joint inventories")
mx = load_obj("extra.pkl")["main"]
sp = mx[(mx.mon == 0) & (mx.int_coords == 0)].copy().reset_index(drop=True)
qv = [c for c in Q if sp[c].nunique() > 1]
Xb = np.column_stack([np.ones(len(sp))] + [sp[c].values.astype(float) for c in BASE + qv])
y = sp.lnE.values
e, _ = fe_resid(y, Xb, sp.country_code.values)

# ---- (1) boundary pairs
lat = np.radians(sp.latitude.values); lon = np.radians(sp.longitude.values)
xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
pairs = np.array(sorted(cKDTree(xyz).query_pairs(2 * np.sin(RMAX / 6371.0 / 2))))
i, j = pairs[:, 0], pairs[:, 1]
dist = 2 * 6371.0 * np.arcsin(np.linalg.norm(xyz[i] - xyz[j], axis=1) / 2)
ok = sp.country_code.values[i] == sp.country_code.values[j]
i, j, dist = i[ok], j[ok], dist[ok]
gam = 0.5 * (e[i] - e[j]) ** 2
band = np.minimum((dist // 4).astype(int), 4)
N3, N2 = sp.N3.values, sp.N2.values
cat = np.where(N3[i] == N3[j], 0, np.where(N2[i] == N2[j], 1, 2))
rows = []
for cm in [None, "it", "es", "be"]:
    msk = np.ones(len(i), bool) if cm is None else (sp.country_code.values[i] == cm)
    ii, jj, g, b, k = i[msk], j[msk], gam[msk], band[msk], cat[msk]
    cc = sp.country_code.values[ii]
    X = np.column_stack([np.ones(len(g)), (k == 1).astype(float), (k == 2).astype(float)] +
                        [(b == q).astype(float) for q in range(1, 5)] + [(cc == u).astype(float) for u in np.unique(cc)[1:]])
    XtXi = np.linalg.pinv(X.T @ X); beta = XtXi @ X.T @ g; u = g - X @ beta; sc = X * u[:, None]
    S = np.zeros((len(sp), X.shape[1])); np.add.at(S, ii, sc); np.add.at(S, jj, sc)
    se = np.sqrt(np.abs(np.diag(XtXi @ (S.T @ S - sc.T @ sc) @ XtXi)))
    ureg, rinv = np.unique(N2[ii], return_inverse=True)
    rng = np.random.default_rng(SEED); bs = []
    for _ in range(NB):
        w = np.bincount(rng.integers(0, len(ureg), len(ureg)), minlength=len(ureg)).astype(float)[rinv]
        if w[k == 1].sum() == 0 or w[k == 2].sum() == 0:
            continue
        Xw = X * w[:, None]
        bs.append(np.linalg.lstsq(Xw.T @ X, Xw.T @ g, rcond=None)[0][1:3])
    bs = np.array(bs); g0 = g[k == 0].mean()
    rows.append(dict(sample=cm or "all", pairs_same_N3=int((k == 0).sum()), pairs_cross_N3_same_N2=int((k == 1).sum()), pairs_cross_N2=int((k == 2).sum()),
                     gamma_same_N3=g0, d_cross_N3=beta[1], se_d_cross_N3=se[1], ci_lo_N3=np.quantile(bs[:, 0], .025), ci_hi_N3=np.quantile(bs[:, 0], .975),
                     d_cross_N2=beta[2], se_d_cross_N2=se[2], ci_lo_N2=np.quantile(bs[:, 1], .025), ci_hi_N2=np.quantile(bs[:, 1], .975)))
border = pd.DataFrame(rows)
border["rel_N3"] = border.d_cross_N3 / border.gamma_same_N3; border["rel_N2"] = border.d_cross_N2 / border.gamma_same_N3
print(f"pairs within {RMAX:.0f} km (same country): {len(i)}")
print(border[["sample", "pairs_same_N3", "pairs_cross_N3_same_N2", "pairs_cross_N2", "gamma_same_N3", "d_cross_N3", "se_d_cross_N3", "rel_N3",
              "d_cross_N2", "se_d_cross_N2", "rel_N2"]].round(4).to_string(index=False))
vg = pd.DataFrame(dict(km=band * 4 + 2, cat=cat, g=gam)).groupby(["km", "cat"]).g.agg(["mean", "size"]).reset_index()

# ---- (2) functional urban areas
ensure_file(FUA_GEOJSON, FUA_GEOJSON_URL)
sp["FUA"] = points_in_fua(sp.longitude.values, sp.latitude.values, FUA_GEOJSON)
ins = sp.FUA.notna().values
f = sp[ins].copy().reset_index(drop=True)
qf = [c for c in Q if f[c].nunique() > 1]
Xf = np.column_stack([np.ones(len(f))] + [f[c].values.astype(float) for c in BASE + qf])
ef, _ = fe_resid(f.lnE.values, Xf, f.country_code.values)
multi = f.groupby("FUA").N3.nunique()


def dummies(lab):
    return pd.get_dummies(pd.Series(lab), drop_first=True).astype(float).values


def rss_rank(*blocks):
    X = np.column_stack([np.ones(len(f))] + list(blocks))
    b, *_ = np.linalg.lstsq(X, ef, rcond=None); r = ef - X @ b
    return r @ r, np.linalg.matrix_rank(X)


D3, DF = dummies(f.N3.values), dummies(f.FUA.values)
rss0, k0 = rss_rank()
rss3, k3 = rss_rank(D3); rssF, kF = rss_rank(DF); rssB, kB = rss_rank(D3, DF)
nf = len(f)
from scipy.stats import f as fdist
F_fua_given_n3 = ((rss3 - rssB) / (kB - k3)) / (rssB / (nf - kB)); p1 = 1 - fdist.cdf(F_fua_given_n3, kB - k3, nf - kB)
F_n3_given_fua = ((rssF - rssB) / (kB - kF)) / (rssB / (nf - kB)); p2 = 1 - fdist.cdf(F_n3_given_fua, kB - kF, nf - kB)
adj = lambda rss, k: 1 - (rss / (nf - k)) / (rss0 / (nf - k0))
fua = dict(n_sample=len(sp), n_inside=int(ins.sum()), share_inside=float(ins.mean()), n_fua=int(f.FUA.nunique()), fua_multi_n3=int((multi > 1).sum()),
           munis_in_multi=int(f.FUA.isin(multi[multi > 1].index).sum()), n3_groups=int(f.N3.nunique()),
           absorbed_N3=1 - rss3 / rss0, absorbed_FUA=1 - rssF / rss0, absorbed_both=1 - rssB / rss0,
           adj_N3=adj(rss3, k3), adj_FUA=adj(rssF, kF), adj_both=adj(rssB, kB),
           F_fua_given_n3=F_fua_given_n3, df_fua_given_n3=(kB - k3, nf - kB), p_fua_given_n3=p1,
           F_n3_given_fua=F_n3_given_fua, df_n3_given_fua=(kB - kF, nf - kB), p_n3_given_fua=p2)
Wf = knn_weights(f.latitude.values, f.longitude.values, k=10)[0]
fua.update(I10_country=moran_value(ef, Wf), I10_N3=moran_value(fe_resid(f.lnE.values, Xf, f.N3.values)[0], Wf),
           I10_FUA=moran_value(fe_resid(f.lnE.values, Xf, f.FUA.values)[0], Wf))
print(f"\nFUA: {fua['n_inside']} of {fua['n_sample']} municipalities ({fua['share_inside']*100:.1f}%) lie in {fua['n_fua']} FUAs; "
      f"{fua['fua_multi_n3']} FUAs span more than one NUTS3 ({fua['munis_in_multi']} municipalities)")
print(f"   variance of country-FE residuals absorbed: NUTS3 {fua['absorbed_N3']:.3f} (adj {fua['adj_N3']:.3f}) | FUA {fua['absorbed_FUA']:.3f} (adj {fua['adj_FUA']:.3f}) | both {fua['absorbed_both']:.3f}")
print(f"   F(FUA | NUTS3) = {F_fua_given_n3:.2f} (df {kB-k3}, {nf-kB}), p = {p1:.3f} | F(NUTS3 | FUA) = {F_n3_given_fua:.2f} (df {kB-kF}, {nf-kB}), p = {p2:.3g}")
print(f"   Moran's I (k=10): country FE {fua['I10_country']:.3f}, NUTS3 FE {fua['I10_N3']:.3f}, FUA FE {fua['I10_FUA']:.3f}")
try:
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")
    dd = pd.DataFrame(dict(e=ef, n3=f.N3.values, fua=f.FUA.values, one=1))
    fit = smf.mixedlm("e ~ 1", dd, groups=dd["one"], vc_formula={"n3": "0 + C(n3)", "fua": "0 + C(fua)"}, re_formula="0").fit(reml=True, method=["lbfgs"])
    vc = dict(zip(fit.model.exog_vc.names, fit.vcomp)); tot = sum(vc.values()) + fit.scale
    fua.update(reml_N3=vc["n3"] / tot, reml_FUA=vc["fua"] / tot, reml_municipal=fit.scale / tot)
    print(f"   crossed random-intercept variance shares (REML): NUTS3 {fua['reml_N3']:.3f}, FUA {fua['reml_FUA']:.3f}, municipal {fua['reml_municipal']:.3f}")
except ImportError:
    print("   statsmodels not installed: crossed variance components skipped (optional)")

# ---- (3) joint inventories and identical values
def sig4(x):
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.floor(np.log10(np.abs(x)))
        return np.where(np.isfinite(mag), np.round(x / 10 ** mag, 3) * 10 ** mag, x)


bl = mx[mx.mon == 0].copy()
bl["pc4"] = sig4(bl.total_emissions_reported_tonnes / bl.population_inventory_year)
dup = bl.duplicated(["N3", "pc4"], keep=False)
joint2 = bl.adhesion_type.str.contains("collectively", na=False)
joint1 = bl.adhesion_type.str.contains("individually", na=False)
flag_ids = set(bl.loc[dup | joint2, "GCoM_ID"])
keep = ~sp.GCoM_ID.isin(flag_ids).values
sk = sp[keep].reset_index(drop=True)
Xk = np.column_stack([np.ones(len(sk))] + [sk[c].values.astype(float) for c in BASE + qv])
Wk = knn_weights(sk.latitude.values, sk.longitude.values, k=10)[0]
jt = dict(baseline=len(bl), joint_option2=int(joint2.sum()), joint_option1=int(joint1.sum()), dup_pc_within_N3=int(dup.sum()),
          dup_largest_cluster=int(bl[dup].groupby(["N3", "pc4"]).size().max()) if dup.any() else 0,
          I10_country_excl=moran_value(fe_resid(sk.lnE.values, Xk, sk.country_code.values)[0], Wk),
          I10_N3_excl=moran_value(fe_resid(sk.lnE.values, Xk, sk.N3.values)[0], Wk), n_excl=int((~keep).sum()))
print(f"\njoint SECAP option 2 (shared inventory): {jt['joint_option2']} baseline inventories; option 1: {jt['joint_option1']}")
print(f"identical per-capita emissions (4 significant digits) within a NUTS3: {jt['dup_pc_within_N3']} baseline inventories (largest group {jt['dup_largest_cluster']})")
print(f"excluding these {jt['n_excl']} from the spatial sample: Moran's I (k=10) country FE {jt['I10_country_excl']:.3f}, NUTS3 FE {jt['I10_N3_excl']:.3f}")
save_obj("borders_fua.pkl", dict(border=border, vg=vg, fua=fua, joint=jt))

# ---- Fig. S10 (figures/revision/Figure_R2)
FIG_REV = os.path.join(OUT_DIR, "figures", "revision"); os.makedirs(FIG_REV, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 2.8))
names = {0: "same NUTS3", 1: "across NUTS3 (same NUTS2)", 2: "across NUTS2"}
cols = {0: "#0072B2", 1: "#E69F00", 2: "#56B4E9"}   # colour-vision-safe (Okabe and Ito); markers differ as well
marks = {0: "o", 1: "s", 2: "^"}
for k_, nm in names.items():
    q = vg[vg.cat == k_]
    ax[0].plot(q.km, q["mean"], marks[k_] + "-", color=cols[k_], label=nm)
ax[0].set_xlabel("distance between municipalities (km)"); ax[0].set_ylabel("semivariance of residuals")
ax[0].set_title("(a) Residual dissimilarity by boundary type"); ax[0].legend(frameon=False, fontsize=6)
bb_ = border.set_index("sample")
xs = np.arange(len(bb_)); wdt = 0.38
ax[1].bar(xs - wdt / 2, bb_.rel_N3 * 100, wdt, color=cols[1], label="NUTS3 boundary (same NUTS2)")
ax[1].bar(xs + wdt / 2, bb_.rel_N2 * 100, wdt, color=cols[2], label="NUTS2 boundary")
ax[1].errorbar(xs - wdt / 2, bb_.rel_N3 * 100, yerr=1.96 * bb_.se_d_cross_N3 / bb_.gamma_same_N3 * 100, fmt="none", ecolor="k", capsize=2, lw=0.8)
ax[1].errorbar(xs + wdt / 2, bb_.rel_N2 * 100, yerr=1.96 * bb_.se_d_cross_N2 / bb_.gamma_same_N3 * 100, fmt="none", ecolor="k", capsize=2, lw=0.8)
ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xticks(xs); ax[1].set_xticklabels(bb_.index)
ax[1].set_ylabel("% higher than same-NUTS3 pairs"); ax[1].set_title("(b) Boundary discontinuity (distance-matched)")
ax[1].legend(frameon=False, fontsize=6)
plt.tight_layout()
save_figure(fig, FIG_REV, "Figure_R2")
print("saved: work/borders_fua.pkl, figures/revision/Figure_R2")
