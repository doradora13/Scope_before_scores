# -*- coding: utf-8 -*-
"""
08_repeated_inventories.py - baseline vs monitoring inventories of the same municipality.

(1) Pairs, in four panels (sheet Table2_Pairs of Main_Tables.xlsx; panels B-D = Panels A-C of manuscript Table 3):
    A: change by coverage-change category (also Fig. S8a); B: within-municipality regression; C: change in the
    sectors reported in both inventories; D: pairs with one removed sector, observed change vs the change implied
    by the removed sector's share, ln(1 - share)
(2) Selection logit for submitting a monitoring inventory (observed covariates only, including the baseline OLS
    residual) -> inverse probability weights 1/p (Table S5)
(3) Annualised change of constant-coverage pairs: total, per capita, energy use, carbon intensity
    (identity decomposition, Eq. (11)) (Table 4)
(4) Removal and addition counts by sector and change distributions by category (Table S6)
Input : work/data.pkl
Output: work/repeated.pkl, figures/main/Figure_3 (Fig. S8), Figure_4 (Fig. S11),
        figures/supplementary/Figure_S5 (Fig. S6), Figure_S6 (Fig. S7)
Manuscript: Tables 3, 4, S5, S6; Figs. S6, S7, S8, S11; Sections 5.3-5.4 and 6.4-6.5; Supplementary Sections S5-S6
"""
from lib import *

banner("08 repeated inventories: coverage change, constant-coverage change, selection (Tables 3, 4, S5, S6)")
main = load_obj("data.pkl")["main"]
Xl, _ = design(main, BASE + Q, fe="country_code")
main["res"] = ols(main.lnE.values, Xl)["e"]          # baseline OLS residual (a covariate of the selection model)

d, s1 = build_pairs(main)
corder = ["same coverage", "addition only", "removal only", "both"]
print(f"pairs: {len(d)} municipalities;", d.cat.value_counts().reindex(corder).to_dict())

# ---- panel A: change by coverage-change category
panelA = []
for c in corder:
    g = d[d.cat == c]; gr = g.dR.dropna()
    panelA.append(dict(category=c, n=len(g), median_years=g.dy.median(), mean_all=g.dT.mean(), se_all=g.dT.std() / np.sqrt(len(g)),
                       mean_both=gr.mean(), se_both=gr.std() / np.sqrt(len(gr))))
panelA = pd.DataFrame(panelA)
# ---- panel B: within-municipality regression (Table 3, Panel A)
X, nm = design(d, ["n_rem", "n_add", "dy", "dP"], fe="cty")
oB = ols(d.dT.values, X, cluster=d.nuts3.values)
panelB = pd.DataFrame([dict(var=v, coef=oB["b"][nm.index(v)], se=oB["se"][nm.index(v)], t=oB["t"][nm.index(v)]) for v in ["n_rem", "n_add", "dy", "dP"]])
# ---- panel C: sectors reported in both inventories (Table 3, Panel B)
dd = d.dropna(subset=["dR"]).copy()
for c in ["addition only", "removal only", "both"]:
    dd[c] = (dd.cat == c).astype(int)
X, nm = design(dd, ["addition only", "removal only", "both", "dy"], fe="cty")
oC = ols(dd.dR.values, X, cluster=dd.nuts3.values)
panelC = pd.DataFrame([dict(var=v, coef=oC["b"][nm.index(v)], se=oC["se"][nm.index(v)], t=oC["t"][nm.index(v)])
                       for v in ["addition only", "removal only", "both", "dy"]])
# ---- panel D: single-sector removals (Table 3, Panel C)
g2 = s1.groupby("sector").agg(n=("obs", "size"), share=("share", "mean"), observed=("obs", "mean"), mechanical=("mech", "mean")).sort_values("n", ascending=False)
panelD = g2.reset_index()
panelD.loc[len(panelD)] = ["All single removals", len(s1), s1.share.mean(), s1.obs.mean(), s1.mech.mean()]
panelD["ratio"] = panelD.observed / panelD.mechanical
print(panelA.round(3).to_string(index=False))
print("panel B:", {r.var: (round(r.coef, 3), round(r.t, 2)) for r in panelB.itertuples()}, f"n={oB['n']} R2={oB['r2']:.3f}")
print("panel C:", {r.var: (round(r.coef, 3), round(r.t, 2)) for r in panelC.itertuples()}, f"n={oC['n']} R2={oC['r2']:.3f}")
print(panelD.round(3).to_string(index=False))

# ---- selection model (Table S5) and inverse probability weights
bb = main[main.mon == 0].copy().reset_index(drop=True)
bb["has_mon"] = bb.GCoM_ID.isin(main[main.mon == 1].GCoM_ID.unique()).astype(float)
covs = ["lnPop", "lnGDP", "lnDens", "HDDk", "year_c", "sc", "core5_complete_flag", "co2eq", "res"]
X, nm = design(bb, covs, fe="country_code")
yb = bb.has_mon.values
bl = logit_fit(yb, X)
sel = logit_cluster_se(yb, X, bl, bb.nuts3.values)
p = 1 / (1 + np.exp(-(X @ bl)))
pw = pd.Series(p, index=bb.GCoM_ID.values)
logit_tab = pd.DataFrame([dict(var=c, coef=bl[nm.index(c)], se=sel[nm.index(c)], t=bl[nm.index(c)] / sel[nm.index(c)]) for c in covs])
print("selection logit (t, province-clustered):", {r.var: round(r.t, 1) for r in logit_tab.itertuples()},
      f"| baseline n={len(bb)}, share with monitoring={yb.mean():.3f}")

# ---- constant-coverage pairs (Table 4)
cc = d[d.cat == "same coverage"].copy()
cc["TOT"] = cc.dT / cc.dy * 100
cc["PC"] = (cc.dT - cc.dP) / cc.dy * 100
cc["A"] = cc.dA / cc.dy * 100
cc["I"] = (cc.dT - cc.dA) / cc.dy * 100
cc["w"] = 1 / pw.reindex(cc.id).values
t3 = []
for c, lab in [("TOT", "Total reported emissions"), ("PC", "Per capita"), ("A", "Reported energy use (MWh)"), ("I", "Carbon intensity (t per MWh)")]:
    v = cc[c]; m_ = v.mean(); h = 1.96 * v.std() / np.sqrt(len(v))
    t3.append(dict(component=lab, key=c, mean=m_, ci_lo=m_ - h, ci_hi=m_ + h, median=v.median(), share_neg=float((v < 0).mean()),
                   ipw_mean=np.average(v, weights=cc.w)))
t3 = pd.DataFrame(t3)
tot, tot_w = t3.loc[0, "mean"], t3.loc[0, "ipw_mean"]
t3["share_of_total"] = np.where(t3.key.isin(["A", "I"]), t3["mean"] / tot, np.nan)
t3["share_of_total_ipw"] = np.where(t3.key.isin(["A", "I"]), t3["ipw_mean"] / tot_w, np.nan)
t3_cty = pd.DataFrame([dict(country=c, pairs=len(g), total=g.TOT.median(), per_capita=g.PC.median(), energy=g.A.median(), intensity=g.I.median())
                       for c, g in cc.groupby("cty") if len(g) >= 20])
print(t3.drop(columns="key").round(3).to_string(index=False))
print(f"checks among constant-coverage pairs: energy use available {cc.A.notna().sum()}/{len(cc)}, "
      f"emission-factor type changes {int(cc.ef_change.sum())}, unit changes {int(cc.unit_change.sum())}")

# ---- Table S6
s6_counts = pd.DataFrame([dict(sector=s, removed=int(d.rem.str.contains(s).sum()), added=int(d["add"].str.contains(s).sum())) for s in SECTOR_NAMES])
s6_dist = pd.DataFrame([dict(category=c, n=int((d.cat == c).sum()), **dict(zip(["p10", "p25", "median", "p75", "p90"],
                                                                              np.percentile(d.loc[d.cat == c, "dT"], [10, 25, 50, 75, 90]))))
                        for c in corder])

save_obj("repeated.pkl", dict(pairs=d, single=s1, panelA=panelA, panelB=panelB, panelB_n=oB["n"], panelB_r2=oB["r2"],
                              panelC=panelC, panelC_n=oC["n"], panelC_r2=oC["r2"], panelD=panelD, logit=logit_tab,
                              n_baseline=len(bb), share_mon=float(yb.mean()), cc=cc, t3=t3, t3_cty=t3_cty,
                              s6_counts=s6_counts, s6_dist=s6_dist))

# ---- Fig. S8 (figures/main/Figure_3)
fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 3.0))
x = np.arange(4); w = 0.38
se = lambda v: v.std(ddof=1) / np.sqrt(len(v))
mt = [d.loc[d.cat == c, "dT"].mean() for c in corder]; et = [1.96 * se(d.loc[d.cat == c, "dT"]) for c in corder]
mr = [d.loc[d.cat == c, "dR"].dropna().mean() for c in corder]; er = [1.96 * se(d.loc[d.cat == c, "dR"].dropna()) for c in corder]
ax[0].bar(x - w / 2, mt, w, yerr=et, capsize=2, color="#4c78a8", label="all reported sectors")
ax[0].bar(x + w / 2, mr, w, yerr=er, capsize=2, color="#d9822b", label="sectors in both inventories")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{c}\n(n={(d.cat == c).sum()})" for c in corder]); ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_ylabel("mean change in ln(reported emissions)"); ax[0].set_title("(a) Change by coverage-change category")
ax[0].legend(frameon=False, loc="lower left")
g2p = g2[g2.n >= 6]; yy = np.arange(len(g2p))
ax[1].scatter(g2p.mechanical, yy, s=25, color="grey", label="implied by removed share, ln(1 - share)")
ax[1].scatter(g2p.observed, yy, marker="s", s=25, color="#d9822b", label="observed change in total")
for yi, (mm_, ob) in enumerate(zip(g2p.mechanical, g2p.observed)):
    ax[1].plot([mm_, ob], [yi, yi], color="lightgrey", zorder=0)
ax[1].set_yticks(yy); ax[1].set_yticklabels([f"{s} (n={int(n)})" for s, n in zip(g2p.index, g2p.n)]); ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_xlabel("change in ln(reported total)"); ax[1].set_title("(b) Single-sector removals"); ax[1].legend(frameon=False, loc="lower left")
plt.tight_layout()
save_figure(fig, FIG_MAIN, "Figure_3")

# ---- Fig. S11 (figures/main/Figure_4)
fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 3.0))
cs = [c for c in ["it", "es", "be", "pt"] if (cc.cty == c).sum() >= 20]; pos = np.arange(len(cs))
for off, col, colr in [(-0.18, "TOT", "#4c78a8"), (0.18, "PC", "#d9822b")]:
    bp = ax[0].boxplot([cc.loc[cc.cty == c, col] for c in cs], positions=pos + off, widths=0.32, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(colr); patch.set_alpha(0.6)
ax[0].set_xticks(pos); ax[0].set_xticklabels([f"{c} (n={(cc.cty == c).sum()})" for c in cs]); ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_ylabel("annualised change (% per year)"); ax[0].set_title("(a) Constant coverage: total (blue), per capita (orange)")
comp = [("TOT", "total"), ("A", "energy use"), ("I", "carbon intensity")]; x = np.arange(3); w = 0.36
mu = [cc[c].mean() for c, _ in comp]; ci = [1.96 * cc[c].std() / np.sqrt(len(cc)) for c, _ in comp]
mw = [np.average(cc[c], weights=cc.w) for c, _ in comp]
ax[1].bar(x - w / 2, mu, w, yerr=ci, capsize=2, color="#4c78a8", label="unweighted mean (95% CI)")
ax[1].bar(x + w / 2, mw, w, color="#d9822b", label="selection-weighted mean")
ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in comp]); ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylabel("annualised change (% per year)"); ax[1].set_title("(b) ln E = ln(energy use) + ln(carbon intensity)")
ax[1].legend(frameon=False, loc="lower right")
plt.tight_layout()
save_figure(fig, FIG_MAIN, "Figure_4")

# ---- Fig. S6 (figures/supplementary/Figure_S5): share submitting a monitoring inventory
fig, ax = plt.subplots(1, 3, figsize=(FIG_W, 2.6))
g1 = bb.groupby("inventory_year").has_mon.agg(["mean", "count"]); g1 = g1[g1["count"] >= 30]
ax[0].bar(g1.index, g1["mean"], color="#4c78a8"); ax[0].set_xlabel("baseline inventory year")
ax[0].set_ylabel("share with monitoring inventory"); ax[0].set_title("(a) By baseline year")
g2_ = bb.groupby("sc").has_mon.mean()
ax[1].bar(g2_.index, g2_.values, color="#d9822b"); ax[1].set_xlabel("reported sectors (baseline)"); ax[1].set_title("(b) By coverage")
qu = pd.qcut(bb.res, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]); g3 = bb.groupby(qu, observed=True).has_mon.mean()
ax[2].bar(range(5), g3.values, color="grey"); ax[2].set_xticks(range(5)); ax[2].set_xticklabels(g3.index)
ax[2].set_xlabel("baseline residual quintile"); ax[2].set_title("(c) By baseline residual")
plt.tight_layout()
save_figure(fig, FIG_SUPP, "Figure_S5")

# ---- Fig. S7 (figures/supplementary/Figure_S6): change distributions by category
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for c, col in [("same coverage", "#0072B2"), ("removal only", "#D55E00"), ("addition only", "#000000")]:   # colour-vision-safe (Okabe and Ito)
    v = d.loc[d.cat == c, "dT"]
    ax.hist(v, bins=60, range=(-3, 1.5), alpha=0.6, color=col, label=f"{c} (n={len(v)}, mean {v.mean():.2f})")
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("change in ln(reported emissions)"); ax.set_ylabel("municipalities")
ax.legend(frameon=False); ax.set_title("Change between inventories by coverage change")
plt.tight_layout()
save_figure(fig, FIG_SUPP, "Figure_S6")
print("saved: work/repeated.pkl, figures/main/Figure_3, Figure_4, figures/supplementary/Figure_S5, Figure_S6")
