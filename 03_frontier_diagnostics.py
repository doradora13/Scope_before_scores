# -*- coding: utf-8 -*-
"""
03_frontier_diagnostics.py - conventional stochastic frontier (SFA) diagnostics.

(A) cost frontiers by inefficiency distribution (half-normal, truncated normal, exponential) and a production-type
    frontier (one-sided term subtracted)
(B) residual skewness, Coelli M3T, cost and production-type sigma_u and LR by subsample
(C) sensitivity to trimming the residuals (1%, 2.5%, 5%; lower tail only or symmetric)
Input : work/data.pkl
Output: work/frontier.pkl, figures/main/Figure_2 (Fig. S3), figures/supplementary/Figure_S2 (Fig. S2)
Manuscript: Table S2, Figs. S2-S3, Section 6.2, Supplementary Section S2 (cost-frontier sigma_u ~ 0, production type
            0.52 [LR 502], 0.378 after 1% lower-tail trimming)
"""
from lib import *
from lib import _loglik_vec

banner("03 conventional frontier diagnostics (Table S2, Figs. S2-S3)")
main = load_obj("data.pkl")["main"]
y = main.lnE.values
X0, _ = design(main, BASE)
X2, _ = design(main, BASE + Q, fe="country_code")

# ---- (A) by one-sided distribution
dist_rows = []
for dist, lab in [("hn", "half-normal"), ("tn", "truncated-normal"), ("exp", "exponential")]:
    r = sfa(y, X0, dist=dist, compute_se=False)
    dist_rows.append(dict(model=f"Cost frontier ({lab})", covariates="structural", sigma_u=r["su"], sigma_v=r["sv"],
                          LR_sigma_u0=r["lr_sigma_u"], mean_efficiency=r["ce"].mean()))
r = sfa(y, X2, "hn", compute_se=False)
dist_rows.append(dict(model="Cost frontier (half-normal)", covariates="structural + reporting + country FE", sigma_u=r["su"],
                      sigma_v=r["sv"], LR_sigma_u0=r["lr_sigma_u"], mean_efficiency=r["ce"].mean()))
r = sfa(-y, -X2, "hn", compute_se=False)
dist_rows.append(dict(model="Production-type frontier (half-normal; one-sided term subtracted)",
                      covariates="structural + reporting + country FE", sigma_u=r["su"], sigma_v=r["sv"],
                      LR_sigma_u0=r["lr_sigma_u"], mean_efficiency=np.nan))
print(pd.DataFrame(dist_rows).round(3).to_string(index=False), flush=True)

# ---- (B) by subsample
subs = [("All inventories", main),
        ("Cities (degree of urbanisation 1)", main[main.degree_urbanisation == 1]),
        ("Complete reporters (>=5 sectors, core five)", main[(main.sc >= 5) & (main.core5_complete_flag == 1)]),
        ("CO2 units", main[main.co2eq == 0]), ("CO2-equivalent units", main[main.co2eq == 1]),
        ("Italy", main[main.country_code == "it"]), ("Spain", main[main.country_code == "es"]),
        ("Belgium", main[main.country_code == "be"]),
        ("Population >= 50,000", main[main.population_inventory_year >= 50000]),
        ("Population matched (no mismatch flag)", main[main.pop_mismatch == 0])]
sub_rows = []
for name, s in subs:
    s = s.reset_index(drop=True)
    fe = "country_code" if s.country_code.nunique() > 1 else None
    cols = BASE + [c for c in Q if s[c].nunique() > 1]
    X, _ = design(s, cols, fe=fe)
    o = ols(s.lnE.values, X); st = skew_tests(o["e"])
    rc = sfa(s.lnE.values, X, "hn", compute_se=False)
    rp = sfa(-s.lnE.values, -X, "hn", compute_se=False)
    sub_rows.append(dict(sample=name, n=len(s), skewness=st["skew"], M3T=st["m3t"], cost_sigma_u=rc["su"], cost_LR=rc["lr_sigma_u"],
                         production_sigma_u=rp["su"], production_LR=rp["lr_sigma_u"]))
    print(f"  {name}: done", flush=True)
print(pd.DataFrame(sub_rows).round(3).to_string(index=False))

# ---- (C) trimming sensitivity
e = ols(y, X2)["e"]
trim_rows = []
for q in [0.01, 0.025, 0.05]:
    for rule in ["lower tail", "symmetric"]:
        keep = e > np.quantile(e, q) if rule == "lower tail" else (e > np.quantile(e, q)) & (e < np.quantile(e, 1 - q))
        rc = sfa(y[keep], X2[keep], "hn", compute_se=False)
        trim_rows.append(dict(rule=rule, share_trimmed=q if rule == "lower tail" else 2 * q, n=int(keep.sum()),
                              skewness=skew_tests(ols(y[keep], X2[keep])["e"])["skew"], sigma_u=rc["su"], LR=rc["lr_sigma_u"]))
print(pd.DataFrame(trim_rows).round(3).to_string(index=False))
res = dict(dist=pd.DataFrame(dist_rows), subs=pd.DataFrame(sub_rows), trim=pd.DataFrame(trim_rows), resid=e)

# ---- Fig. S3 (figures/main/Figure_2)
sb, tr = res["subs"], res["trim"]
short = {"All inventories": "All", "Cities (degree of urbanisation 1)": "Cities", "Complete reporters (>=5 sectors, core five)": "Complete reporters",
         "CO2 units": "CO$_2$ units", "CO2-equivalent units": "CO$_2$-eq units", "Italy": "Italy", "Spain": "Spain", "Belgium": "Belgium",
         "Population >= 50,000": "Pop. $\\geq$ 50,000", "Population matched (no mismatch flag)": "Population matched"}
fig, ax = plt.subplots(1, 3, figsize=(FIG_W, 2.7))
ax[0].barh([short[s] for s in sb["sample"]], sb.skewness, color=np.where(sb.skewness < 0, "#4c78a8", "#d9822b"))
ax[0].axvline(0, color="k", lw=0.8); ax[0].invert_yaxis()
ax[0].set_xlabel("residual skewness"); ax[0].set_title("(a) Skewness by subsample")
q = np.linspace(0.001, 0.999, 400); eq = np.quantile(e, q); sn = e.std(); sl = np.mean(np.abs(e - np.median(e)))
ax[1].plot(stats.norm.ppf(q) * sn, eq, ".", ms=2, label="normal")
ax[1].plot(stats.laplace.ppf(q) * sl, eq, ".", ms=2, label="Laplace")
lim = [-3.5, 2]
ax[1].plot(lim, lim, "k--", lw=0.7); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("theoretical quantile"); ax[1].set_ylabel("residual quantile"); ax[1].set_title("(b) Quantile-quantile plot")
ax[1].legend(frameon=False, markerscale=4)
for rule, mk in [("lower tail", "o-"), ("symmetric", "s--")]:
    t = tr[tr.rule == rule]
    ax[2].plot(t.share_trimmed * 100, t.sigma_u, mk, label=f"{rule} trimming")
ax[2].axhline(0, color="grey", lw=0.6)
ax[2].set_xlabel("share of observations trimmed (%)"); ax[2].set_ylabel("cost-frontier $\\sigma_u$")
ax[2].set_title("(c) Inefficiency after trimming"); ax[2].legend(frameon=False)
plt.tight_layout()
save_figure(fig, FIG_MAIN, "Figure_2", tight=True)   # tight=True keeps the title and x-axis label of panel (c) from being clipped

# ---- Fig. S2: profile likelihood of sigma_u (cost type vs production type)
k = X0.shape[1]
grid = np.linspace(0.01, 0.7, 15)
prof = {"cost type (one-sided term added)": [], "production type (one-sided term subtracted)": []}
for su in grid:
    for lab, (yy_, XX) in zip(prof.keys(), [(y, X0), (-y, -X0)]):
        o = ols(yy_, XX)
        th0 = np.r_[o["b"], np.log(np.sqrt(max(o["sig2"] - (1 - 2 / np.pi) * su ** 2, 1e-4)))]
        rr_ = optimize.minimize(lambda t: -np.sum(_loglik_vec(np.r_[t[:k], np.log(su), t[k]], yy_, XX, "hn")), th0, method="L-BFGS-B")
        prof[lab].append(-rr_.fun)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for lab, v in prof.items():
    ax.plot(grid, np.array(v) - max(v), "o-", ms=3, label=lab)
ax.set_xlabel("$\\sigma_u$ (fixed)"); ax.set_ylabel("profile log-likelihood minus maximum")
ax.legend(frameon=False); ax.set_title("Profile likelihood of the inefficiency scale")
plt.tight_layout()
save_figure(fig, FIG_SUPP, "Figure_S2")
res["profile"] = dict(grid=grid, **{k_: np.array(v) for k_, v in prof.items()})
save_obj("frontier.pkl", res)
print("saved: work/frontier.pkl, figures/main/Figure_2, figures/supplementary/Figure_S2")
