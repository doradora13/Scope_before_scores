# -*- coding: utf-8 -*-
"""
02_conditional_model.py - conditional model with sector coverage (number of reported sectors); OLS, NUTS3-clustered SEs.

Models: (1) structural covariates, (2) + reporting variables, (3) + country fixed effects,
        (4) + natural splines of the structural covariates.
Input : work/data.pkl
Output: work/model.pkl, figures/main/Figure_1 (Fig. 1)
Manuscript: Eq. (10), Table 2, Fig. 1, Section 6.1 (one reported sector -91.4%, two sectors -43.5%, etc., relative to
            five sectors), residual skewness (Section 6.2)
"""
from lib import *

banner("02 conditional model of reported emissions (Table 2, Fig. 1)")
main = load_obj("data.pkl")["main"]
y = main.lnE.values

X1, n1 = design(main, BASE)
X2, n2 = design(main, BASE + Q)
X3, n3 = design(main, BASE + Q, fe="country_code")
X4 = np.column_stack([X3] + spline_controls(main))
n4 = n3 + [f"spline_{j}" for j in range(X4.shape[1] - X3.shape[1])]

specs = []
for X, nm in [(X1, n1), (X2, n2), (X3, n3), (X4, n4)]:
    op = ols(y, X, cluster=main.nuts3.values)      # NUTS3 clusters (reported in the manuscript)
    om = ols(y, X, cluster=main.GCoM_ID.values)    # municipality clusters (for reference)
    st = skew_tests(op["e"])
    specs.append(dict(names=nm, b=op["b"], se=op["se"], t=op["t"], se_muni=om["se"], t_muni=om["t"],
                      n=op["n"], r2=op["r2"], skew=st["skew"], m3t=st["m3t"]))

implied = {}
for v in ["sc1", "sc2", "sc3", "sc4", "sc6"]:
    implied[v] = [np.exp(specs[s]["b"][specs[s]["names"].index(v)]) - 1 for s in (2, 3)]   # models (3), (4)

for j, s in enumerate(specs, 1):
    print(f"model ({j}): n={s['n']} R2={s['r2']:.3f} skewness={s['skew']:.2f} M3T={s['m3t']:.1f}")
print("coverage coefficients, model (3) [t, province-clustered] and implied difference vs five sectors:")
for v in ["sc1", "sc2", "sc3", "sc4", "sc6"]:
    i = n3.index(v)
    print(f"  {v}: {specs[2]['b'][i]:.3f} (t={specs[2]['t'][i]:.1f}) -> {implied[v][0] * 100:.1f}%")
save_obj("model.pkl", dict(specs=specs, implied=implied, n_clusters=int(main.nuts3.nunique()),
                           n_muni=int(main.GCoM_ID.nunique())))

# ---- Fig. 1
cty = country_group(main)
order = ["it", "es", "be", "pt", "el", "other"]
fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
ct = pd.crosstab(cty, main.sc, normalize="index").loc[order]
ct.plot(kind="bar", stacked=True, ax=ax[0], colormap="viridis", width=0.75)
ax[0].set_ylabel("share of inventories"); ax[0].set_xlabel(""); ax[0].set_title("(a) Reported sectors by country")
ax[0].tick_params(axis="x", rotation=0)
ax[0].legend(title="sectors", frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
data = [main.loc[main.sc == s, "reported_emissions_per_capita"].values for s in range(1, 7)]
ax[1].boxplot(data, showfliers=False)
ax[1].set_xticklabels([f"{s}\n({len(d)})" for s, d in zip(range(1, 7), data)])
ax[1].set_xlabel("number of reported sectors (n)"); ax[1].set_ylabel("t CO$_2$(eq) per capita")
ax[1].set_title("(b) Reported per-capita emissions by coverage")
plt.tight_layout()
save_figure(fig, FIG_MAIN, "Figure_1", tight=True)   # tight=True keeps the title of panel (b) from being clipped on the right
print("saved: work/model.pkl, figures/main/Figure_1")
