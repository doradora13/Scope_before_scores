# -*- coding: utf-8 -*-
"""
09_spatial.py - spatial autocorrelation of OLS residuals (Moran's I, Eq. (12)), country vs NUTS3 fixed effects.

Sample: baseline inventories whose coordinates are not on the integer-degree grid. Weights: row-standardised
k nearest neighbours (k = 5, 10, 20); 199 permutations.
Within countries (Italy, Spain, Belgium; k = 10): before and after NUTS3 fixed effects.
Input : work/data.pkl
Output: work/spatial.pkl, figures/main/Figure_5 (Fig. S9)
Manuscript: Table S7, Fig. S9, Section 6.6, Supplementary Section S7 (country FE 0.186/0.173/0.151 -> NUTS3 FE
            -0.004/0.003/0.008 for k = 5/10/20; Spain 0.080)
"""
from lib import *

banner("09 spatial autocorrelation of residuals (Table S7, Fig. S9)")
main = load_obj("data.pkl")["main"]
sp = main[(main.mon == 0) & (main.int_coords == 0)].copy().reset_index(drop=True)
qv = [c for c in Q if sp[c].nunique() > 1]
ec = ols(sp.lnE.values, design(sp, BASE + qv, fe="country_code")[0])["e"]
en = ols(sp.lnE.values, design(sp, BASE + qv, fe="nuts3")[0])["e"]

rows = []
for k in [5, 10, 20]:
    W, _ = knn_weights(sp.latitude.values, sp.longitude.values, k=k)
    a = morans_i(ec, W, nperm=199); b_ = morans_i(en, W, nperm=199)
    rows.append(dict(k=k, I_country=a["I"], z_country=a["z"], p_country=a["p_perm"], I_province=b_["I"], z_province=b_["z"], p_province=b_["p_perm"]))
moran = pd.DataFrame(rows)
rows = []
for c in ["it", "es", "be"]:
    s_ = sp[sp.country_code == c].reset_index(drop=True)
    qv_ = [x for x in Q if s_[x].nunique() > 1]
    e0 = ols(s_.lnE.values, design(s_, BASE + qv_)[0])["e"]
    e1 = ols(s_.lnE.values, design(s_, BASE + qv_, fe="nuts3")[0])["e"]
    W, _ = knn_weights(s_.latitude.values, s_.longitude.values, k=10)
    a = morans_i(e0, W, nperm=199); b_ = morans_i(e1, W, nperm=199)
    rows.append(dict(country=c, n=len(s_), I_no_province=a["I"], z_no_province=a["z"], I_province=b_["I"], z_province=b_["z"]))
within = pd.DataFrame(rows)
print(f"spatial sample n={len(sp)}")
print(moran.round(3).to_string(index=False))
print(within.round(3).to_string(index=False))
save_obj("spatial.pkl", dict(moran=moran, within=within, n=len(sp)))

# ---- Fig. S9 (figures/main/Figure_5)
W10, _ = knn_weights(sp.latitude.values, sp.longitude.values, k=10)
lim = np.quantile(np.abs(np.r_[ec, en]), 0.98)
fig, ax = plt.subplots(2, 2, figsize=(FIG_W, 6.3))
N3_LABEL = "NUTS3 fixed effects" if USE_REAL_NUTS3 else "province fixed effects"   # panel title follows USE_REAL_NUTS3 in lib.py
for j, (ee, lab) in enumerate([(ec, "country fixed effects"), (en, N3_LABEL)]):
    sc_ = ax[0, j].scatter(sp.longitude, sp.latitude, c=np.clip(ee, -lim, lim), s=1.2, cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
    ax[0, j].set_xlim(-10, 30); ax[0, j].set_ylim(34, 62); ax[0, j].set_xlabel("longitude"); ax[0, j].set_ylabel("latitude")
    ax[0, j].set_title(f"({'ab'[j]}) Residual after {lab}")
    plt.colorbar(sc_, ax=ax[0, j], shrink=0.85, label="residual (log points)")
    z = (ee - ee.mean()) / ee.std(); lag = W10 @ z; slope = np.polyfit(z, lag, 1)[0]
    ax[1, j].scatter(z, lag, s=1.2, alpha=0.4, rasterized=True)
    xx = np.linspace(-4, 4, 10); ax[1, j].plot(xx, slope * xx, "r-", lw=1)
    ax[1, j].axhline(0, color="k", lw=0.5); ax[1, j].axvline(0, color="k", lw=0.5); ax[1, j].set_xlim(-4, 4)
    ax[1, j].set_xlabel("standardised residual"); ax[1, j].set_ylabel("mean of 10 nearest neighbours")
    ax[1, j].set_title(f"({'cd'[j]}) Moran scatter: slope {slope:.3f}")
plt.tight_layout()
save_figure(fig, FIG_MAIN, "Figure_5", tight=True)   # no boundary disclaimer in the figure: no boundary lines are drawn
print("saved: work/spatial.pkl, figures/main/Figure_5")
