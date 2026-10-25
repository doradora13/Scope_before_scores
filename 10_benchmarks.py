# -*- coding: utf-8 -*-
"""
10_benchmarks.py - peer-quantile benchmarks for complete reporters (conditional tau-quantile regression) and screening
stability.

Gap = 1 - exp(q_tau - ln E) (positive only above the benchmark; Eq. (13)). Aggregate gap = sum of emissions above the
benchmark / sum of reported emissions.
Screening stability: the tau = 0.25 top-quartile classification is repeated in 30 bootstrap replications that resample
NUTS3 regions with replacement within countries.
Realised-change reference: quantiles of the annualised change of constant-coverage pairs (from stage 08).
Input : work/data.pkl, work/repeated.pkl
Output: work/benchmarks.pkl, figures/main/Figure_6 (Fig. 3)
Manuscript: Table S8, Fig. 3, Section 6.3 (gaps 12.7/21.6/34.2% at tau = 0.50/0.25/0.10; top quartile in at least 90%
            of the 30 replications: 518 inventories (16.7%) and 17.7% unstable with real NUTS3 codes, 540 inventories
            and 16.5% unstable with the GDP-value proxy)
"""
from lib import *

B, SEED = 30, 11     # keep these values to reproduce the manuscript
banner("10 peer-quantile benchmarks and screening stability (Table S8, Fig. 3)")
main = load_obj("data.pkl")["main"]
cc = load_obj("repeated.pkl")["cc"]
hq = main[(main.sc >= 5) & (main.core5_complete_flag == 1)].reset_index(drop=True)
yq = hq.lnE.values
qcols = BASE + [c for c in ["sc6", "co2eq", "lca", "mon"] if hq[c].nunique() > 1]
XQ, _ = design(hq, qcols, fe="country_code")
E = hq.total_emissions_reported_tonnes.values

bench = {}
for tau in [0.10, 0.25, 0.50]:
    bq = qreg(yq, XQ, tau); qv_ = XQ @ bq
    rr = np.clip(1 - np.exp(qv_ - yq), 0, None)
    bench[tau] = dict(rr=rr, q=qv_, agg=1 - np.sum(E * np.exp(np.minimum(qv_ - yq, 0))) / E.sum(),
                      share_below=float(np.mean(rr == 0)), med_above=float(np.median(rr[rr > 0])))
    print(f"  tau={tau:.2f}: aggregate gap {bench[tau]['agg'] * 100:.1f}% | at/below {bench[tau]['share_below'] * 100:.1f}% | "
          f"median gap above {bench[tau]['med_above'] * 100:.1f}%", flush=True)

rng = np.random.default_rng(SEED)
cls = np.zeros((B, len(hq)))
groups = {c: g.nuts3.unique() for c, g in hq.groupby("country_code")}
rows_by_prov = {p_: np.where(hq.nuts3.values == p_)[0] for p_ in hq.nuts3.unique()}
for b_ in range(B):
    idx = np.concatenate([rows_by_prov[p_] for c, ps in groups.items() for p_ in rng.choice(ps, len(ps), replace=True)])
    bq = qreg(yq[idx], XQ[idx], 0.25)
    rr = np.clip(1 - np.exp(XQ @ bq - yq), 0, None)
    cls[b_] = rr > np.quantile(rr, 0.75)
ptop = cls.mean(0)
stable = ptop >= 0.9
unstable = (ptop > 0.1) & (ptop < 0.9)
ref = dict(top_quartile=np.percentile(cc.TOT, 25), median=cc.TOT.median())
print(f"  consistently flagged (top quartile in >=90% of replications): {int(stable.sum())} of {len(hq)} ({stable.mean() * 100:.1f}%) | unstable: {unstable.mean() * 100:.1f}%")
print(f"  realised-change reference (constant coverage): 25th percentile {ref['top_quartile']:.2f}%/yr, median {ref['median']:.2f}%/yr")
save_obj("benchmarks.pkl", dict(bench={t: {k: v for k, v in bench[t].items() if k in ("agg", "share_below", "med_above")} for t in bench},
                                stable=int(stable.sum()), unstable_share=float(unstable.mean()), n_hq=len(hq), ptop=ptop, ref=ref))

# ---- Fig. 3 (figures/main/Figure_6)
fig, ax = plt.subplots(1, 3, figsize=(FIG_W, 2.8))
b25 = bench[0.25]
ax[0].scatter(hq.lnEpc, hq.lnEpc + np.minimum(b25["q"] - yq, 0), s=1.5, alpha=0.4, rasterized=True)
l_ = [-1.5, 3.5]; ax[0].plot(l_, l_, "k--", lw=0.7)
ax[0].set_xlabel("ln reported emissions per capita"); ax[0].set_ylabel("ln benchmark per capita ($\\tau$ = 0.25)")
ax[0].set_title("(a) Reported vs benchmark")
for tau, col in [(0.10, "#d9822b"), (0.25, "#4c78a8"), (0.50, "grey")]:
    ax[1].hist(bench[tau]["rr"], bins=40, range=(0, 1), alpha=0.5, color=col, label=f"$\\tau$ = {tau:.2f}: aggregate {bench[tau]['agg'] * 100:.1f}%")
ax[1].set_xlabel("gap to benchmark (share of reported emissions)"); ax[1].set_ylabel("inventories")
ax[1].set_title("(b) Benchmark gaps by target quantile"); ax[1].legend(frameon=False)
ax[2].scatter(hq.longitude, hq.latitude, s=1, c="lightgrey", rasterized=True)
ax[2].scatter(hq.longitude[stable], hq.latitude[stable], s=2, c="#d9822b", label=f"flagged (n={stable.sum()})", rasterized=True)
ax[2].set_xlim(-10, 30); ax[2].set_ylim(34, 62); ax[2].set_xlabel("longitude"); ax[2].set_ylabel("latitude")
ax[2].set_title("(c) Consistently flagged"); ax[2].legend(frameon=False, loc="upper right", markerscale=4)
plt.tight_layout()
save_figure(fig, FIG_MAIN, "Figure_6", tight=True)   # panel (c) wording matches the text and caption; the boundary disclaimer is in the caption, not the figure
print("saved: work/benchmarks.pkl, figures/main/Figure_6")
