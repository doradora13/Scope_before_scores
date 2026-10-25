# -*- coding: utf-8 -*-
"""
04_two_tier.py - exponential two-tier stochastic frontier y = Xb + v + u - w (u: unexplained excess, w: coverage shortfall).

Specifications: (a) structural covariates, (b) + reporting variables, (c) + country fixed effects,
                (d) + sigma_w depends on reporting variables (ln sigma_w = delta0 + z'delta),
                (e) + sigma_u depends on municipal characteristics (ln sigma_u = gamma0 + h'gamma, standardised).
Standard errors: NUTS3-cluster sandwich. LR(sigma_u = 0) is the comparison with the shortfall-only model
(exponential production type).
Input : work/data.pkl
Output: work/twotier.pkl, figures/supplementary/Figure_S3 (Fig. S4)
Manuscript: Table S3 (specifications, delta, gamma, components by number of sectors), Fig. S4, Section 6.2,
            Supplementary Section S3
"""
import time
from lib import *

banner("04 two-tier stochastic frontier, specifications (a)-(e) (Table S3, Fig. S4)")
main = load_obj("data.pkl")["main"]
y = main.lnE.values
cl = main.nuts3.values
Xa, na = design(main, BASE)
Xb, nb = design(main, BASE + Q)
Xc, nc = design(main, BASE + Q, fe="country_code")
Zw = main[ZW_COLS].values.astype(float)
Zu_raw = main[["lnPop", "lnGDP", "lnDens", "city_flag"]].values.astype(float)
Zu = (Zu_raw - Zu_raw.mean(0)) / Zu_raw.std(0)


def wonly_llf(y, X):
    """Shortfall-only model (w only) = production-type SFA with an exponential one-sided term subtracted."""
    return sfa(-y, -X, "exp", compute_se=False)["llf"]


out, lr = {}, {}
t0 = time.time()
for tag, X in [("a", Xa), ("b", Xb), ("c", Xc)]:
    r = sfa_2tier_fast(y, X, cluster=cl, compute_se=True)
    out[tag] = r
    lr[tag] = 2 * (r["llf"] - wonly_llf(y, X))
    print(f"({tag}) sv={r['sv']:.3f} su={r['su']:.3f} sw={r['sw']:.3f} llf={r['llf']:.1f} LR(sigma_u=0)={lr[tag]:.1f} "
          f"[{time.time() - t0:.0f}s]", flush=True)
rc = out["c"]
rd = sfa_2tier_fast(y, Xc, Zw=Zw, cluster=cl, compute_se=True, start=np.r_[rc["theta"], np.zeros(Zw.shape[1])])
out["d"] = rd
print(f"(d) sv={rd['sv']:.3f} su={rd['su']:.3f} sw={rd['sw']:.3f} llf={rd['llf']:.1f} LR(hetero-w)={2 * (rd['llf'] - rc['llf']):.1f} "
      f"[{time.time() - t0:.0f}s]", flush=True)
print("   delta:", "  ".join(f"{c}={rd['dw'][j]:.3f} (t={rd['dw'][j] / rd['dw_se'][j]:.1f})" for j, c in enumerate(ZW_COLS)))
re_ = sfa_2tier_fast(y, Xc, Zu=Zu, Zw=Zw, cluster=cl, compute_se=True,
                     start=np.r_[rd["theta"][:Xc.shape[1] + 3], np.zeros(Zu.shape[1]), rd["dw"]])
out["e"] = re_
print(f"(e) llf={re_['llf']:.1f} LR(hetero-u)={2 * (re_['llf'] - rd['llf']):.1f} [{time.time() - t0:.0f}s]")
print("   gamma:", "  ".join(f"{c}={re_['du'][j]:.3f} (t={re_['du'][j] / re_['du_se'][j]:.1f})"
                           for j, c in enumerate(["lnPop", "lnGDP", "lnDens", "city_flag"])))
comp = main.assign(w=rd["w"], u=rd["u"]).groupby("sc")[["w", "u"]].mean()
print("spec (d) mean components by number of sectors:\n", comp.round(3).to_string())

keep = ["theta", "se", "sv", "su", "sw", "llf", "aic", "bic", "u", "w", "dw", "dw_se", "du", "du_se"]
save_obj("twotier.pkl", dict(out={k: {f: v[f] for f in keep if f in v} for k, v in out.items()}, lr=lr, k=Xc.shape[1],
                             comp=comp, zw_cols=ZW_COLS))

# ---- Fig. S4 (figures/supplementary/Figure_S3)
kk = Xc.shape[1]
fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 2.9))
ax[0].bar(comp.index - 0.18, comp.w, 0.36, color="#d9822b", label="E[w|$\\varepsilon$] shortfall")
ax[0].bar(comp.index + 0.18, comp.u, 0.36, color="#4c78a8", label="E[u|$\\varepsilon$] unexplained excess")
ax[0].set_xlabel("number of reported sectors"); ax[0].set_ylabel("mean component (log points)")
ax[0].set_title("(a) Two-tier components by coverage"); ax[0].legend(frameon=False)
lsw, se_lsw, dw, dse = rd["theta"][kk + 2], rd["se"][kk + 2], rd["dw"], rd["dw_se"]
grp = [("$\\leq$3", [0]), ("4", [1]), ("5 (ref.)", []), ("6", [2])]
vals = [np.exp(lsw + sum(dw[j] for j in ix)) for _, ix in grp]
ses = [np.sqrt(se_lsw ** 2 + sum(dse[j] ** 2 for j in ix)) for _, ix in grp]
lo = [np.exp(np.log(v) - 1.96 * s_) for v, s_ in zip(vals, ses)]
hi = [np.exp(np.log(v) + 1.96 * s_) for v, s_ in zip(vals, ses)]
ax[1].bar(range(4), vals, color="#d9822b")
ax[1].errorbar(range(4), vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none", ecolor="k", capsize=3)
ax[1].axhline(rd["su"], color="#4c78a8", ls="--", label=f"$\\sigma_u$ = {rd['su']:.3f}")
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([g for g, _ in grp])
ax[1].set_xlabel("number of reported sectors"); ax[1].set_ylabel("$\\sigma_w$ (approx. 95% CI)")
ax[1].set_title("(b) Shortfall scale by coverage"); ax[1].legend(frameon=False)
plt.tight_layout()
save_figure(fig, FIG_SUPP, "Figure_S3")
print("saved: work/twotier.pkl, figures/supplementary/Figure_S3")
