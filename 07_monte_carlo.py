# -*- coding: utf-8 -*-
"""
07_monte_carlo.py - Monte Carlo: does symmetric heavy-tailed noise create an 'excess tier' in the exponential two-tier model?

Nine data-generating processes x R replications (n = N). For each scenario the script records
  the rejection rates of LR(sigma_u = 0) and LR(sigma_u = sigma_w) (asymmetry test),
  the rejection rate for spurious heteroskedasticity in an unrelated random variable, and the estimate and rejection
  rate for true heteroskedasticity (delta = 0.8).
Input : none (data are simulated); the figure also uses work/asym.pkl (stage 05)
Output: work/montecarlo.pkl, tables/montecarlo.csv, figures/supplementary/Figure_S4 (Fig. S5)
Manuscript: Table S4, Fig. S5, Supplementary Section S4 (under symmetric heavy-tailed noise LR(sigma_u = 0) rejects in
            100% of replications, while the asymmetry test rejects in 0-6%)
Unrestricted and restricted (sigma_u = sigma_w) models are fitted as in stage 05 (several starting values + grid over
the common scale, full convergence).
"""
import time
import warnings
from lib import *

# hide the RuntimeWarnings (overflow) raised when the optimiser passes through extreme parameter regions; results are unaffected
warnings.filterwarnings("ignore", category=RuntimeWarning)

R, N, SEED = 50, 2000, 123       # keep these values to reproduce the manuscript
BATCHES = [("A", ["normal", "laplace", "t3", "t5"]),
           ("B", ["contaminated", "true2tier", "wonly_normal", "wonly_laplace", "wonly_hetero_laplace"])]
banner(f"07 Monte Carlo: {sum(len(s) for _, s in BATCHES)} scenarios x {R} replications, n = {N} (Table S4, Fig. S5)")


def wonly(y, X):
    return sfa(-y, -X, "exp", compute_se=False)["llf"]


def scen(name, rng):
    n = N
    X = np.c_[np.ones(n), rng.normal(size=n), rng.normal(size=n)]
    b = np.array([1.0, 0.5, -0.3])
    z = (rng.uniform(size=n) < 0.4).astype(float)
    sd = 0.45
    if name == "normal": e = rng.normal(0, sd, n)
    elif name == "laplace": e = rng.laplace(0, sd / np.sqrt(2), n)
    elif name == "t3": e = rng.standard_t(3, n) * sd / np.sqrt(3)
    elif name == "t5": e = rng.standard_t(5, n) * sd / np.sqrt(5 / 3)
    elif name == "contaminated": e = np.where(rng.uniform(size=n) < 0.1, rng.normal(0, 1.0, n), rng.normal(0, 0.35, n))
    elif name == "true2tier": e = rng.normal(0, 0.15, n) + rng.exponential(0.25, n) - rng.exponential(0.30, n)
    elif name == "wonly_normal": e = rng.normal(0, 0.32, n) - rng.exponential(0.30, n)
    elif name == "wonly_laplace": e = rng.laplace(0, 0.25, n) - rng.exponential(0.30, n)
    elif name == "wonly_hetero_laplace": e = rng.laplace(0, 0.25, n) - rng.exponential(1, n) * np.exp(np.log(0.22) + 0.8 * z)
    else: raise ValueError(name)
    return X @ b + e, X, z


rows = []
for batch, names in BATCHES:
    rng = np.random.default_rng(SEED)          # each batch (A, B) restarts from the same seed
    for name in names:
        t0 = time.time()
        rec = []
        for r in range(R):
            y, X, z = scen(name, rng)
            f = sfa_2tier_best(y, X)                                      # unrestricted model: three starting values
            lw = wonly(y, X)
            ls = sfa_2tier_sym(y, X, f["theta"], profile=True)["llf"]   # restricted model (sigma_u = sigma_w): four starting values + grid
            zr = (rng.uniform(size=N) < 0.4).astype(float)[:, None]    # random 'reporting' variable unrelated to the data-generating process
            fh = sfa_2tier_fast(y, X, Zw=zr, start=np.r_[f["theta"], 0.0])
            ftrue = sfa_2tier_fast(y, X, Zw=z[:, None], start=np.r_[f["theta"], 0.0])
            rec.append(dict(sv=f["sv"], su=f["su"], sw=f["sw"], LR_su=2 * (f["llf"] - lw), LR_sym=2 * (f["llf"] - ls),
                            LR_het_random=2 * (fh["llf"] - f["llf"]), dw_true=ftrue["dw"][0], LR_het_true=2 * (ftrue["llf"] - f["llf"]),
                            meanCE=f["ce"].mean(), minEu=f["u"].min()))
        d = pd.DataFrame(rec)
        rows.append(dict(scenario=name, n=N, R=R, sv=d.sv.mean().round(3), su=d.su.mean().round(3), sw=d.sw.mean().round(3),
                         su_over_sw=(d.su / d.sw).mean().round(2), rej_LRsu_1pct=(d.LR_su > 5.412).mean().round(2),
                         mean_LRsu=d.LR_su.mean().round(1), rej_asym_1pct=(d.LR_sym > 6.635).mean().round(2),
                         mean_LRsym=d.LR_sym.mean().round(1), rej_het_random_1pct=(d.LR_het_random > 6.635).mean().round(2),
                         dw_true=d.dw_true.mean().round(2), rej_het_true=(d.LR_het_true > 6.635).mean().round(2),
                         meanCE=d.meanCE.mean().round(3), minEu=d.minEu.mean().round(3)))
        print(f"  {name:22s} done in {time.time() - t0:.0f}s", flush=True)
mc = pd.DataFrame(rows)
print(mc[["scenario", "sv", "su", "sw", "rej_LRsu_1pct", "rej_asym_1pct", "rej_het_random_1pct", "dw_true", "rej_het_true"]].to_string(index=False))
mc.to_csv(os.path.join(TAB_DIR, "montecarlo.csv"), index=False, encoding="utf-8-sig")
save_obj("montecarlo.pkl", mc)

# ---- Fig. S5 (figures/supplementary/Figure_S4): (a) Monte Carlo rejection rates, (b) sigma_u vs sigma_w by sample in the data (stage 05)
asym = load_obj("asym.pkl")
fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 3.1))
x = np.arange(len(mc))
ax[0].bar(x - 0.2, mc.rej_LRsu_1pct, 0.4, color="#d9822b", label="test of $\\sigma_u$ = 0")
ax[0].bar(x + 0.2, mc.rej_asym_1pct, 0.4, color="#4c78a8", label="test of $\\sigma_u$ = $\\sigma_w$")
ax[0].set_xticks(x); ax[0].set_xticklabels(mc.scenario, rotation=40, ha="right")
ax[0].axhline(0.01, color="k", ls=":", lw=0.8); ax[0].set_ylabel("rejection rate at 1%")
ax[0].set_title(f"(a) Monte Carlo (n = {N:,}; {R} replications)"); ax[0].legend(frameon=False)
colr = np.where(asym.LR_asym > CHI2_1_CRIT_1PCT, np.where(asym.su > asym.sw, "#d9822b", "#4c78a8"), "grey")
ax[1].scatter(asym.sw, asym.su, s=np.clip(asym.LR_asym, 5, 150) / 2 + 8, c=colr)
ax[1].plot([0, 0.65], [0, 0.65], "k--", lw=0.7); ax[1].set_xlim(0, 0.65); ax[1].set_ylim(0.1, 0.35)
ax[1].set_xlabel("$\\sigma_w$ (shortfall)"); ax[1].set_ylabel("$\\sigma_u$ (excess)"); ax[1].set_title("(b) Estimated scales by sample")
plt.tight_layout()
save_figure(fig, FIG_SUPP, "Figure_S4")
print("saved: work/montecarlo.pkl, tables/montecarlo.csv, figures/supplementary/Figure_S4")
