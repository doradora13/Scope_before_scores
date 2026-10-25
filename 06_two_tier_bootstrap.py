# -*- coding: utf-8 -*-
"""
06_two_tier_bootstrap.py - NUTS3 cluster bootstrap of two-tier specification (c).

NUTS3 regions (column nuts3: real codes, or the GDP-value proxy if USE_REAL_NUTS3 = False) are resampled with
replacement to obtain the distributions of sigma_u, sigma_w and the asymmetry LR. Because sigma_v can reach the
boundary at 0 in resamples, in every replication the unrestricted model is run to full convergence from three starting
values and the restricted model from four, and the solution with the highest likelihood is kept. With a single
starting value the LR is badly wrong in some replications.
  All inventories: 30 replications (with country fixed effects); Spanish complete reporters: 40 replications.
Input : work/data.pkl
Output: work/bootstrap.pkl
Manuscript: Table S3 (bootstrap panel: the share of replications with sigma_u > sigma_w is 100% for Spanish complete
            reporters and 7% for all inventories), Supplementary Section S3
"""
import time
import warnings
from lib import *

# hide the RuntimeWarnings (overflow) raised when the optimiser passes through extreme parameter regions; results are unaffected
warnings.filterwarnings("ignore", category=RuntimeWarning)

B_MAIN, B_SPAIN, SEED = 30, 40, 7     # keep these values to reproduce the manuscript
banner("06 province-cluster bootstrap of two-tier specification (c) (Table S3 panel)")
main = load_obj("data.pkl")["main"]
rng = np.random.default_rng(SEED)


def boot(s, B, fe, robust):
    s = s.reset_index(drop=True)
    cols = BASE + [c for c in Q if s[c].nunique() > 1]
    X, _ = design(s, cols, fe=fe)
    yy = s.lnE.values
    cl = s.nuts3.values
    uniq = np.unique(cl)
    r0 = sfa_2tier_fast(yy, X)
    out = []
    t0 = time.time()
    for b in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(cl == c)[0] for c in pick])
        if robust:
            r = sfa_2tier_best(yy[idx], X[idx], start=r0["theta"])            # three starting values
            sym = sfa_2tier_sym(yy[idx], X[idx], r["theta"])                  # four starting values
        else:
            r = sfa_2tier_fast(yy[idx], X[idx], start=r0["theta"])            # one starting value, full convergence
            sym = sfa_2tier_sym(yy[idx], X[idx], r["theta"], multistart=False)
        out.append((r["su"], r["sw"], 2 * (r["llf"] - sym["llf"])))
        if (b + 1) % 10 == 0:
            print(f"    replication {b + 1}/{B} [{time.time() - t0:.0f}s]", flush=True)
    return np.array(out)


def summary(a, label):
    return dict(sample=label, B=len(a), su_lo=np.percentile(a[:, 0], 2.5), su_hi=np.percentile(a[:, 0], 97.5),
                sw_lo=np.percentile(a[:, 1], 2.5), sw_hi=np.percentile(a[:, 1], 97.5), share_su_gt_sw=float(np.mean(a[:, 0] > a[:, 1])),
                median_LR=float(np.median(a[:, 2])), share_LR_gt_crit=float(np.mean(a[:, 2] > CHI2_1_CRIT_1PCT)))


print("  all inventories (B=%d)" % B_MAIN, flush=True)
bm = boot(main, B_MAIN, "country_code", robust=True)
sq = main[(main.country_code == "es") & (main.sc >= 5) & (main.core5_complete_flag == 1)]
print("  Spain, complete reporters (B=%d)" % B_SPAIN, flush=True)
bs = boot(sq, B_SPAIN, None, robust=True)
tab = pd.DataFrame([summary(bm, "All inventories"), summary(bs, "Spain, complete reporters")])
print(tab.round(3).to_string(index=False))
save_obj("bootstrap.pkl", dict(table=tab, draws_main=bm, draws_spain=bs))
print("saved: work/bootstrap.pkl")
