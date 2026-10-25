# -*- coding: utf-8 -*-
"""
21_validation_simulation.py - does the screening protocol work on data simulated from the measurement model?
(Section 3.6; Supplementary Section S15)

Data generation (plasmode design that keeps the observed GCoM design, i.e. covariates, countries and reported-sector sets;
7,494 inventories per replication):
    complete-scope emissions  ln E*_i = mu_i + v_i + u_i
        mu_i : constant, structural variables and country effects of model (3) (Table 2), at the observed x_i and country
        v_i  ~ N(0, sigma_v^2),  u_i ~ |N(0, sigma_u^2)| (half-normal excess),  sigma_u = 0, 0.25, 0.5
        sigma_v is set so that var(v + u) equals the residual variance of the complete reporters in model (3) (0.403^2).
    reported emissions        ln E_i = ln E*_i - w_i,  w_i = -ln(sum of the shares of the sectors in the observed reported set S_i)
        The sector shares are those of one six-sector inventory (donor) drawn at random from the same country
        (from all donors for countries with fewer than 20 donors).
        Coverage is independent of u (the baseline assumption of Propositions 1-2).
Three approaches are compared (all half-normal cost frontiers with country fixed effects):
    pooled    all inventories, no coverage controls
    controls  all inventories, sector-count dummies + core-five completeness indicator (the common 'just control for it' approach)
    screened  complete reporters only (3,108 inventories; the screening implied by Proposition 2)
Recorded: OLS residual skewness, maximum likelihood estimate of sigma_u, rejection of sigma_u = 0 by the LR test at 5%
      (mixed chi-square critical value 2.706), rank correlation between the residuals and the true u (complete reporters),
      share of incomplete reporters among the apparent 'top 10% efficient' (lowest 10% of residuals), and
      tau = 0.25 quantile benchmarks: the difference between the all-inventory benchmark (no coverage controls) and the
      complete-reporter benchmark, and the precision of the top-quartile flags (share of flagged inventories in the true
      top quartile of u).
Maximum likelihood: if the third moment of the residuals is not positive (wrong skewness), the OLS solution (sigma_u = 0) is a
          stationary point of the likelihood and the maximum likelihood solution (Waldman, 1982); sigma_u = 0 and LR = 0 are
          then recorded. Otherwise the likelihood is maximised with analytic gradients (same likelihood as lib.sfa).
Input  : work/data.pkl (donors are chosen by the same rule as in stage 20)
Outputs: work/validation.pkl, tables/Validation_Simulation.xlsx (Table S19), figures/framework/Figure_M3 (Fig. S12)
"""
import time
from lib import *

R, SEED = 100, 21                  # replications per scenario and random seed (keep unchanged to reproduce the manuscript values)
SIGMA_U = [0.0, 0.25, 0.5]
MIN_DONORS = 20
FIG_FW = os.path.join(OUT_DIR, "figures", "framework")
os.makedirs(FIG_FW, exist_ok=True)

banner(f"21 validation simulation: {len(SIGMA_U)} scenarios x {R} replications on the GCoM design (n = 7,494)")
main = load_obj("data.pkl")["main"].reset_index(drop=True)
n = len(main)
cty = main.country_code.values
sc = main.sc.values
complete = ((main.sc >= 5) & (main.core5_complete_flag == 1)).values
setmask = np.array([[ch == "1" for ch in s] for s in main.sector_set.values])


# ---------------------------------------------------------------- half-normal cost frontier (analytic gradient)
def _hn_parts(th, y, X):
    k = X.shape[1]; eps = y - X @ th[:k]
    su, sv = np.exp(th[k]), np.exp(th[k + 1]); s2 = su * su + sv * sv; s = np.sqrt(s2)
    a = su / (sv * s); z = eps * a
    lPhi = log_ndtr(z); mills = np.exp(stats.norm.logpdf(z) - lPhi)
    return k, eps, su, sv, s2, s, a, lPhi, mills


def _hn_nll(th, y, X):
    k, eps, su, sv, s2, s, a, lPhi, mills = _hn_parts(th, y, X)
    return -np.sum(LN2 - np.log(s) - LNSQRT2PI - 0.5 * eps * eps / s2 + lPhi)


def _hn_grad(th, y, X):
    k, eps, su, sv, s2, s, a, lPhi, mills = _hn_parts(th, y, X)
    s3 = s2 * s
    gb = X.T @ (eps / s2 - mills * a)
    gsu = np.sum(-su / s2 + eps * eps * su / (s2 * s2) + mills * eps * sv / s3)
    gsv = np.sum(-sv / s2 + eps * eps * sv / (s2 * s2) - mills * eps * su * (s2 + sv * sv) / (sv * sv * s3))
    return -np.r_[gb, su * gsu, sv * gsv]


def frontier(y, X):
    """Return the residual skewness, sigma_u, LR(sigma_u = 0) and the residuals (for ranking)."""
    o = ols(y, X); e = o["e"]; m2 = np.mean(e ** 2); m3 = np.mean(e ** 3)
    out = dict(skew=float(stats.skew(e)), resid=e)
    if m3 <= 0:                                        # wrong skewness: the OLS solution is the maximum likelihood solution (Waldman, 1982)
        out.update(su=0.0, lr=0.0)
        return out
    su2 = (m3 / (np.sqrt(2 / np.pi) * (4 / np.pi - 1))) ** (2 / 3)
    sv2 = max(m2 - (1 - 2 / np.pi) * su2, 0.2 * m2)
    b0 = o["b"].copy(); b0[0] -= np.sqrt(2 / np.pi * su2)
    k = X.shape[1]
    r = optimize.minimize(_hn_nll, np.r_[b0, 0.5 * np.log(su2), 0.5 * np.log(sv2)], args=(y, X), jac=_hn_grad,
                          method="L-BFGS-B", bounds=[(None, None)] * k + [(-8, 4), (-8, 4)],
                          options=dict(maxiter=5000, ftol=1e-13, gtol=1e-8))
    out.update(su=float(np.exp(r.x[k])), lr=float(max(2 * (-r.fun - o["llf"]), 0.0)))
    return out


# check that the analytic-gradient optimiser matches lib.sfa (observed complete reporters in Spain, whose residual skewness is positive)
_hq = main[complete & (cty == "es")]
_X, _ = design(_hq, BASE)
_a = frontier(_hq.lnE.values, _X); _b = sfa(_hq.lnE.values, _X, "hn", compute_se=False)
print(f"check on real complete reporters: fast su={_a['su']:.4f} LR={_a['lr']:.3f} | lib.sfa su={_b['su']:.4f} LR={_b['lr_sigma_u']:.3f}")

# ---------------------------------------------------------------- components of the data-generating process
X3, n3 = design(main, BASE + Q, fe="country_code")
b3 = ols(main.lnE.values, X3)["b"]
keep = [j for j, nm in enumerate(n3) if nm == "const" or nm in BASE or nm.startswith("country_code=")]
mu = X3[:, keep] @ b3[keep]
e3 = main.lnE.values - X3 @ b3
var_target = float(np.var(e3[complete]))              # residual variance of the complete reporters
E6 = main[SECTOR_COLS].values.astype(float)
six = sc == 6
share6 = E6[six] / E6[six].sum(1, keepdims=True)
cty6 = cty[six]
donor_n = pd.Series(cty6).value_counts()
pool_idx = np.arange(six.sum())
donor_pool = {c: (np.where(cty6 == c)[0] if donor_n.get(c, 0) >= MIN_DONORS else pool_idx) for c in np.unique(cty)}
rows_by_cty = {c: np.where(cty == c)[0] for c in np.unique(cty)}

X_pool, _ = design(main, BASE, fe="country_code")
X_ctrl, _ = design(main, BASE + ["sc1", "sc2", "sc3", "sc4", "sc6", "core5_complete_flag"], fe="country_code")
hq = main[complete].reset_index(drop=True)
X_scr, _ = design(hq, BASE + ["sc6"], fe="country_code")
print(f"target var(v+u) = {var_target:.4f} (sd {np.sqrt(var_target):.3f}); donors: {six.sum()}")


def one_rep(su, rng):
    sv = np.sqrt(max(var_target - (1 - 2 / np.pi) * su ** 2, 1e-4))
    v = rng.normal(0, sv, n)
    u = np.abs(rng.normal(0, su, n)) if su > 0 else np.zeros(n)
    j = np.empty(n, int)
    for c, rows in rows_by_cty.items():
        j[rows] = rng.choice(donor_pool[c], len(rows), replace=True)
    pi = (share6[j] * setmask).sum(1)
    w = -np.log(np.clip(pi, 1e-6, None))
    y = mu + v + u - w
    out = {}
    for nm, yy, XX, rows in [("pooled", y, X_pool, np.arange(n)), ("controls", y, X_ctrl, np.arange(n)),
                             ("screened", y[complete], X_scr, np.where(complete)[0])]:
        f = frontier(yy, XX)
        e = f["resid"]
        if nm == "screened":
            rho = stats.spearmanr(e, u[complete]).correlation if su > 0 else np.nan
            inc10 = np.nan
        else:
            rho = stats.spearmanr(e[complete], u[complete]).correlation if su > 0 else np.nan
            inc10 = float((~complete)[e <= np.quantile(e, 0.10)].mean())
        out[nm] = dict(skew=f["skew"], wrong=float(f["skew"] < 0), su_hat=f["su"], reject=float(f["lr"] > KP_CRIT[1][0.05]),
                       rho=rho, inc_top10=inc10)
    # tau = 0.25 benchmarks (evaluated on the complete reporters)
    yc = y[complete]
    q_scr = X_scr @ qreg(yc, X_scr, 0.25)
    q_mix = (X_pool @ qreg(y, X_pool, 0.25))[complete]
    g_s = np.clip(1 - np.exp(q_scr - yc), 0, None); g_m = np.clip(1 - np.exp(q_mix - yc), 0, None)
    f_s = g_s > np.quantile(g_s, 0.75); f_m = g_m > np.quantile(g_m, 0.75)
    if su > 0:
        top = u[complete] > np.quantile(u[complete], 0.75)
        prec_s, prec_m = float(top[f_s].mean()), float(top[f_m].mean())
    else:
        prec_s = prec_m = np.nan
    out["bench"] = dict(shift=float(np.mean(q_mix - q_scr)), above_scr=float(np.mean(yc > q_scr)), above_mix=float(np.mean(yc > q_mix)),
                        prec_scr=prec_s, prec_mix=prec_m, overlap=float(np.mean(f_s & f_m) / np.mean(f_s)))
    out["true"] = dict(skew_w_all=float(stats.skew(-w)), mean_w_complete=float(w[complete].mean()), mean_w_incomplete=float(w[~complete].mean()))
    return out


rng = np.random.default_rng(SEED)
records = []
t0 = time.time()
for su in SIGMA_U:
    for r_ in range(R):
        o = one_rep(su, rng)
        for k_, dct in o.items():
            records.append(dict(sigma_u=su, rep=r_, part=k_, **dct))
    print(f"  sigma_u = {su:.2f}: {R} replications done [{time.time() - t0:.0f}s]", flush=True)
sim = pd.DataFrame(records)

# ---------------------------------------------------------------- summary
rows = []
for su in SIGMA_U:
    for nm in ["pooled", "controls", "screened"]:
        s_ = sim[(sim.sigma_u == su) & (sim.part == nm)]
        rows.append(dict(sigma_u=su, approach=nm, mean_skewness=s_["skew"].mean(), share_wrong_skew=s_.wrong.mean(),
                         mean_sigma_u_hat=s_.su_hat.mean(), sd_sigma_u_hat=s_.su_hat.std(), reject_sigma_u0=s_.reject.mean(),
                         rank_corr_complete=s_.rho.mean(), incomplete_share_top10=s_.inc_top10.mean()))
t_sum = pd.DataFrame(rows)
rows = []
for su in SIGMA_U:
    s_ = sim[(sim.sigma_u == su) & (sim.part == "bench")]
    rows.append(dict(sigma_u=su, benchmark_shift=s_["shift"].mean(), above_screened=s_.above_scr.mean(), above_mixed=s_.above_mix.mean(),
                     precision_screened=s_.prec_scr.mean(), precision_mixed=s_.prec_mix.mean(), flag_overlap=s_.overlap.mean()))
t_bench = pd.DataFrame(rows)
print("\nfrontier diagnostics (means over replications; reject = LR(sigma_u = 0) at 5%):")
print(t_sum.round(3).to_string(index=False))
print("\ntau = 0.25 benchmarks applied to complete reporters (precision = flagged inventories in the true top quartile of u):")
print(t_bench.round(3).to_string(index=False))
print(f"share of incomplete inventories in the sample: {(~complete).mean():.3f}")
mcse = sim[sim.part.isin(["pooled", "controls", "screened"])].groupby(["sigma_u", "part"]).reject.agg(lambda x: np.sqrt(x.mean() * (1 - x.mean()) / len(x)))
print("Monte Carlo standard error of rejection rates (max):", round(float(mcse.max()), 3))

save_obj("validation.pkl", dict(sim=sim, t_sum=t_sum, t_bench=t_bench, R=R, sigma_u=SIGMA_U, var_target=var_target))
with pd.ExcelWriter(os.path.join(TAB_DIR, "Validation_Simulation.xlsx")) as xw:
    t_sum.to_excel(xw, sheet_name="frontier", index=False)
    t_bench.to_excel(xw, sheet_name="benchmarks", index=False)

# ---------------------------------------------------------------- Figure_M3 (Fig. S12)
fig, ax = plt.subplots(1, 3, figsize=(FIG_W, 2.6))
col = {"pooled": "#7f7f7f", "controls": "#ff7f0e", "screened": "#1f77b4"}
lab = {"pooled": "all inventories", "controls": "all + coverage controls", "screened": "complete reporters only"}
xx = np.arange(len(SIGMA_U))
for j, nm in enumerate(["pooled", "controls", "screened"]):
    s_ = t_sum[t_sum.approach == nm]
    ax[0].plot(SIGMA_U, s_.mean_sigma_u_hat, "o-", color=col[nm], label=lab[nm], ms=3)
    ax[1].bar(xx + (j - 1) * 0.27, s_.reject_sigma_u0, 0.27, color=col[nm])
ax[0].plot(SIGMA_U, SIGMA_U, color="k", ls="--", lw=0.8, label="true $\\sigma_u$")
ax[0].set_xlabel("true $\\sigma_u$"); ax[0].set_ylabel("mean estimated $\\sigma_u$")
ax[0].legend(frameon=False, fontsize=6); ax[0].set_title("(a) Excess scale recovered", fontsize=8)
ax[1].axhline(0.05, color="k", ls=":", lw=0.8)
ax[1].set_xticks(xx); ax[1].set_xticklabels([f"{s:.2f}" for s in SIGMA_U]); ax[1].set_xlabel("true $\\sigma_u$")
ax[1].set_ylabel("rejection rate of $\\sigma_u=0$ (5%)"); ax[1].set_ylim(0, 1.05); ax[1].set_title("(b) Excess detected", fontsize=8)
ax[2].bar(xx - 0.2, t_bench.above_screened, 0.4, color=col["screened"], label="complete-only benchmark")
ax[2].bar(xx + 0.2, t_bench.above_mixed, 0.4, color=col["pooled"], label="all-inventory benchmark")
ax[2].axhline(0.75, color="k", ls=":", lw=0.8)
ax[2].set_xticks(xx); ax[2].set_xticklabels([f"{s:.2f}" for s in SIGMA_U]); ax[2].set_xlabel("true $\\sigma_u$")
ax[2].set_ylabel("complete reporters above benchmark"); ax[2].set_ylim(0, 1.25); ax[2].set_yticks(np.arange(0, 1.01, 0.2))
ax[2].legend(frameon=False, fontsize=6, loc="upper center"); ax[2].set_title("(c) Benchmark contamination", fontsize=8)
plt.tight_layout()
save_figure(fig, FIG_FW, "Figure_M3", tight=True)
print(f"saved: work/validation.pkl, tables/Validation_Simulation.xlsx, figures/framework/Figure_M3 [{time.time() - t0:.0f}s]")
