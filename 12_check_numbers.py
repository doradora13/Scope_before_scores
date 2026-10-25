# -*- coding: utf-8 -*-
"""
12_check_numbers.py - compares 131 numbers printed in the manuscript with the results of this run.

The expected values are the numbers printed in the submitted manuscript and its supplementary material, computed with
the real NUTS3 codes (USE_REAL_NUTS3 = True in lib.py). Each computed value is rounded to the number of decimals
printed and compared with the expected value:
  OK    : equal to the expected value
  ~     : differs by one unit in the last printed digit (rounding boundary; small differences across computing
          environments)
  DIFF  : differs by more -> check whether the input data or the code have changed
  ENV   : not OK, but the item is known to vary slightly across computing environments (ENV_SENSITIVE below);
          the expected value is kept as printed
Input : work/*.pkl (stages 01-10)
Output: tables/check_manuscript_numbers.csv
"""
from lib import *

banner("12 check: numbers printed in the manuscript vs this run")
data = load_obj("data.pkl"); main = data["main"]
md = load_obj("model.pkl"); fr = load_obj("frontier.pkl"); tt = load_obj("twotier.pkl"); asym = load_obj("asym.pkl").set_index("sample")
bt = load_obj("bootstrap.pkl")["table"].set_index("sample"); mc = load_obj("montecarlo.pkl").set_index("scenario")
rp = load_obj("repeated.pkl"); spt = load_obj("spatial.pkl"); bm = load_obj("benchmarks.pkl")

s3 = md["specs"][2]
coef3 = lambda v: s3["b"][s3["names"].index(v)]
t_3 = lambda v: s3["t"][s3["names"].index(v)]
subs = fr["subs"].set_index("sample"); dist = fr["dist"]; trim = fr["trim"]
o = tt["out"]; comp = tt["comp"]
pA = rp["panelA"].set_index("category"); pB = rp["panelB"].set_index("var"); pC = rp["panelC"].set_index("var"); pD = rp["panelD"].set_index("sector")
t3 = rp["t3"].set_index("key"); t3c = rp["t3_cty"].set_index("country"); lg = rp["logit"].set_index("var"); s6 = rp["s6_counts"].set_index("sector")
mo = spt["moran"].set_index("k"); wi = spt["within"].set_index("country"); d = rp["pairs"]

# (item, location in the manuscript, computed value, printed value, decimals, percent flag: computed value x 100)
C = [
    ("analysis inventories", "Table S1", len(main), 7494, 0, False),
    ("municipalities", "Table S1", main.GCoM_ID.nunique(), 5735, 0, False),
    ("countries", "Table S1", main.country_code.nunique(), 28, 0, False),
    ("NUTS3 regions", "Table S1", main.nuts3.nunique(), 546, 0, False),
    ("complete reporters", "Table S1", int(((main.sc >= 5) & (main.core5_complete_flag == 1)).sum()), 3108, 0, False),
    ("spatial sample", "Table S1", spt["n"], 5582, 0, False),
    ("baseline-monitoring pairs", "Table S1 / Table 3", len(d), 1758, 0, False),
    ("constant-coverage pairs", "Table S1 / Table 4", int((d.cat == "same coverage").sum()), 1266, 0, False),
    ("R2 model (1)", "Table 2", md["specs"][0]["r2"], 0.912, 3, False),
    ("R2 model (3)", "Table 2", s3["r2"], 0.939, 3, False),
    ("1 sector coef, model (3)", "Table 2", coef3("sc1"), -2.458, 3, False),
    ("1 sector t, model (3)", "Table 2", t_3("sc1"), -6.4, 1, False),
    ("2 sectors coef, model (3)", "Table 2", coef3("sc2"), -0.570, 3, False),
    ("3 sectors coef, model (3)", "Table 2", coef3("sc3"), -0.420, 3, False),
    ("4 sectors coef, model (3)", "Table 2", coef3("sc4"), -0.274, 3, False),
    ("6 sectors coef, model (3)", "Table 2", coef3("sc6"), 0.266, 3, False),
    ("implied 1 sector vs five", "Table 2 notes / Section 6.1", md["implied"]["sc1"][0], -91.4, 1, True),
    ("implied 2 sectors vs five", "Table 2 notes / Section 6.1", md["implied"]["sc2"][0], -43.5, 1, True),
    ("implied 3 sectors vs five", "Table 2 notes", md["implied"]["sc3"][0], -34.3, 1, True),
    ("implied 4 sectors vs five", "Table 2 notes", md["implied"]["sc4"][0], -24.0, 1, True),
    ("residual skewness (1)", "Table 2", md["specs"][0]["skew"], -2.94, 2, False),
    ("residual skewness (2)", "Table 2", md["specs"][1]["skew"], -1.79, 2, False),
    ("residual skewness (3)", "Table 2", s3["skew"], -1.96, 2, False),
    ("residual skewness (4)", "Table 2", md["specs"][3]["skew"], -2.08, 2, False),
    ("Coelli M3T (3)", "Table 2", s3["m3t"], -69.1, 1, False),
    ("cost frontier sigma_u (half-normal, structural)", "Table S2 A", dist.sigma_u.iloc[0], 0.002, 3, False),
    ("cost frontier sigma_u (truncated-normal)", "Table S2 A", dist.sigma_u.iloc[1], 0.074, 3, False),
    ("production-type sigma_u", "Table S2 A", dist.sigma_u.iloc[4], 0.521, 3, False),
    ("production-type LR", "Table S2 A", dist.LR_sigma_u0.iloc[4], 502.16, 2, False),
    ("complete reporters skewness", "Table S2 B", subs.loc["Complete reporters (>=5 sectors, core five)", "skewness"], -0.14, 2, False),
    ("complete reporters M3T", "Table S2 B", subs.loc["Complete reporters (>=5 sectors, core five)", "M3T"], -3.2, 1, False),
    ("complete reporters production sigma_u", "Table S2 B", subs.loc["Complete reporters (>=5 sectors, core five)", "production_sigma_u"], 0.304, 3, False),
    ("complete reporters production LR", "Table S2 B", subs.loc["Complete reporters (>=5 sectors, core five)", "production_LR"], 11.05, 2, False),
    ("Spain skewness", "Table S2 B", subs.loc["Spain", "skewness"], 0.16, 2, False),
    ("Belgium skewness", "Table S2 B", subs.loc["Belgium", "skewness"], 0.36, 2, False),
    ("1% lower-tail trim: skewness", "Table S2 C", trim.skewness.iloc[0], 0.26, 2, False),
    ("1% lower-tail trim: sigma_u", "Table S2 C", trim.sigma_u.iloc[0], 0.378, 3, False),
    ("1% lower-tail trim: LR", "Table S2 C", trim.LR.iloc[0], 86.48, 2, False),
    ("two-tier (a) sigma_u", "Table S3", o["a"]["su"], 0.258, 3, False),
    ("two-tier (a) sigma_w", "Table S3", o["a"]["sw"], 0.359, 3, False),
    ("two-tier (a) LR vs shortfall-only", "Table S3", tt["lr"]["a"], 206.1, 1, False),
    ("two-tier (c) sigma_u", "Table S3", o["c"]["su"], 0.269, 3, False),
    ("two-tier (c) sigma_w", "Table S3", o["c"]["sw"], 0.295, 3, False),
    ("two-tier (c) LR vs shortfall-only", "Table S3", tt["lr"]["c"], 503.5, 1, False),
    ("two-tier (d) sigma_w", "Table S3", o["d"]["sw"], 0.205, 3, False),
    ("two-tier (d) delta, 3 or fewer sectors", "Table S3", o["d"]["dw"][0], 1.145, 3, False),
    ("two-tier (d) t, 3 or fewer sectors", "Table S3", o["d"]["dw"][0] / o["d"]["dw_se"][0], 2.76, 2, False),
    ("two-tier (d) mean E[w|e], 1 sector", "Table S3", comp.loc[1, "w"], 1.418, 3, False),
    ("asymmetry LR, MAIN (c)", "Table S3", asym.loc["MAIN (c) base+Q+country FE", "LR_asym"], 8.3, 1, False),
    ("asymmetry LR, complete reporters", "Table S3", asym.loc["HighQ (sc>=5 & core5)", "LR_asym"], 0.3, 1, False),
    ("asymmetry LR, 3 or fewer sectors", "Table S3", asym.loc["sc<=3", "LR_asym"], 74.9, 1, False),
    ("asymmetry LR, CO2 units", "Table S3", asym.loc["CO2", "LR_asym"], 2.5, 1, False),
    ("asymmetry LR, cities", "Table S3", asym.loc["cities (degurb=1)", "LR_asym"], 21.4, 1, False),
    ("asymmetry LR, Belgium", "Table S3", asym.loc["Belgium", "LR_asym"], 8.3, 1, False),
    ("asymmetry LR, Spain complete reporters", "Table S3", asym.loc["Spain HighQ", "LR_asym"], 24.1, 1, False),
    ("bootstrap all: sigma_u 2.5%", "Table S3", bt.loc["All inventories", "su_lo"], 0.250, 3, False),
    ("bootstrap all: sigma_u 97.5%", "Table S3", bt.loc["All inventories", "su_hi"], 0.289, 3, False),
    ("bootstrap all: share sigma_u > sigma_w", "Table S3", bt.loc["All inventories", "share_su_gt_sw"], 7, 0, True),
    ("bootstrap all: sigma_w 2.5%", "Table S3", bt.loc["All inventories", "sw_lo"], 0.269, 3, False),
    ("bootstrap all: median LR", "Table S3", bt.loc["All inventories", "median_LR"], 7.7, 1, False),
    ("bootstrap all: share LR > 6.63", "Table S3", bt.loc["All inventories", "share_LR_gt_crit"], 60, 0, True),
    ("bootstrap Spain: share sigma_u > sigma_w", "Table S3", bt.loc["Spain, complete reporters", "share_su_gt_sw"], 100, 0, True),
    ("bootstrap Spain: sigma_w 97.5%", "Table S3", bt.loc["Spain, complete reporters", "sw_hi"], 0.159, 3, False),
    ("bootstrap Spain: sigma_u 97.5%", "Table S3", bt.loc["Spain, complete reporters", "su_hi"], 0.283, 3, False),
    ("bootstrap Spain: median LR", "Table S3", bt.loc["Spain, complete reporters", "median_LR"], 23.6, 1, False),
    ("bootstrap Spain: share LR > 6.63", "Table S3", bt.loc["Spain, complete reporters", "share_LR_gt_crit"], 82, 0, True),
    ("MC normal: rejection LR(sigma_u=0)", "Table S4", mc.loc["normal", "rej_LRsu_1pct"], 0, 0, True),
    ("MC normal: mean sigma_v", "Table S4", mc.loc["normal", "sv"], 0.433, 3, False),
    ("MC normal: mean sigma_w", "Table S4", mc.loc["normal", "sw"], 0.068, 3, False),
    ("MC laplace: mean sigma_v", "Table S4", mc.loc["laplace", "sv"], 0.012, 3, False),
    ("MC t3: mean sigma_v", "Table S4", mc.loc["t3", "sv"], 0.079, 3, False),
    ("MC hetero shortfall: mean sigma_v", "Table S4", mc.loc["wonly_hetero_laplace", "sv"], 0.111, 3, False),
    ("MC laplace: rejection LR(sigma_u=0)", "Table S4", mc.loc["laplace", "rej_LRsu_1pct"], 100, 0, True),
    ("MC laplace: rejection asymmetry test", "Table S4", mc.loc["laplace", "rej_asym_1pct"], 2, 0, True),
    ("MC t3: rejection asymmetry test", "Table S4", mc.loc["t3", "rej_asym_1pct"], 6, 0, True),
    ("MC true two-tier: rejection asymmetry test", "Table S4", mc.loc["true2tier", "rej_asym_1pct"], 62, 0, True),
    ("MC shortfall-only (normal): rejection asymmetry test", "Table S4", mc.loc["wonly_normal", "rej_asym_1pct"], 100, 0, True),
    ("MC hetero shortfall: mean delta (true 0.8)", "Table S4", mc.loc["wonly_hetero_laplace", "dw_true"], 0.62, 2, False),
    ("same coverage: mean change", "Table 3 notes / Section 6.4", pA.loc["same coverage", "mean_all"], -0.146, 3, False),
    ("removal only: mean change", "Section 6.4", pA.loc["removal only", "mean_all"], -0.911, 3, False),
    ("removal only: n", "Section 6.4", pA.loc["removal only", "n"], 382, 0, False),
    ("sectors removed (count): coef", "Table 3 A", pB.loc["n_rem", "coef"], -0.562, 3, False),
    ("sectors removed (count): t", "Table 3 A", pB.loc["n_rem", "t"], -7.41, 2, False),
    ("sectors in both: removal only", "Table 3 B", pC.loc["removal only", "coef"], -0.271, 3, False),
    ("sectors in both: addition only", "Table 3 B", pC.loc["addition only", "coef"], -0.141, 3, False),
    ("sectors in both: both", "Table 3 B", pC.loc["both", "coef"], -0.438, 3, False),
    ("single removals: observed", "Table 3 C / Section 6.4", pD.loc["All single removals", "observed"], -0.784, 3, False),
    ("single removals: implied", "Table 3 C / Section 6.4", pD.loc["All single removals", "mechanical"], -0.302, 3, False),
    ("single removals: ratio", "Table 3 C", pD.loc["All single removals", "ratio"], 2.59, 2, False),
    ("waste removals (single)", "Table 3 C", pD.loc["waste", "n"], 160, 0, False),
    ("constant coverage: total mean", "Table 4 A / abstract", t3.loc["TOT", "mean"], -2.24, 2, False),
    ("constant coverage: total 95% CI low", "Table 4 A", t3.loc["TOT", "ci_lo"], -2.49, 2, False),
    ("constant coverage: total median", "Table 4 A", t3.loc["TOT", "median"], -1.54, 2, False),
    ("constant coverage: share decreasing", "Table 4 A", t3.loc["TOT", "share_neg"], 79.7, 1, True),
    ("constant coverage: total IPW", "Table 4 A / Section 6.5", t3.loc["TOT", "ipw_mean"], -1.84, 2, False),
    ("constant coverage: per capita mean", "Table 4 A", t3.loc["PC", "mean"], -2.82, 2, False),
    ("constant coverage: per capita median", "Table 4 A", t3.loc["PC", "median"], -2.25, 2, False),
    ("constant coverage: energy use mean", "Table 4 A", t3.loc["A", "mean"], -1.55, 2, False),
    ("constant coverage: energy share of total", "Table 4 A", t3.loc["A", "share_of_total"], 69.1, 1, True),
    ("constant coverage: carbon intensity mean", "Table 4 A", t3.loc["I", "mean"], -0.69, 2, False),
    ("constant coverage: carbon intensity share", "Table 4 A", t3.loc["I", "share_of_total"], 30.9, 1, True),
    ("constant coverage: Spain median", "Table 4 B", t3c.loc["es", "total"], -2.53, 2, False),
    ("constant coverage: Italy median", "Table 4 B", t3c.loc["it", "total"], -1.07, 2, False),
    ("selection logit t: ln population", "Table S5", lg.loc["lnPop", "t"], 2.65, 2, False),
    ("selection logit t: ln density", "Table S5", lg.loc["lnDens", "t"], 3.24, 2, False),
    ("selection logit coef: core five complete", "Table S5", lg.loc["core5_complete_flag", "coef"], 0.724, 3, False),
    ("selection logit coef: CO2-eq units", "Table S5", lg.loc["co2eq", "coef"], 0.728, 3, False),
    ("selection logit coef: baseline residual", "Table S5", lg.loc["res", "coef"], 0.374, 3, False),
    ("selection logit SE: baseline residual", "Table S5", lg.loc["res", "se"], 0.204, 3, False),
    ("selection logit t: HDD", "Table S5", lg.loc["HDDk", "t"], 2.87, 2, False),
    ("selection logit t: baseline year", "Table S5", lg.loc["year_c", "t"], -6.01, 2, False),
    ("selection logit t: sector count", "Table S5", lg.loc["sc", "t"], 0.05, 2, False),
    ("selection logit t: baseline residual", "Table S5", lg.loc["res", "t"], 1.84, 2, False),
    ("share submitting monitoring", "Table S5", rp["share_mon"], 30.7, 1, True),
    ("waste removed (all pairs)", "Table S6", s6.loc["waste", "removed"], 185, 0, False),
    ("Moran I, country FE, k=5", "Table S7", mo.loc[5, "I_country"], 0.186, 3, False),
    ("Moran I, country FE, k=10", "Table S7", mo.loc[10, "I_country"], 0.173, 3, False),
    ("Moran I, country FE, k=20", "Table S7", mo.loc[20, "I_country"], 0.151, 3, False),
    ("Moran I, NUTS3 FE, k=5", "Table S7", mo.loc[5, "I_province"], -0.004, 3, False),
    ("Moran I, NUTS3 FE, k=10", "Table S7", mo.loc[10, "I_province"], 0.003, 3, False),
    ("Moran I, NUTS3 FE, k=20", "Table S7", mo.loc[20, "I_province"], 0.008, 3, False),
    ("Moran permutation p, NUTS3 FE, k=20", "Table S7", mo.loc[20, "p_province"], 0.035, 3, False),
    ("Spain within-country I after NUTS3 FE", "Table S7", wi.loc["es", "I_province"], 0.080, 3, False),
    ("Spain within-country z after NUTS3 FE", "Table S7", wi.loc["es", "z_province"], 7.5, 1, False),
    ("benchmark gap, tau = 0.10", "Table S8", bm["bench"][0.10]["agg"], 34.2, 1, True),
    ("benchmark gap, tau = 0.25", "Table S8", bm["bench"][0.25]["agg"], 21.6, 1, True),
    ("benchmark gap, tau = 0.50", "Table S8", bm["bench"][0.50]["agg"], 12.7, 1, True),
    ("consistently flagged (count)", "Table S8", bm["stable"], 518, 0, False),
    ("consistently flagged (share)", "Table S8", bm["stable"] / bm["n_hq"], 16.7, 1, True),
    ("unstable classification (share)", "Table S8", bm["unstable_share"], 17.7, 1, True),
    ("realised-change reference: 25th percentile", "Table S8", bm["ref"]["top_quartile"], -3.18, 2, False),
]

# items known to vary slightly across computing environments (floating-point handling): the printed value is kept and the status is ENV
ENV_SENSITIVE = {
    "cost frontier sigma_u (truncated-normal)": "likelihood flat in sigma_u (LR about 0): 0.063 or 0.074 across computing environments",
    "MC true two-tier: rejection asymmetry test": "62% or 64% across computing environments (31 or 32 of 50 replications)",
}
rows_ = []
for label, where, val, exp_, dp, pct in C:
    v = float(val) * (100 if pct else 1)
    shown = round(v, dp)
    unit = 10 ** (-dp)
    gap = abs(shown - exp_)
    status = "OK" if gap < 0.5 * unit else ("~" if gap < 1.5 * unit else "DIFF")
    if status != "OK" and label in ENV_SENSITIVE:
        status = "ENV"
    rows_.append(dict(item=label, where=where, expected=exp_, this_run=shown, unit="%" if pct else "", dp=dp, status=status,
                      note=ENV_SENSITIVE.get(label, "")))
res = pd.DataFrame(rows_)
res.drop(columns="dp").to_csv(os.path.join(TAB_DIR, "check_manuscript_numbers.csv"), index=False, encoding="utf-8-sig")
w = max(len(x) for x in res.item)
for r in res.itertuples():
    note = f"   <- {r.note}" if r.status == "ENV" else ""
    print(f"  [{r.status:4s}] {r.item:{w}s}  expected {r.expected:>9.{r.dp}f}{r.unit:1s}  this run {r.this_run:>9.{r.dp}f}{r.unit:1s}   ({r.where}){note}")
n_ok, n_near, n_env, n_diff = [(res.status == k).sum() for k in ("OK", "~", "ENV", "DIFF")]
print(f"\n{len(res)} numbers checked: {n_ok} OK, {n_near} within one unit of the last printed digit, "
      f"{n_env} known to vary across computing environments, {n_diff} DIFF")
print("saved: tables/check_manuscript_numbers.csv")
