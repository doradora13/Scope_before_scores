# -*- coding: utf-8 -*-
"""
17_screening_contrast.py - how much the results change when comparability is not screened first (Supplementary Section S13).

(1) Over time: mean change over all 1,758 baseline-monitoring pairs vs the 1,266 constant-coverage pairs (log and annualised); share of the summed decline from coverage-change pairs
(2) Cross-section: share of inventories with three or fewer sectors among the lowest 10% of per-capita emissions ('apparently best') and the lowest 10% of conditional residuals
      - residuals from structural covariates only (Table 2, model (1); no coverage controls)
      - residuals after adding the reporting variables and country fixed effects (Table 2, model (3))
Input : work/data.pkl, work/repeated.pkl
Output: work/contrast.pkl (Table S13, written by stage 18)
"""
from lib import *
from scipy.stats import spearmanr

banner("17 what changes when comparability is not screened first")
main = load_obj("data.pkl")["main"].copy()
d = load_obj("repeated.pkl")["pairs"].copy()
d["ann"] = 100 * d.dT / d.dy
same = d.cat == "same coverage"
ts = dict(n_all=len(d), n_same=int(same.sum()), mean_all=d.dT.mean(), mean_same=d[same].dT.mean(),
          ann_all=d.ann.mean(), ann_same=d[same].ann.mean(), med_ann_all=d.ann.median(), med_ann_same=d[same].ann.median(),
          share_pairs_changed=float((~same).mean()), share_sum_from_changed=float(d.loc[~same, "dT"].sum() / d.dT.sum()))
ts["ratio_log"] = ts["mean_all"] / ts["mean_same"]; ts["ratio_ann"] = ts["ann_all"] / ts["ann_same"]
print(f"mean dlnE: all {ts['n_all']} pairs {ts['mean_all']:.3f} ({(np.exp(ts['mean_all'])-1)*100:.1f}%) vs constant coverage {ts['mean_same']:.3f} "
      f"({(np.exp(ts['mean_same'])-1)*100:.1f}%) -> x{ts['ratio_log']:.2f}")
print(f"annualised: {ts['ann_all']:.2f} vs {ts['ann_same']:.2f} %/yr -> x{ts['ratio_ann']:.2f} (medians {ts['med_ann_all']:.2f} vs {ts['med_ann_same']:.2f})")
print(f"coverage-change pairs: {ts['share_pairs_changed']*100:.1f}% of pairs, {ts['share_sum_from_changed']*100:.1f}% of the summed log decline")
cat_tab = d.groupby("cat").ann.agg(["size", "mean", "median"]).reindex(["same coverage", "addition only", "removal only", "both"]).reset_index()

le3 = (main.sc <= 3).values; le2 = (main.sc <= 2).values
low = (main.lnEpc <= main.lnEpc.quantile(0.10)).values
e1 = ols(main.lnE.values, design(main, BASE)[0])["e"]
e3 = ols(main.lnE.values, design(main, BASE + Q, fe="country_code")[0])["e"]
b1 = e1 <= np.quantile(e1, 0.10); b3 = e3 <= np.quantile(e3, 0.10)
xs = dict(share_le3_all=float(le3.mean()), share_le2_all=float(le2.mean()), share_le3_lowpc=float(le3[low].mean()), share_le2_lowpc=float(le2[low].mean()),
          share_le3_best_naive=float(le3[b1].mean()), share_le3_best_adjusted=float(le3[b3].mean()),
          overlap_best_deciles=float((b1 & b3).sum() / b1.sum()), spearman_naive_adjusted=float(spearmanr(e1, e3).correlation))
print(f"<=3 sectors: {xs['share_le3_all']*100:.1f}% of inventories | {xs['share_le3_lowpc']*100:.1f}% of the lowest-10% per-capita emitters "
      f"(1-2 sectors: {xs['share_le2_lowpc']*100:.1f}% vs {xs['share_le2_all']*100:.1f}%)")
print(f"lowest-10% residual: structural model {xs['share_le3_best_naive']*100:.1f}% <=3 sectors | with coverage controls + country FE {xs['share_le3_best_adjusted']*100:.1f}%; "
      f"overlap of the two 'best' deciles {xs['overlap_best_deciles']*100:.1f}%, Spearman {xs['spearman_naive_adjusted']:.3f}")
save_obj("contrast.pkl", dict(ts=ts, cat_tab=cat_tab, xs=xs))
print("saved: work/contrast.pkl")
