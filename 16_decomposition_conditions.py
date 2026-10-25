# -*- coding: utf-8 -*-
"""
16_decomposition_conditions.py - conditions under which the energy-use / carbon-intensity decomposition, Eq. (11), can be interpreted (Supplementary Section S12).

Eq. (11), dlnE = dlnA + dln(E/A), is an identity whenever E, A > 0; the conditions for interpreting its terms are checked in the data (1,266 constant-coverage pairs).
(1) Boundary: waste emissions have no activity data (MWh) -> exact three-factor decomposition
      dlnE = dlnA + dln(E_en/A) + dln(E/E_en)   (E_en = energy-sector emissions = total emissions - waste)
(2) LMDI decomposition of energy carbon intensity: E_en/A = s_d I_d + s_i I_i  (d: direct fuel combustion, i: grid electricity and heat)
      -> electricity-share (structure) effect + grid emission-factor effect + fuel emission-factor effect (summing exactly to dln(E_en/A))
(3) Emission-factor regime: the guidebook (Bertoldi et al. 2018, 2.4.4) asks for the same emission factors in the BEI and the MEI.
      Share of pairs with the same implied grid emission factor (indirect emissions / indirect activity) in BEI and MEI, and the relation of its change to the
      change in the JRC national electricity emission factor (NEEFE) over the same years (slope 0 = held fixed, 1 = updated to the inventory-year national factor)
(4) Estimated activity data: pairs with identical per-capita energy use or identical energy carbon intensity in both inventories -> excluded, decomposition repeated
Annualised % = 100 x log change / elapsed years. IPW: inverse probability weights 1/p from stage 08.
Input : work/extra.pkl, work/repeated.pkl, NEEFE_XLSX (set at the top of lib.py; downloaded automatically if missing)
Output: work/decomp.pkl (Table S12, written by stage 18), figures/revision/Figure_R3 (Fig. 5)
"""
from lib import *

banner("16 conditions behind the energy-use / carbon-intensity decomposition")
mx = load_obj("extra.pkl")["main"]
rp = load_obj("repeated.pkl")
cc = rp["cc"].copy().reset_index(drop=True)
B = mx[mx.mon == 0].set_index("GCoM_ID"); M = mx[mx.mon == 1].set_index("GCoM_ID")
g = lambda df, col: df.loc[cc.id, col].values.astype(float)
E0, E1 = g(B, "total_emissions_reported_tonnes"), g(M, "total_emissions_reported_tonnes")
N0, N1 = g(B, "energy_emissions_tonnes"), g(M, "energy_emissions_tonnes")
A0, A1 = g(B, "total_activity_MWh"), g(M, "total_activity_MWh")
Ed0, Ed1, Ei0, Ei1 = g(B, "emissions_dir"), g(M, "emissions_dir"), g(B, "emissions_ind"), g(M, "emissions_ind")
Ad0, Ad1, Ai0, Ai1 = g(B, "activity_data_dir"), g(M, "activity_data_dir"), g(B, "activity_data_ind"), g(M, "activity_data_ind")
P0, P1 = g(B, "population_inventory_year"), g(M, "population_inventory_year")
y0, y1 = B.loc[cc.id, "inventory_year"].values.astype(int), M.loc[cc.id, "inventory_year"].values.astype(int)
wst = g(B, "waste_emissions_tonnes")
dt = cc.dy.values.astype(float)
cc["has_waste"] = np.isfinite(wst)
cc["waste_share0"] = np.where(np.isfinite(wst), wst / E0, 0.0)
cc["TOT"] = 100 * np.log(E1 / E0) / dt
cc["A"] = 100 * np.log(A1 / A0) / dt
cc["CIen"] = 100 * np.log((N1 / A1) / (N0 / A0)) / dt
cc["NE"] = 100 * np.log((E1 / N1) / (E0 / N0)) / dt
assert np.allclose(cc.TOT, cc.A + cc.CIen + cc.NE)


def summ(df, lab):
    r = dict(sample=lab, n=len(df))
    for k in ["TOT", "A", "CIen", "NE"]:
        r[k] = df[k].mean(); r[k + "_ipw"] = np.average(df[k], weights=df.w)
    for k in ["A", "CIen", "NE"]:
        r["share_" + k] = r[k] / r["TOT"]
    return r


# ---- (2) LMDI
L = lambda a, b: np.where(np.isclose(a, b), a, (a - b) / (np.log(a) - np.log(b)))
okl = (Ed0 > 0) & (Ed1 > 0) & (Ei0 > 0) & (Ei1 > 0) & (Ad0 > 0) & (Ad1 > 0) & (Ai0 > 0) & (Ai1 > 0)
c0, c1 = N0 / A0, N1 / A1
with np.errstate(all="ignore"):
    wd = L(Ed1 / A1, Ed0 / A0) / L(c1, c0); wi = L(Ei1 / A1, Ei0 / A0) / L(c1, c0)
    cc["STR"] = 100 * (wd * np.log((Ad1 / A1) / (Ad0 / A0)) + wi * np.log((Ai1 / A1) / (Ai0 / A0))) / dt
    cc["EFi"] = 100 * wi * np.log((Ei1 / Ai1) / (Ei0 / Ai0)) / dt
    cc["EFd"] = 100 * wd * np.log((Ed1 / Ad1) / (Ed0 / Ad0)) / dt
    dln_efi = np.log((Ei1 / Ai1) / (Ei0 / Ai0)); dln_efd = np.log((Ed1 / Ad1) / (Ed0 / Ad0))
cc["lmdi_ok"] = okl
lm = cc[cc.lmdi_ok]
assert np.allclose(lm.STR + lm.EFi + lm.EFd, lm.CIen)

rows = [summ(cc, "all constant-coverage"), summ(cc[cc.has_waste], "with waste (both inventories)"), summ(cc[~cc.has_waste], "without waste")]
for c in ["it", "es", "be"]:
    rows.append(summ(cc[cc.cty == c], c))
three = pd.DataFrame(rows)
for k in ["STR", "EFi", "EFd"]:
    three[k] = [lm[k].mean(), lm[cc.has_waste[okl]][k].mean(), lm[~cc.has_waste[okl]][k].mean()] + [lm[lm.cty == c][k].mean() for c in ["it", "es", "be"]]
print(f"constant-coverage pairs {len(cc)}; waste in both inventories {int(cc.has_waste.sum())} ({cc.has_waste.mean()*100:.1f}%), "
      f"mean baseline waste share {cc.loc[cc.has_waste, 'waste_share0'].mean()*100:.1f}%; LMDI sample {int(okl.sum())}")
print(three.round(3).to_string(index=False))

# ---- (3) emission-factor regime
names = {"Austria": "at", "Belgium": "be", "Bulgaria": "bg", "Croatia": "hr", "Cyprus": "cy", "Czech Republic": "cz", "Denmark": "dk", "Estonia": "ee",
         "Finland": "fi", "France": "fr", "Germany": "de", "Greece": "el", "Hungary": "hu", "Ireland": "ie", "Italy": "it", "Latvia": "lv", "Lithuania": "lt",
         "Luxembourg": "lu", "Malta": "mt", "Netherlands": "nl", "Poland": "pl", "Portugal": "pt", "Romania": "ro", "Slovak Republic": "sk", "Slovenia": "si",
         "Spain": "es", "Sweden": "se", "United Kingdom": "uk"}
ensure_file(NEEFE_XLSX, NEEFE_XLSX_URL)


def neefe(sheet):
    t = pd.read_excel(NEEFE_XLSX, sheet_name=sheet, header=None)
    years = [int(float(v)) for v in t.iloc[1, 1:32].values]
    return {names[str(t.iloc[r, 0]).strip()]: dict(zip(years, pd.to_numeric(t.iloc[r, 1:32], errors="coerce").values))
            for r in range(2, t.shape[0]) if str(t.iloc[r, 0]).strip() in names}


NE = {"co2": neefe("Table 1_EU_IPCC_CO2"), "ghg": neefe("Table 2_EU_IPCC_GHG"), "lca": neefe("Table 3_EU_LCA_GHG")}
lca, co2eq = g(B, "lca"), g(B, "co2eq")
n0 = np.full(len(cc), np.nan); n1 = np.full(len(cc), np.nan)
for k in range(len(cc)):
    tab = NE["lca"] if lca[k] == 1 else (NE["ghg"] if co2eq[k] == 1 else NE["co2"])
    c = cc.cty.values[k]
    if c in tab:
        n0[k], n1[k] = tab[c].get(y0[k], np.nan), tab[c].get(y1[k], np.nan)
dln_n = np.log(n1 / n0)
okn = okl & np.isfinite(dln_n)
o = ols(dln_efi[okn], np.column_stack([np.ones(okn.sum()), dln_n[okn]]), cluster=cc.nuts3.values[okn])
closer_bei = np.abs(np.log((Ei1 / Ai1)[okn] / n0[okn])) < np.abs(np.log((Ei1 / Ai1)[okn] / n1[okn]))
ef = dict(n=int(okn.sum()), share_grid_ef_identical=float(np.mean(np.abs(dln_efi[okl]) < 0.001)), share_grid_ef_within2=float(np.mean(np.abs(dln_efi[okl]) < 0.02)),
          share_fuel_ef_identical=float(np.mean(np.abs(dln_efd[okl]) < 0.001)), slope=o["b"][1], slope_t=o["t"][1], const=o["b"][0],
          mean_dln_neefe=float(np.mean(dln_n[okn])), mean_dln_grid_ef=float(np.mean(dln_efi[okn])), share_closer_to_bei_factor=float(np.mean(closer_bei)))
by = []
for c in ["it", "es", "be"]:
    m_ = okn & (cc.cty.values == c)
    oo = ols(dln_efi[m_], np.column_stack([np.ones(m_.sum()), dln_n[m_]]))
    by.append(dict(country=c, n=int(m_.sum()), grid_ef_identical=float(np.mean(np.abs(dln_efi[okl & (cc.cty.values == c)]) < 0.001)),
                   slope=oo["b"][1], mean_dln_neefe=float(np.mean(dln_n[m_])), mean_dln_grid_ef=float(np.mean(dln_efi[m_])),
                   closer_to_bei_factor=float(np.mean(np.abs(np.log((Ei1 / Ai1)[m_] / n0[m_])) < np.abs(np.log((Ei1 / Ai1)[m_] / n1[m_]))))))
ef_cty = pd.DataFrame(by)
print(f"\nimplied grid emission factor identical in BEI and MEI: {ef['share_grid_ef_identical']*100:.1f}% of pairs (within 2%: {ef['share_grid_ef_within2']*100:.1f}%); "
      f"fuel factor identical: {ef['share_fuel_ef_identical']*100:.1f}%")
print(f"dln(implied grid EF) on dln(national NEEFE over the same years): slope {ef['slope']:.3f} (t {ef['slope_t']:.1f}); "
      f"mean dln NEEFE {ef['mean_dln_neefe']:.3f} vs mean dln implied {ef['mean_dln_grid_ef']:.3f}; "
      f"MEI factor closer to the BEI-year national value: {ef['share_closer_to_bei_factor']*100:.1f}%")
print(ef_cty.round(3).to_string(index=False))

# ---- (4) estimated activity data
pc_fixed = np.abs(np.log(A1 / A0) - np.log(P1 / P0)) < 0.001
ci_fixed = np.abs(np.log((N1 / A1) / (N0 / A0))) < 0.001
est = dict(share_pc_energy_identical=float(pc_fixed.mean()), share_intensity_identical=float(ci_fixed.mean()),
           corr_dlnA_dlnPop=float(np.corrcoef(np.log(A1 / A0), np.log(P1 / P0))[0, 1]))
clean = summ(cc[~pc_fixed & ~ci_fixed], "excluding identical per-capita energy or intensity")
three = pd.concat([three, pd.DataFrame([clean])], ignore_index=True)
print(f"\nper-capita energy identical in BEI and MEI: {est['share_pc_energy_identical']*100:.1f}% | energy carbon intensity identical: "
      f"{est['share_intensity_identical']*100:.1f}% | corr(dlnA, dlnPop) {est['corr_dlnA_dlnPop']:.3f}")
print("excluding them:", {k: (round(v, 3) if isinstance(v, (float, np.floating)) else v) for k, v in clean.items()})

# ---- (5) influence of regional clusters: recompute without one NUTS2 region at a time / exclude energy-use changes above 10% per year (suspected implausible data revisions)
cc["N2"] = B.loc[cc.id, "N2"].values
cc["N3"] = B.loc[cc.id, "N3"].values
loo = []
for r_ in cc.N2.dropna().unique():
    s_ = cc[cc.N2 != r_]
    loo.append(dict(N2=r_, pairs_dropped=int((cc.N2 == r_).sum()), TOT=s_.TOT.mean(), A=s_.A.mean(), CIen=s_.CIen.mean(),
                    share_A=s_.A.mean() / s_.TOT.mean(), share_CIen=s_.CIen.mean() / s_.TOT.mean()))
loo = pd.DataFrame(loo).sort_values("share_A").reset_index(drop=True)
top = loo.iloc[0]
reg = cc[cc.N2 == top.N2]
big = cc.A.abs() > 10
sec_chg = {}
for s in ["municipal", "tertiary", "residential", "manufacturing", "transport"]:
    a0 = B.loc[reg.id, f"{s}_activity_MWh"].values.astype(float); a1 = M.loc[reg.id, f"{s}_activity_MWh"].values.astype(float)
    okk = (a0 > 0) & (a1 > 0)
    sec_chg[s] = float(np.median(np.log(a1[okk] / a0[okk]))) if okk.any() else np.nan
infl = dict(region=top.N2, pairs=int(len(reg)), share_of_pairs=float(len(reg) / len(cc)), region_TOT=float(reg.TOT.mean()), region_A=float(reg.A.mean()),
            contrib_TOT=float(reg.TOT.sum() / len(cc) / cc.TOT.mean()), contrib_A=float(reg.A.sum() / len(cc) / cc.A.mean()),
            share_A_without=float(top.share_A), share_CIen_without=float(top.share_CIen), share_A_loo_max=float(loo.share_A.max()),
            pairs_abs_A_gt10=int(big.sum()), pairs_abs_A_gt10_in_region=int((big & (cc.N2 == top.N2)).sum()),
            region_median_dln_activity=sec_chg, region_bei_year=float(B.loc[reg.id, "inventory_year"].median()),
            region_mei_year=float(M.loc[reg.id, "inventory_year"].median()))
three = pd.concat([three, pd.DataFrame([summ(cc[cc.N2 != top.N2], f"excluding {top.N2}"), summ(cc[~big], "excluding |energy-use change| > 10%/yr")])],
                  ignore_index=True)
print(f"\nmost influential NUTS2 region: {top.N2} ({infl['pairs']} pairs, {infl['share_of_pairs']*100:.1f}%): mean {infl['region_TOT']:.2f}%/yr, "
      f"energy use {infl['region_A']:.2f}%/yr; it contributes {infl['contrib_TOT']*100:.0f}% of the pooled mean change and {infl['contrib_A']*100:.0f}% of the pooled energy-use term")
print(f"   without it: energy-use share {infl['share_A_without']*100:.0f}%, carbon-intensity share {infl['share_CIen_without']*100:.0f}% "
      f"(leave-one-NUTS2-out range of the energy-use share {loo.share_A.min()*100:.0f}-{loo.share_A.max()*100:.0f}%)")
print(f"   median dln sector activity in {top.N2} (BEI {infl['region_bei_year']:.0f} -> MEI {infl['region_mei_year']:.0f}):",
      {k: round(v, 2) for k, v in sec_chg.items()})
print(f"   pairs with |energy-use change| > 10%/yr: {infl['pairs_abs_A_gt10']} ({infl['pairs_abs_A_gt10_in_region']} in {top.N2})")
print(three.tail(2).round(3).to_string(index=False))
save_obj("decomp.pkl", dict(three=three, ef=ef, ef_cty=ef_cty, est=est, cc=cc, loo=loo, infl=infl))

# ---- Figure_R3 (Fig. 5)
FIG_REV = os.path.join(OUT_DIR, "figures", "revision"); os.makedirs(FIG_REV, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(FIG_W, 2.9))
t_ = three.set_index("sample").loc[["all constant-coverage", "it", "es", "be"]]
# colour-vision-safe palette (Okabe and Ito): categories stay distinguishable under protan, deutan and tritan simulation
comps = [("A", "energy use", "#0072B2"), ("STR", "electricity share", "#56B4E9"), ("EFi", "grid factor", "#D55E00"),
         ("EFd", "fuel factor", "#F0E442"), ("NE", "non-energy (waste)", "#999999")]
xs = np.arange(len(t_))
for s_, sign in [(1, ">"), (-1, "<")]:
    base = np.zeros(len(t_))
    for k, lab, col in comps:
        v = t_[k].values
        v = np.where(v * s_ > 0, v, 0)
        ax[0].bar(xs, v, bottom=base, color=col, label=lab if s_ == 1 else None, width=0.6)
        base += v
ax[0].scatter(xs, t_.TOT, color="k", zorder=3, s=12, label="total")
neg = np.sum([np.minimum(t_[k].values, 0) for k, _, _ in comps], axis=0).min()
ax[0].set_ylim(neg - 0.3, 2.3)
ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xticks(xs); ax[0].set_xticklabels(["pooled", "it", "es", "be"])
ax[0].set_ylabel("% per year"); ax[0].set_title("(a) Exact decomposition of constant-coverage change")
ax[0].legend(frameon=False, fontsize=5.5, ncol=3, loc="upper center")
okp = okn
ax[1].scatter(dln_n[okp], dln_efi[okp], s=3, alpha=0.4, color="#0072B2", rasterized=True)
xx = np.linspace(np.nanmin(dln_n[okp]), np.nanmax(dln_n[okp]), 10)
ax[1].plot(xx, xx, "k--", lw=0.8, label="updated to inventory-year national factor (slope 1)")
ax[1].plot(xx, ef["const"] + ef["slope"] * xx, "-", color="#D55E00", lw=1, label=f"fitted slope {ef['slope']:.2f}")
ax[1].axhline(0, color="grey", lw=0.6, label="factor held fixed (0)")
ax[1].set_ylim(-1.2, 0.8); ax[1].set_xlabel("dln national grid factor (BEI to MEI year)"); ax[1].set_ylabel("dln implied grid factor")
ax[1].set_title("(b) Grid emission factor: fixed or updated?"); ax[1].legend(frameon=False, fontsize=5.5)
plt.tight_layout()
save_figure(fig, FIG_REV, "Figure_R3")
print("saved: work/decomp.pkl, figures/revision/Figure_R3")
