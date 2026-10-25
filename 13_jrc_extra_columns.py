# -*- coding: utf-8 -*-
"""
13_jrc_extra_columns.py - adds columns from the JRC source file: real NUTS3 codes, direct/indirect energy emissions and
activity, adhesion type.

Variables that the analysis file (GlobalCoM_SFA_ready.csv) lacks are taken from JRC GlobalCoM.xlsx (public release,
Kona et al. 2021) and attached to the same inventories:
  - real NUTS3 (2016) codes -> NUTS2, NUTS1 (first 4 and 3 characters of the code)
  - direct (fuel combustion) and indirect (grid electricity and heat) emissions of the energy sectors, and the
    activity (MWh) of each
  - energy-sector emissions = total emissions - waste (activity data are not collected for waste; JRC file, Table 1)
  - adhesion type (individual / joint SECAP option 1 or 2)
It also checks how far the GDP-value proxy identifier (543 groups) matches the real NUTS3 codes one-to-one.
Input : work/data.pkl, JRC_XLSX (top of lib.py; downloaded automatically if missing)
Output: work/extra.pkl
Manuscript: Section 4.2, Table S9, Supplementary Section S9
"""
from lib import *

banner("13 extra columns from the JRC file: real NUTS3, direct/indirect energy, adhesion type")
main = load_obj("data.pkl")["main"].copy()
ex = jrc_inventory_extra()
n0 = len(main)
drop = [c for c in ex.columns if c in main.columns and c not in ("GCoM_ID", "inventory_type")]
mx = main.drop(columns=drop).merge(ex, on=["GCoM_ID", "inventory_type"], how="left")
assert len(mx) == n0, "merge changed the number of inventories"
rel = np.abs(mx.jrc_total_emissions / mx.total_emissions_reported_tonnes - 1)
print(f"inventories matched to the JRC file: {int(mx.jrc_total_emissions.notna().sum())} / {n0}; "
      f"total emissions identical (rel. diff < 1e-6): {int((rel < 1e-6).sum())}")
mx["N3"] = mx.NUTS3
mx["N2"] = mx.NUTS3.str[:4]
mx["N1"] = mx.NUTS3.str[:3]

# ---- GDP-value proxy identifier vs real NUTS3 codes (Table S9)
proxy = mx.country_code + "_" + mx.gdp_per_capita_nuts3_eur.round(2).astype(str)
m = mx.assign(proxy=proxy).dropna(subset=["N3"])
p2n = m.groupby("proxy").N3.nunique(); n2p = m.groupby("N3").proxy.nunique()
ok = m.proxy.isin(p2n[p2n == 1].index) & m.N3.isin(n2p[n2p == 1].index)
merged = (m[m.proxy.isin(p2n[p2n > 1].index)].groupby(["country_code", "proxy", "N3"])
          .agg(inventories=("GCoM_ID", "size"), example=("signatory_name", "first")).reset_index())
conc = dict(proxies=m.proxy.nunique(), nuts3=m.N3.nunique(), proxies_one_to_one=int((p2n == 1).sum()),
            proxies_merging=int((p2n > 1).sum()), inv_one_to_one=int(ok.sum()), inv=len(m), share_one_to_one=float(ok.mean()),
            missing_nuts3=int(mx.N3.isna().sum()))
print(f"proxy groups {conc['proxies']} vs real NUTS3 codes {conc['nuts3']}: {conc['proxies_one_to_one']} proxies map 1:1, "
      f"{conc['proxies_merging']} merge two NUTS3 regions; inventories in 1:1 groups {conc['inv_one_to_one']}/{conc['inv']} "
      f"({conc['share_one_to_one']*100:.1f}%)")
print(merged.to_string(index=False))
lv = {k: int(mx[k].nunique()) for k in ["N1", "N2", "N3"]}
print("distinct regions in the analysis sample:", lv)
print("adhesion type (baseline inventories):", mx[mx.mon == 0].adhesion_type.value_counts().to_dict())
save_obj("extra.pkl", dict(main=mx, concordance=conc, merged=merged, levels=lv))
print("saved: work/extra.pkl")
