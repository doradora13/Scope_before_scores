# -*- coding: utf-8 -*-
"""
19_export_derived_data.py - exports the derived data listed in the manuscript's Data availability statement as CSV files (for the replication repository).

(1) coverage_indicators.csv      : per inventory of the analysis sample (7,494): sector reporting indicators, number of reported sectors, completeness
                                   of the core five sectors, unit and emission-factor type, data-quality flags
(2) inventory_pairs.csv          : per baseline-monitoring pair (1,758 municipalities): coverage-change category, removed and added sectors, log changes,
                                   and for constant-coverage pairs (1,266) the annualised changes and inverse probability weights
(3) nuts3_assignment.csv         : per municipality (5,735): NUTS 2016 codes (levels 3, 2 and 1) and the GDP-value proxy identifier (compared in Table S9)
(4) derived_data_dictionary.csv  : variable descriptions for the three files above
Input : work/data.pkl, work/repeated.pkl, work/extra.pkl (stages 01, 08, 13)
Output: OUT_DIR/derived_data/*.csv
Manuscript: Data availability, Tables 3 and 4, Tables S1 and S9
"""
from lib import *

banner("19 derived data for the replication repository (coverage indicators, pairs, NUTS3 codes)")
DER_DIR = os.path.join(OUT_DIR, "derived_data")
os.makedirs(DER_DIR, exist_ok=True)
main = load_obj("data.pkl")["main"]
rep = load_obj("repeated.pkl")
extra = load_obj("extra.pkl")["main"]

# ---- (1) sector coverage by inventory
cov = main[["GCoM_ID", "emission_inventory_id", "inventory_type", "inventory_year", "country_code"]].copy()
for k, s in enumerate(SECTOR_NAMES):
    cov[f"reports_{s}"] = main.sector_set.str[k].astype(int)
cov["n_reported_sectors"] = main.sc.astype(int)
cov["core_five_complete"] = main.core5_complete_flag.astype(int)
cov["complete_reporter"] = ((main.sc >= 5) & (main.core5_complete_flag == 1)).astype(int)
cov["co2eq_units"] = main.co2eq.astype(int)
cov["lca_factors"] = main.lca.astype(int)
cov["population_mismatch_flag"] = main.pop_mismatch.astype(int)
cov["integer_coordinates_flag"] = main.int_coords.astype(int)
n_bad = int((cov[[f"reports_{s}" for s in SECTOR_NAMES]].sum(axis=1) != cov.n_reported_sectors).sum())
print(f"(1) coverage indicators: {len(cov)} inventories; sector flags disagreeing with the reported sector count: {n_bad}")

# ---- (2) baseline-monitoring pairs
d = rep["pairs"].copy()
pm = main[main.GCoM_ID.isin(d.id)]
bid = pm[pm.mon == 0].set_index("GCoM_ID").emission_inventory_id
mid = pm[pm.mon == 1].set_index("GCoM_ID").emission_inventory_id
pairs = pd.DataFrame({
    "GCoM_ID": d.id.values, "country_code": d.cty.values,
    "baseline_inventory_id": bid.reindex(d.id).values, "monitoring_inventory_id": mid.reindex(d.id).values,
    "years_between": d.dy.values, "coverage_change": d.cat.values, "constant_coverage": (d.cat == "same coverage").astype(int).values,
    "n_sectors_removed": d.n_rem.values, "n_sectors_added": d.n_add.values,
    "sectors_removed": d.rem.values, "sectors_added": d["add"].values,
    "emission_factor_type_change": d.ef_change.values, "unit_change": d.unit_change.values,
    "dln_total_emissions": d.dT.values, "dln_emissions_common_sectors": d.dR.values,
    "dln_population": d.dP.values, "dln_energy_use": d.dA.values})
cc = rep["cc"].set_index("id")[["TOT", "PC", "A", "I", "w"]].rename(columns={
    "TOT": "annual_pct_total", "PC": "annual_pct_per_capita", "A": "annual_pct_energy_use",
    "I": "annual_pct_carbon_intensity", "w": "ipw_weight"})
pairs = pairs.merge(cc, left_on="GCoM_ID", right_index=True, how="left")
assert pairs.baseline_inventory_id.notna().all() and pairs.monitoring_inventory_id.notna().all()
print(f"(2) inventory pairs: {len(pairs)} municipalities;", pairs.coverage_change.value_counts().to_dict())

# ---- (3) NUTS codes by municipality
first = main.drop_duplicates("GCoM_ID").set_index("GCoM_ID")
nuts = extra.drop_duplicates("GCoM_ID")[["GCoM_ID", "country_code", "N3", "N2", "N1"]].rename(
    columns={"N3": "nuts3_2016", "N2": "nuts2_2016", "N1": "nuts1_2016"}).reset_index(drop=True)
nuts["gdp_value_proxy_group"] = nuts.GCoM_ID.map(first.country_code + "_" + first.gdp_per_capita_nuts3_eur.round(2).astype(str))
print(f"(3) NUTS3 assignment: {len(nuts)} municipalities, {nuts.nuts3_2016.nunique()} NUTS3 regions, "
      f"{nuts.gdp_value_proxy_group.nunique()} GDP-value proxy groups")

# ---- (4) variable descriptions
sector_txt = ", ".join(SECTOR_NAMES)
dic = [("coverage_indicators.csv", "GCoM_ID", "Signatory identifier in the JRC GCoM dataset"),
       ("coverage_indicators.csv", "emission_inventory_id", "Inventory identifier in the JRC GCoM dataset"),
       ("coverage_indicators.csv", "inventory_type", "baseline_emission_inventory or monitoring_emission_inventory"),
       ("coverage_indicators.csv", "inventory_year", "Inventory year"),
       ("coverage_indicators.csv", "country_code", "Two-letter country code"),
       ("coverage_indicators.csv", "reports_<sector>", f"1 if the inventory reports emissions for the sector ({sector_txt})"),
       ("coverage_indicators.csv", "n_reported_sectors", "Number of reported sectors (1-6)"),
       ("coverage_indicators.csv", "core_five_complete", "1 if the five core sectors are complete (core5_complete_flag of the analysis file)"),
       ("coverage_indicators.csv", "complete_reporter", "1 if at least five sectors are reported and the core five are complete (benchmark sample)"),
       ("coverage_indicators.csv", "co2eq_units", "1 if emissions are reported in CO2-equivalent units, 0 if in CO2"),
       ("coverage_indicators.csv", "lca_factors", "1 if life-cycle (LCA) emission factors are used, 0 if IPCC factors"),
       ("coverage_indicators.csv", "population_mismatch_flag", "1 if inventory-year population differs from the 2018 population by a factor above 2"),
       ("coverage_indicators.csv", "integer_coordinates_flag", "1 if both coordinates are whole degrees (excluded from spatial analyses)"),
       ("inventory_pairs.csv", "GCoM_ID", "Municipality with exactly one baseline and one later monitoring inventory in the analysis sample"),
       ("inventory_pairs.csv", "baseline_inventory_id / monitoring_inventory_id", "Inventory identifiers of the pair"),
       ("inventory_pairs.csv", "years_between", "Monitoring year minus baseline year"),
       ("inventory_pairs.csv", "coverage_change", "same coverage, removal only, addition only or both"),
       ("inventory_pairs.csv", "constant_coverage", "1 if the same sectors are reported in both inventories"),
       ("inventory_pairs.csv", "n_sectors_removed / n_sectors_added", "Number of sectors reported only in the baseline / only in the monitoring inventory"),
       ("inventory_pairs.csv", "sectors_removed / sectors_added", "Names of those sectors"),
       ("inventory_pairs.csv", "emission_factor_type_change", "1 if the emission-factor type (IPCC or LCA) differs between the two inventories"),
       ("inventory_pairs.csv", "unit_change", "1 if the reporting unit (CO2 or CO2-equivalent) differs between the two inventories"),
       ("inventory_pairs.csv", "dln_total_emissions", "Change in ln total reported emissions"),
       ("inventory_pairs.csv", "dln_emissions_common_sectors", "Change in ln emissions summed over sectors reported in both inventories"),
       ("inventory_pairs.csv", "dln_population", "Change in ln inventory-year population"),
       ("inventory_pairs.csv", "dln_energy_use", "Change in ln reported energy use (MWh); missing if not reported in both inventories"),
       ("inventory_pairs.csv", "annual_pct_total / _per_capita / _energy_use / _carbon_intensity",
        "Constant-coverage pairs only: annualised change in percent (100 x log change / years); intensity = total minus energy use"),
       ("inventory_pairs.csv", "ipw_weight", "Constant-coverage pairs only: inverse probability weight 1/p from the monitoring-selection logit"),
       ("nuts3_assignment.csv", "nuts3_2016 / nuts2_2016 / nuts1_2016", "NUTS 2016 codes from Table 3 of the JRC GCoM dataset"),
       ("nuts3_assignment.csv", "gdp_value_proxy_group", "GDP-value proxy identifier: country code and NUTS3 GDP per capita value (compared with the NUTS3 codes in Table S9)")]
cov.to_csv(os.path.join(DER_DIR, "coverage_indicators.csv"), index=False)
pairs.to_csv(os.path.join(DER_DIR, "inventory_pairs.csv"), index=False)
nuts.to_csv(os.path.join(DER_DIR, "nuts3_assignment.csv"), index=False)
pd.DataFrame(dic, columns=["file", "variable", "description"]).to_csv(os.path.join(DER_DIR, "derived_data_dictionary.csv"), index=False)
print("saved:", ", ".join(sorted(os.listdir(DER_DIR))), "in", DER_DIR)
