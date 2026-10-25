# -*- coding: utf-8 -*-
"""
01_data.py - variable construction, data-quality flags, NUTS3 ('province') identifier, analysis sample.

Input : DATA_PATH (top of lib.py)
Output: work/data.pkl (full data 'df' and analysis sample 'main')
        figures/supplementary/Figure_S1 (map of sample locations; Fig. S1)
Manuscript: Section 4 (Data), Supplementary Section S1 (sample rows of Table S1, Fig. S1)
"""
import warnings
from lib import *

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
banner("01 data: variables, flags, NUTS3 identifier, analysis sample")
raw = pd.read_csv(DATA_PATH)
df = raw.copy()
df["lnE"] = np.log(df.total_emissions_reported_tonnes)
df["lnPop"] = np.log(df.population_inventory_year)
df["lnGDP"] = np.log(df.gdp_per_capita_nuts3_eur)
df["lnDens"] = np.log(df.population_density_2018_per_km2)
df["HDDk"] = df.hdd / 1000.0
df["co2eq"] = df.emission_unit.str.contains("eq", na=False).astype(int)
df["lca"] = (df.emission_factor_type == "LCA").astype(int)
df["mon"] = (df.inventory_type == "monitoring_emission_inventory").astype(int)
df["year_c"] = df.inventory_year - 2008
df["lnEpc"] = np.log(df.reported_emissions_per_capita)
df["city_flag"] = (df.city_or_greater_city != "Other").astype(int)
df["sc"] = df.reported_sector_count
for s in [1, 2, 3, 4, 6]:
    df[f"sc{s}"] = (df.sc == s).astype(int)
df["sc_le3"] = (df.sc <= 3).astype(int)

# ---- data-quality flags
df["nonmunicipal"] = df.signatory_name.str.lower().str.contains(NONMUNI_PATTERN, regex=True, na=False).astype(int)   # the one record with a blank name is kept as a municipality
df["int_coords"] = ((df.latitude == df.latitude.round(0)) & (df.longitude == df.longitude.round(0))).astype(int)
ratio = df.population_inventory_year / df.population_2018
df["pop_mismatch"] = ((ratio < 0.5) | (ratio > 2)).fillna(False).astype(int)
# GDP-value proxy identifier: country + NUTS3 GDP per capita value, which almost always identifies one NUTS3 region (checked in stage 13)
df["nuts3"] = df.country_code + "_" + df.gdp_per_capita_nuts3_eur.round(2).astype(str)
if USE_REAL_NUTS3:   # setting at the top of lib.py: use the real NUTS3 (2016) codes from the JRC file (signatories without a code keep the proxy)
    _, _t3 = load_jrc()
    df["nuts3"] = df.GCoM_ID.map(_t3.set_index("GCoM_ID").NUTS3).fillna(df.nuts3)
    print("USE_REAL_NUTS3 = True: province identifier = real NUTS3 (2016) code from the JRC file")
df["sector_set"] = df[SECTOR_COLS].notna().apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
need = ["lnE", "lnPop", "lnGDP", "lnDens", "HDDk"]
df["covars_ok"] = df[need].notna().all(axis=1).astype(int)
main = df[(df.covars_ok == 1) & (df.nonmunicipal == 0)].copy().reset_index(drop=True)

print(f"raw: {len(df)} inventories / {df.GCoM_ID.nunique()} signatories | covariates complete: {int(df.covars_ok.sum())} | "
      f"non-municipal: {int(df.nonmunicipal.sum())} inventories ({df.loc[df.nonmunicipal == 1, 'GCoM_ID'].nunique()} entities)")
print(f"analysis sample: {len(main)} inventories / {main.GCoM_ID.nunique()} municipalities / {main.country_code.nunique()} countries / "
      f"{main.nuts3.nunique()} province proxies")
print("country shares (%):", (main.country_code.value_counts(normalize=True).head(5) * 100).round(1).to_dict())
print("complete reporters (>=5 sectors, core five):", int(((main.sc >= 5) & (main.core5_complete_flag == 1)).sum()),
      "| baseline:", int((main.mon == 0).sum()), "| monitoring:", int((main.mon == 1).sum()),
      "| spatial sample (baseline, non-integer coordinates):", int(((main.mon == 0) & (main.int_coords == 0)).sum()))
save_obj("data.pkl", dict(df=df, main=main))

# ---- Fig. S1: sample locations
cty = country_group(main)
fig, ax = plt.subplots(figsize=(FIG_W, 5.2))
colors = {"it": "#0072B2", "es": "#E69F00", "be": "#000000", "pt": "#56B4E9", "el": "#D55E00", "other": "#999999"}   # colour-vision-safe (Okabe and Ito)
for c in ["other", "el", "pt", "be", "es", "it"]:
    msk = cty == c
    ax.scatter(main.longitude[msk], main.latitude[msk], s=1.5, c=colors[c], label=f"{c} ({msk.sum()})", rasterized=True)
ax.set_xlim(-12, 32); ax.set_ylim(33, 63); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.legend(frameon=False, markerscale=5, title="country (inventories)")
# no title inside the figure: the caption carries the title and the sample counts
plt.tight_layout()
save_figure(fig, FIG_SUPP, "Figure_S1", tight=True)   # the map disclaimer is given in the figure caption
print("saved: work/data.pkl, figures/supplementary/Figure_S1")
