# -*- coding: utf-8 -*-
"""
05_asymmetry_tests.py - asymmetry test LR(sigma_u = sigma_w) by sample (two-tier specification (c)).

LR(sigma_u = 0) rejects even under symmetric heavy-tailed noise (Monte Carlo, stage 07), so it is not evidence of an
excess tier. What is identified is the asymmetry sigma_u - sigma_w; the chi2(1) 1% critical value is 6.63.
The unrestricted and restricted (sigma_u = sigma_w) models are run to full convergence from several starting values
(for the restricted model including a grid search over the common scale), and the solution with the highest likelihood
is kept. In samples where sigma_v reaches the boundary at 0 (three or fewer sectors, cities), a single starting value
inflates the LR badly.
Input : work/data.pkl
Output: work/asym.pkl, tables/asym_tests.csv
Manuscript: Table S3 (asymmetry-test panel), Fig. S5(b), Supplementary Section S3 (excess-side asymmetry only in Spain
            [all inventories and complete reporters], Belgium, and four- and six-sector inventories)
"""
import warnings
from lib import *

# hide the RuntimeWarnings (overflow) raised when the optimiser passes through extreme parameter regions; results are unaffected
warnings.filterwarnings("ignore", category=RuntimeWarning)

banner("05 asymmetry tests by sample (Table S3 panel, Fig. S5b)")
main = load_obj("data.pkl")["main"]


def run(s, name, fe=True):
    s = s.reset_index(drop=True)
    f = "country_code" if (fe and s.country_code.nunique() > 1) else None
    cols = BASE + [c for c in Q if s[c].nunique() > 1]
    X, _ = design(s, cols, fe=f)
    y = s.lnE.values
    r = sfa_2tier_best(y, X)                                 # unrestricted model: highest likelihood over three starting values
    sym = sfa_2tier_sym(y, X, r["theta"], profile=True)     # restricted model: four starting values + grid search over the common scale
    lw = sfa(-y, -X, "exp", compute_se=False)["llf"]
    row = dict(sample=name, n=len(s), sv=round(r["sv"], 3), su=round(r["su"], 3), sw=round(r["sw"], 3),
               LR_su0=round(2 * (r["llf"] - lw), 1), LR_asym=round(2 * (r["llf"] - sym["llf"]), 1),
               skew=round(skew_tests(ols(y, X)["e"])["skew"], 2))
    print(f"  {name:28s} n={row['n']:5d} su={row['su']:.3f} sw={row['sw']:.3f} LR(su=0)={row['LR_su0']:6.1f} "
          f"LR(su=sw)={row['LR_asym']:6.1f} skew={row['skew']:.2f}", flush=True)
    return row


HQ = (main.sc >= 5) & (main.core5_complete_flag == 1)
rows = [run(main.assign(**{c: 0 for c in Q}), "MAIN (a) base covariates", fe=False),   # reporting variables set to 0 (hence dropped) = specification (a)
        run(main, "MAIN (c) base+Q+country FE")]
for name, s in {"HighQ (sc>=5 & core5)": main[HQ], "sc=6": main[main.sc == 6], "sc=5": main[main.sc == 5],
                "sc=4": main[main.sc == 4], "sc<=3": main[main.sc <= 3], "Italy": main[main.country_code == "it"],
                "Spain": main[main.country_code == "es"], "Belgium": main[main.country_code == "be"],
                "CO2": main[main.co2eq == 0], "CO2eq": main[main.co2eq == 1],
                "cities (degurb=1)": main[main.degree_urbanisation == 1],
                "pop>=20k": main[main.population_inventory_year >= 20000],
                "Italy HighQ": main[(main.country_code == "it") & HQ], "Spain HighQ": main[(main.country_code == "es") & HQ]}.items():
    rows.append(run(s, name))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(TAB_DIR, "asym_tests.csv"), index=False, encoding="utf-8-sig")
save_obj("asym.pkl", tab)
print("saved: work/asym.pkl, tables/asym_tests.csv")
