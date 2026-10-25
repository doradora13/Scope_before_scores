# -*- coding: utf-8 -*-
"""
lib.py - settings and functions shared by all numbered scripts (01-24).

Settings: edit only the USER SETTINGS block below; no other code needs to change.
  DATA_PATH      : input data file GlobalCoM_SFA_ready.csv. Default: the code folder, i.e. the folder
                   that contains lib.py and the numbered scripts.
  OUT_DIR        : output folder for figures, Excel tables and the intermediate results passed between
                   stages (OUT_DIR/work). Default: a 'results' subfolder of the code folder. Missing
                   folders are created automatically.
  USE_REAL_NUTS3 : NUTS3 (province) identifier used from stage 01 on for clustered standard errors,
                   NUTS3 fixed effects and the NUTS3 cluster bootstrap.
                   True (default): real NUTS3 (2016) codes from the JRC ancillary table (Table 3 of
                     GlobalCoM.xlsx), 546 regions. Reproduces the submitted manuscript.
                   False: GDP-value proxy identifier (country code + NUTS3 GDP per capita), 543 groups.
                     A sensitivity option only: no reported result uses it, and stage 13 compares the two
                     identifiers (Table S9) under either setting. With False, stage 12 (check against the
                     manuscript numbers) reports DIFF for the NUTS3-dependent numbers; this is expected.
  JRC_XLSX, NEEFE_XLSX, FUA_GEOJSON : public source files. A missing file is downloaded on first use
                   (internet access needed) and saved at the path given in the settings block.
  MYCOV7_*, LAU_GEOJSON, NUTS16_*, NUTS21_CODES, HDD_CSV, GDP_CSV : public source files of the check on the
                   MyCovenant 7th release (stage 23; downloaded on first use like the files above).
  MYCOV7_DATA_PATH, MYCOV7_EARLIEST_PATH : analysis-ready 7th-release extracts read by stage 24. If a file is
                   missing at that path, stage 24 uses the copy that stage 23 wrote to OUT_DIR/work.

To use another location, replace the right-hand side of DATA_PATH or OUT_DIR by a full path in quotes:
  Windows       : "C:/data/GlobalCoM_SFA_ready.csv" (forward slashes work; for a path with backslashes,
                  put r before the opening quote, as in the examples in the settings block)
  macOS / Linux : "/path/to/GlobalCoM_SFA_ready.csv"

The estimators in this file use only numpy and scipy (statsmodels is not required).
Frontier sign convention: cost frontier y = Xb + v + u (u >= 0); two-tier frontier y = Xb + v + u - w.
"""
import os
# Reproducibility: run the linear-algebra libraries (BLAS/OpenMP) on one thread unless already set; this must happen before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import optimize, stats, sparse
from scipy.special import log_ndtr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================== USER SETTINGS ==============================
# By default the input file is read from the code folder (HERE = the folder that contains this file) and all
# results are written to its 'results' subfolder. To use other locations, replace the right-hand sides below
# by full paths in quotes, for example
#   Windows    : DATA_PATH = r"C:\data\GlobalCoM_SFA_ready.csv"    OUT_DIR = r"C:\data\results"   (keep the r)
#   macOS/Linux: DATA_PATH = "/path/to/GlobalCoM_SFA_ready.csv"    OUT_DIR = "/path/to/results"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "GlobalCoM_SFA_ready.csv")   # input data file
OUT_DIR = os.path.join(HERE, "results")                       # output folder (created if missing)
# --- Public source files. A missing file is downloaded on first use from the URL defined below and saved at
#     this path (internet access needed). JRC_XLSX is read by stage 01 (if USE_REAL_NUTS3 = True) and stage 13,
#     FUA_GEOJSON by stage 15, NEEFE_XLSX by stage 16.
JRC_XLSX = os.path.join(HERE, "GlobalCoM.xlsx")                        # JRC GlobalCoM source data (NUTS3 codes, direct/indirect emissions and activity)
NEEFE_XLSX = os.path.join(HERE, "JRC-COM-NEEFE_1990-2020.xlsx")        # JRC national electricity emission factors, 1990-2020
FUA_GEOJSON = os.path.join(HERE, "URAU_RG_100K_2018_4326_FUA.geojson")  # Eurostat GISCO functional urban area (FUA) boundaries, 2018
# --- Stages 23-24: check on the MyCovenant 7th release (Supplementary Section S17). Stage 23 reads the source files below
#     (each downloaded on first use) and writes the analysis-ready extracts to OUT_DIR/work; stage 24 reads the extracts at
#     MYCOV7_DATA_PATH and MYCOV7_EARLIEST_PATH and falls back to the stage-23 copies when a file is missing there.
MYCOV7_DF1 = os.path.join(HERE, "df1_Signatories_7th Release - January 2026.xlsx")         # MyCovenant 7th release: signatories
MYCOV7_DF2 = os.path.join(HERE, "df2_Action_Plans_7th Release - January 2026.xlsx")        # action plans and their emission inventories
MYCOV7_DF3 = os.path.join(HERE, "df3_Monitoring_Reports_7th Release - January 2026.xlsx")  # monitoring reports and their emission inventories
LAU_GEOJSON = os.path.join(HERE, "LAU_RG_01M_2018_4326.geojson")              # Eurostat GISCO local administrative units 2018 (population, area)
NUTS16_GEOJSON = os.path.join(HERE, "NUTS_RG_01M_2016_4326_LEVL_3.geojson")   # Eurostat GISCO NUTS 2016 level-3 boundaries
NUTS16_CODES = os.path.join(HERE, "NUTS_AT_2016.csv")                         # NUTS 2016 code list
NUTS21_CODES = os.path.join(HERE, "NUTS_AT_2021.csv")                         # NUTS 2021 code list
HDD_CSV = os.path.join(HERE, "nrg_chddr2_a.csv")                              # Eurostat heating and cooling degree days by NUTS 3 region
GDP_CSV = os.path.join(HERE, "nama_10r_3gdp.csv")                             # Eurostat GDP by NUTS 3 region (used only for a check)
MYCOV7_DATA_PATH = os.path.join(HERE, "MyCovenant7_SFA_ready.csv")            # analysis-ready 7th-release extract (latest entry per signatory, type and year)
MYCOV7_EARLIEST_PATH = os.path.join(HERE, "MyCovenant7_SFA_ready_earliest.csv")  # the same with the earliest entry (sensitivity row of Table S25)
# --- NUTS3 (province) identifier, used from stage 01 on for clustered standard errors, NUTS3 fixed effects
#     and the NUTS3 cluster bootstrap:
#     True  (default): real NUTS3 (2016) codes from the JRC ancillary table (Table 3 of JRC_XLSX), 546 regions.
#                      Reproduces the submitted manuscript.
#     False          : GDP-value proxy identifier (country + NUTS3 GDP per capita), 543 groups. A sensitivity
#                      option only (Table S9 compares both identifiers under either setting); stage 12 then
#                      reports DIFF for NUTS3-dependent numbers.
USE_REAL_NUTS3 = True
# ===========================================================================
JRC_XLSX_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CoM/GlobalCOM/VER14-07-2021/GlobalCoM.xlsx"
NEEFE_XLSX_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CoM/LATEST/JRC-COM-NEEFE_1990-2020.xlsx"
FUA_GEOJSON_URL = "https://gisco-services.ec.europa.eu/distribution/v2/urau/geojson/URAU_RG_100K_2018_4326_FUA.geojson"
_MYCOV7_FTP = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GCOM-MyCovenant/2026_Seventh_release/"
MYCOV7_DF1_URL = _MYCOV7_FTP + "df1_Signatories_7th%20Release%20-%20January%202026.xlsx"
MYCOV7_DF2_URL = _MYCOV7_FTP + "df2_Action_Plans_7th%20Release%20-%20January%202026.xlsx"
MYCOV7_DF3_URL = _MYCOV7_FTP + "df3_Monitoring_Reports_7th%20Release%20-%20January%202026.xlsx"
LAU_GEOJSON_URL = "https://gisco-services.ec.europa.eu/distribution/v2/lau/geojson/LAU_RG_01M_2018_4326.geojson"
NUTS16_GEOJSON_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2016_4326_LEVL_3.geojson"
NUTS16_CODES_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/csv/NUTS_AT_2016.csv"
NUTS21_CODES_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/csv/NUTS_AT_2021.csv"
HDD_CSV_URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/NRG_CHDDR2_A/?format=SDMX-CSV&compressed=false"
GDP_CSV_URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/NAMA_10R_3GDP/?format=SDMX-CSV&compressed=false"

WORK_DIR = os.path.join(OUT_DIR, "work")                      # intermediate results passed between stages (pickle)
FIG_MAIN = os.path.join(OUT_DIR, "figures", "main")           # Figure_1..6 (PDF + 300 dpi PNG) = manuscript Figs. 1, S3, S8, S11, S9, 3
FIG_SUPP = os.path.join(OUT_DIR, "figures", "supplementary")  # Figure_S1..S6 = manuscript Figs. S1, S2, S4-S7
TAB_DIR = os.path.join(OUT_DIR, "tables")                     # Excel tables and CSV files
for _p in (WORK_DIR, FIG_MAIN, FIG_SUPP, TAB_DIR):
    os.makedirs(_p, exist_ok=True)

try:  # keep print from failing on Windows consoles that cannot encode some characters
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)

# ------------------------------------------------------------------ Variables
BASE = ["lnPop", "lnGDP", "lnDens", "HDDk", "year_c"]                     # structural covariates
Q = ["sc1", "sc2", "sc3", "sc4", "sc6", "core5_complete_flag", "co2eq", "lca", "mon"]  # reporting variables (reference category: five reported sectors)
ZW_COLS = ["sc_le3", "sc4", "sc6", "core5_complete_flag", "mon"]         # two-tier specification (d), Table S3: covariates z in ln sigma_w = delta0 + z'delta
SECTOR_NAMES = ["municipal", "tertiary", "residential", "manufacturing", "transport", "waste"]
SECTOR_COLS = [f"{s}_emissions_tonnes" for s in SECTOR_NAMES]
NONMUNI_PATTERN = (r"(provinc|communaut|communes|agglo|m[ée]tropol|county|region |région|unione|union |comunit[àa] montana|consorzio|"
                   r"\bgal\b|\bpays\b|asbl|diputaci|mancomunidad|comarca|cabildo|consell insular|dei comuni|di comuni|des communes|"
                   r"de communes|isola d'elba|valle caudina|valle del chiese|vall[ée]e de chamonix|obiettivozero|\bgrand\b|greater |^land )")
KP_CRIT = {1: {0.10: 1.642, 0.05: 2.706, 0.01: 5.412}, 2: {0.10: 3.808, 0.05: 5.138, 0.01: 8.273}}  # Kodde-Palm mixed chi-square critical values {df: {level: value}}
CHI2_1_CRIT_1PCT = 6.63
LN2 = np.log(2.0)
LNSQRT2PI = 0.5 * np.log(2 * np.pi)

# ------------------------------------------------------------------ Figure settings
FIG_W = 7.48   # full-width figure (inches) = 190 mm
# Figures use DejaVu Sans (shipped with Matplotlib) so that they do not depend on the machine's default font.
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})


def save_figure(fig, folder, name, tight=False):
    """Save as vector PDF (rasterised point layers at 600 dpi) and 300 dpi PNG. tight=True fits the bounding box so that titles, axis labels and annotations reaching beyond the figure edge are not clipped."""
    kw = dict(bbox_inches="tight", pad_inches=0.03) if tight else {}
    fig.savefig(os.path.join(folder, name + ".pdf"), dpi=600, **kw)   # rasterised point layers (maps, scatter plots) are embedded at 600 dpi
    fig.savefig(os.path.join(folder, name + ".png"), dpi=300, **kw)
    plt.close(fig)


# ------------------------------------------------------------------ Save and load helpers
def wpath(name):
    return os.path.join(WORK_DIR, name)


def save_obj(name, obj):
    with open(wpath(name), "wb") as f:
        pickle.dump(obj, f)


def load_obj(name):
    path = wpath(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the earlier step that creates it first (see README).")
    with open(path, "rb") as f:
        return pickle.load(f)


def banner(text):
    print("=" * 78)
    print(text)
    print("=" * 78, flush=True)


# ------------------------------------------------------------------ Design matrices
def design(d, cols, fe=None):
    """Constant + listed columns + optional fixed-effect dummies for column fe (first level dropped)."""
    X = [np.ones(len(d))] + [d[c].values.astype(float) for c in cols]
    names = ["const"] + list(cols)
    if fe is not None:
        dm = pd.get_dummies(d[fe], drop_first=True).astype(float)
        X += [dm[c].values for c in dm.columns]
        names += [f"{fe}={c}" for c in dm.columns]
    return np.column_stack(X), names


def country_group(main):
    """Country groups used in figures: it, es, be, pt, el, other."""
    return np.where(main.country_code.isin(["it", "es", "be", "pt", "el"]), main.country_code, "other")


def ns_raw(x, kn):
    """Natural cubic spline basis with knots kn (number of columns = len(kn) - 2; constant and linear terms not included)."""
    K = len(kn)
    dd = lambda k: (np.maximum(x - kn[k], 0) ** 3 - np.maximum(x - kn[-1], 0) ** 3) / (kn[-1] - kn[k])
    return np.column_stack([dd(k) - dd(K - 2) for k in range(K - 2)])


def spline_controls(main, cols=("lnPop", "lnGDP", "lnDens", "HDDk")):
    """Table 2, model (4): natural cubic spline terms for each structural covariate (five knots at the 0, 25, 50, 75 and 100th percentiles; columns standardised)."""
    out = []
    for c in cols:
        xx = main[c].values
        R = ns_raw(xx, np.quantile(xx, np.linspace(0, 1, 5)))
        out.append((R - R.mean(0)) / R.std(0))
    return out


# ------------------------------------------------------------------ Logit (selection into monitoring)
def logit_fit(y, X, tol=1e-10, maxit=100):
    """Logit maximum likelihood by Newton-Raphson, iterated to full convergence (same solution on any platform)."""
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-(X @ b)))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.lstsq(H, X.T @ (y - p), rcond=None)[0]
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b


def logit_cluster_se(y, X, b, cluster):
    """Cluster-robust (sandwich) standard errors of the logit coefficients."""
    pp = 1 / (1 + np.exp(-(X @ b)))
    H = X.T @ (X * (pp * (1 - pp))[:, None])
    Hi = np.linalg.pinv(H)
    S_ = X * (y - pp)[:, None]
    uq, inv = np.unique(cluster, return_inverse=True)
    M_ = np.zeros((len(uq), X.shape[1]))
    np.add.at(M_, inv, S_)
    return np.sqrt(np.diag(Hi @ (M_.T @ M_) @ Hi))


# ------------------------------------------------------------------ Quantile regression
def qreg(y, X, tau):
    """Quantile regression solved as a linear programme (HiGHS)."""
    from scipy.optimize import linprog
    n, k = X.shape
    A = sparse.hstack([sparse.csr_matrix(X), sparse.identity(n), -sparse.identity(n)]).tocsr()
    c = np.r_[np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)]
    r = linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * k + [(0, None)] * (2 * n), method="highs")
    return r.x[:k]


# ------------------------------------------------------------------ Repeated-inventory pairs
def build_pairs(main):
    """Changes for municipalities with both a baseline inventory (BEI) and a later monitoring inventory (MEI).
    Returns: d (one row per pair), s1 (pairs that drop exactly one sector and add none: observed change vs. the
    change ln(1 - share) implied by the dropped sector's share)."""
    names, cols = SECTOR_NAMES, SECTOR_COLS
    pairs = main.groupby("GCoM_ID").filter(lambda g: len(g) == 2 and g.mon.sum() == 1)
    b = pairs[pairs.mon == 0].set_index("GCoM_ID")
    m = pairs[pairs.mon == 1].set_index("GCoM_ID")
    rows, single = [], []
    for i in b.index.intersection(m.index):
        dy = m.loc[i, "inventory_year"] - b.loc[i, "inventory_year"]
        if dy <= 0:
            continue
        sbs, sms = b.loc[i, "sector_set"], m.loc[i, "sector_set"]
        keep = [k for k in range(6) if sbs[k] == "1" and sms[k] == "1"]
        rem = [k for k in range(6) if sbs[k] == "1" and sms[k] == "0"]
        add = [k for k in range(6) if sbs[k] == "0" and sms[k] == "1"]
        cat = "same coverage" if not rem and not add else ("removal only" if rem and not add else ("addition only" if add and not rem else "both"))
        kb = sum(b.loc[i, cols[k]] for k in keep) if keep else np.nan
        km = sum(m.loc[i, cols[k]] for k in keep) if keep else np.nan
        ab, am = b.loc[i, "total_activity_MWh"], m.loc[i, "total_activity_MWh"]
        okA = pd.notna(ab) and pd.notna(am) and ab > 0 and am > 0
        dT = m.loc[i, "lnE"] - b.loc[i, "lnE"]
        dA = np.log(am) - np.log(ab) if okA else np.nan
        rows.append(dict(id=i, cat=cat, dy=dy, dT=dT, dR=(np.log(km) - np.log(kb)) if (keep and kb > 0 and km > 0) else np.nan,
                         n_rem=len(rem), n_add=len(add), dP=m.loc[i, "lnPop"] - b.loc[i, "lnPop"], dA=dA,
                         cty=b.loc[i, "country_code"], nuts3=b.loc[i, "nuts3"],
                         rem=",".join(names[k] for k in rem), add=",".join(names[k] for k in add),
                         ef_change=int(b.loc[i, "lca"] != m.loc[i, "lca"]), unit_change=int(b.loc[i, "co2eq"] != m.loc[i, "co2eq"])))
        if len(rem) == 1 and not add:
            s_ = b.loc[i, cols[rem[0]]] / b.loc[i, "total_emissions_reported_tonnes"]
            single.append(dict(sector=names[rem[0]], share=s_, obs=dT, mech=np.log(max(1 - s_, 1e-6))))
    return pd.DataFrame(rows), pd.DataFrame(single)


# ============================================================================
# Estimators: OLS, skewness tests, one- and two-tier stochastic frontiers, k-NN spatial weights, Moran's I
# ============================================================================

def ols(y, X, cluster=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    if cluster is None:
        s2 = e @ e / (n - k)
        V = s2 * XtX_inv
    else:
        cl = np.asarray(cluster)
        uniq, inv = np.unique(cl, return_inverse=True)
        G = len(uniq)
        S = np.zeros((k, k))
        Xe = X * e[:, None]
        # sum scores within clusters
        M = np.zeros((G, k))
        np.add.at(M, inv, Xe)
        S = M.T @ M
        adj = (G / (G - 1)) * ((n - 1) / (n - k))
        V = adj * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    sig2 = e @ e / n
    llf = -0.5 * n * (np.log(2 * np.pi) + np.log(sig2) + 1)
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return dict(b=b, se=se, t=b / se, e=e, llf=llf, r2=r2, n=n, k=k, sig2=sig2)


def skew_tests(e):
    """Coelli (1995) M3T test and D'Agostino skewness test on OLS residuals."""
    e = np.asarray(e); n = len(e)
    m2 = np.mean(e ** 2); m3 = np.mean(e ** 3)
    m3t = m3 / np.sqrt(6 * m2 ** 3 / n)
    z, p = stats.skewtest(e)
    return dict(skew=stats.skew(e), m3t=m3t, m3t_p=2 * (1 - stats.norm.cdf(abs(m3t))),
                dagostino_z=z, dagostino_p=p)


def _loglik_vec(theta, y, X, dist, Z=None):
    """Per-observation log-likelihood for cost frontier y = Xb + v + u."""
    k = X.shape[1]
    b = theta[:k]
    eps = y - X @ b
    if dist == 'hn':
        lsu, lsv = theta[k], theta[k + 1]
        su = np.exp(lsu); sv = np.exp(lsv)
        if Z is not None:  # heteroscedastic sigma_u = exp(Z d)
            d = theta[k + 2:]
            su = np.exp(lsu + Z @ d)
        s2 = su ** 2 + sv ** 2; s = np.sqrt(s2); lam = su / sv
        return LN2 - np.log(s) - LNSQRT2PI - 0.5 * (eps / s) ** 2 + log_ndtr(eps * lam / s)
    if dist == 'exp':
        lsu, lsv = theta[k], theta[k + 1]
        su = np.exp(lsu); sv = np.exp(lsv)
        return -np.log(su) + log_ndtr(eps / sv - sv / su) - eps / su + sv ** 2 / (2 * su ** 2)
    if dist == 'tn':
        lsu, lsv, mu = theta[k], theta[k + 1], theta[k + 2]
        su = np.exp(lsu); sv = np.exp(lsv)
        s2 = su ** 2 + sv ** 2; s = np.sqrt(s2); lam = su / sv
        return (-np.log(s) - LNSQRT2PI - 0.5 * ((eps - mu) / s) ** 2
                + log_ndtr(mu / (s * lam) + eps * lam / s) - log_ndtr(mu / su))
    raise ValueError(dist)


def _jlms(theta, y, X, dist, Z=None):
    """E[u|eps] (JLMS) for the cost frontier."""
    k = X.shape[1]
    eps = y - X @ theta[:k]
    if dist == 'hn':
        su = np.exp(theta[k]); sv = np.exp(theta[k + 1])
        if Z is not None:
            su = np.exp(theta[k] + Z @ theta[k + 2:])
        s2 = su ** 2 + sv ** 2
        mus = eps * su ** 2 / s2; ss = np.sqrt(su ** 2 * sv ** 2 / s2)
        r = mus / ss
        return mus + ss * np.exp(stats.norm.logpdf(r) - log_ndtr(r))
    if dist == 'exp':
        su = np.exp(theta[k]); sv = np.exp(theta[k + 1])
        mut = eps - sv ** 2 / su
        r = mut / sv
        return mut + sv * np.exp(stats.norm.logpdf(r) - log_ndtr(r))
    if dist == 'tn':
        su = np.exp(theta[k]); sv = np.exp(theta[k + 1]); mu = theta[k + 2]
        s2 = su ** 2 + sv ** 2
        mus = (su ** 2 * eps + mu * sv ** 2) / s2; ss = np.sqrt(su ** 2 * sv ** 2 / s2)
        r = mus / ss
        return mus + ss * np.exp(stats.norm.logpdf(r) - log_ndtr(r))


def sfa(y, X, dist='hn', Z=None, cluster=None, start=None, verbose=False, compute_se=True):
    """Fit cost-type SFA by ML. Returns dict with coefficients, robust SEs, JLMS u, etc."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    o = ols(y, X)
    e = o['e']
    m2 = np.mean(e ** 2); m3 = np.mean(e ** 3)
    # method-of-moments starting values (half-normal)
    su2 = (m3 / (np.sqrt(2 / np.pi) * (4 / np.pi - 1))) ** (2 / 3) if m3 > 0 else 0.3 * m2
    sv2 = max(m2 - (1 - 2 / np.pi) * su2, 0.2 * m2)
    b0 = o['b'].copy()
    if dist == 'hn':
        b0[0] -= np.sqrt(2 / np.pi * su2)
        th0 = np.r_[b0, 0.5 * np.log(su2), 0.5 * np.log(sv2)]
        if Z is not None:
            th0 = np.r_[th0, np.zeros(Z.shape[1])]
    elif dist == 'exp':
        su = np.sqrt(su2); b0[0] -= su
        th0 = np.r_[b0, np.log(su), 0.5 * np.log(sv2)]
    elif dist == 'tn':
        b0[0] -= np.sqrt(2 / np.pi * su2)
        th0 = np.r_[b0, 0.5 * np.log(su2), 0.5 * np.log(sv2), 0.0]
    if start is not None:
        th0 = start
    nll = lambda th: -np.sum(_loglik_vec(th, y, X, dist, Z))
    best = None
    bounds = [(None, None)] * k + [(-8, 4), (-8, 4)] + [(None, None)] * (len(th0) - k - 2)
    for method in ('L-BFGS-B',):
        try:
            res = optimize.minimize(nll, th0, method=method, bounds=bounds, options=dict(maxiter=5000))
            if best is None or res.fun < best.fun:
                best = res
        except Exception as ex:
            if verbose: print('opt fail', method, ex)
    res = best
    # polish with BFGS from the best point (helps flat sigma_u regions)
    res2 = optimize.minimize(nll, res.x, method='BFGS', options=dict(maxiter=5000, gtol=1e-6))
    if res2.fun <= res.fun:
        res = res2
    th = res.x
    llf = -res.fun
    # numerical Hessian & clustered/robust sandwich
    if compute_se:
        H = _num_hessian(nll, th)
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Hinv = np.linalg.pinv(H)
        J = _num_jac(lambda t: _loglik_vec(t, y, X, dist, Z), th)  # n x p scores
        if cluster is not None:
            cl = np.asarray(cluster); uniq, inv = np.unique(cl, return_inverse=True)
            M = np.zeros((len(uniq), len(th))); np.add.at(M, inv, J)
            S = M.T @ M
        else:
            S = J.T @ J
        V = Hinv @ S @ Hinv
        se = np.sqrt(np.abs(np.diag(V)))
    else:
        se = np.full(len(th), np.nan)
    u = _jlms(th, y, X, dist, Z)
    out = dict(theta=th, b=th[:k], se=se, t=th / se, llf=llf, llf_ols=o['llf'], n=n, k=k,
               u=u, ce=np.exp(-u), eps=y - X @ th[:k], dist=dist, converged=res.success,
               su=np.exp(th[k]), sv=np.exp(th[k + 1]))
    out['lr_sigma_u'] = 2 * (llf - o['llf'])
    out['lambda'] = out['su'] / out['sv']
    out['gamma'] = out['su'] ** 2 / (out['su'] ** 2 + out['sv'] ** 2)
    if dist == 'tn': out['mu'] = th[k + 2]
    if Z is not None: out['delta'] = th[k + 2:]; out['delta_se'] = se[k + 2:]
    p = len(th)
    out['aic'] = -2 * llf + 2 * p; out['bic'] = -2 * llf + np.log(n) * p
    return out


def _num_hessian(f, x, h=1e-4):
    p = len(x); H = np.zeros((p, p)); f0 = f(x)
    for i in range(p):
        for j in range(i, p):
            xi = x.copy(); xi[i] += h; xi[j] += h; fpp = f(xi)
            xi = x.copy(); xi[i] += h; xi[j] -= h; fpm = f(xi)
            xi = x.copy(); xi[i] -= h; xi[j] += h; fmp = f(xi)
            xi = x.copy(); xi[i] -= h; xi[j] -= h; fmm = f(xi)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    return H


def _num_jac(fvec, x, h=1e-5):
    p = len(x); f0 = fvec(x); J = np.zeros((len(f0), p))
    for i in range(p):
        xp = x.copy(); xp[i] += h; xm = x.copy(); xm[i] -= h
        J[:, i] = (fvec(xp) - fvec(xm)) / (2 * h)
    return J


def knn_weights(lat, lon, k=10):
    """Row-standardised k-nearest-neighbour weight matrix (great-circle distance)."""
    lat = np.radians(lat); lon = np.radians(lon)
    xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    tree = cKDTree(xyz)
    d, idx = tree.query(xyz, k=k + 1)
    n = len(lat)
    rows = np.repeat(np.arange(n), k); cols = idx[:, 1:].ravel()
    W = sparse.csr_matrix((np.full(n * k, 1.0 / k), (rows, cols)), shape=(n, n))
    # chord -> km distance of k-th neighbour, for reporting
    dk = 2 * 6371 * np.arcsin(np.clip(d[:, -1] / 2, 0, 1))
    return W, dk


def morans_i(x, W, nperm=499, seed=0):
    x = np.asarray(x, float); z = x - x.mean(); n = len(x)
    S0 = W.sum()
    I = (n / S0) * (z @ (W @ z)) / (z @ z)
    # analytical (normality) moments
    Wd = W + W.T
    S1 = 0.5 * np.sum(Wd.multiply(Wd))
    S2 = np.sum((np.asarray(W.sum(0)).ravel() + np.asarray(W.sum(1)).ravel()) ** 2)
    EI = -1.0 / (n - 1)
    VI = (n ** 2 * S1 - n * S2 + 3 * S0 ** 2) / ((n ** 2 - 1) * S0 ** 2) - EI ** 2
    zscore = (I - EI) / np.sqrt(VI)
    p_norm = 2 * (1 - stats.norm.cdf(abs(zscore)))
    # permutation p-value
    rng = np.random.default_rng(seed)
    sims = np.empty(nperm)
    for s in range(nperm):
        zp = rng.permutation(z)
        sims[s] = (n / S0) * (zp @ (W @ zp)) / (zp @ zp)
    p_perm = (np.sum(sims >= I) + 1) / (nperm + 1) if I > EI else (np.sum(sims <= I) + 1) / (nperm + 1)
    return dict(I=I, EI=EI, z=zscore, p_norm=p_norm, p_perm=p_perm)


def _tt_parts(theta, y, X, Zu=None, Zw=None):
    k = X.shape[1]
    b = theta[:k]; eps = y - X @ b
    lsv, lsu, lsw = theta[k], theta[k + 1], theta[k + 2]
    pos = k + 3
    su = np.exp(lsu); sw = np.exp(lsw); sv = np.exp(lsv)
    if Zu is not None:
        du = theta[pos:pos + Zu.shape[1]]; pos += Zu.shape[1]; su = np.exp(lsu + Zu @ du)
    if Zw is not None:
        dw = theta[pos:pos + Zw.shape[1]]; pos += Zw.shape[1]; sw = np.exp(lsw + Zw @ dw)
    A = -eps / su + sv ** 2 / (2 * su ** 2) + log_ndtr(eps / sv - sv / su)
    B = eps / sw + sv ** 2 / (2 * sw ** 2) + log_ndtr(-eps / sv - sv / sw)
    return eps, sv, su, sw, A, B


def _tt_loglik_vec(theta, y, X, Zu=None, Zw=None):
    eps, sv, su, sw, A, B = _tt_parts(theta, y, X, Zu, Zw)
    return -np.log(su + sw) + np.logaddexp(A, B)


def _tt_cond(theta, y, X, Zu=None, Zw=None):
    """E[u|eps], E[w|eps]."""
    eps, sv, su, sw, A, B = _tt_parts(theta, y, X, Zu, Zw)
    lf = np.logaddexp(A, B)                      # log f(eps) + log(su+sw)
    m = eps - sv ** 2 / su; mp = -eps - sv ** 2 / sw
    # E[max(s,0)|eps] = exp(-eps/su + sv^2/(2su^2)) [m Phi(m/sv) + sv phi(m/sv)] / ((su+sw) f)
    t1 = np.exp(-eps / su + sv ** 2 / (2 * su ** 2) - lf) * (m * stats.norm.cdf(m / sv) + sv * stats.norm.pdf(m / sv))
    t2 = np.exp(eps / sw + sv ** 2 / (2 * sw ** 2) - lf) * (mp * stats.norm.cdf(mp / sv) + sv * stats.norm.pdf(mp / sv))
    c = su * sw / (su + sw)
    return t1 + c, t2 + c


def _tt_grad(theta, y, X, Zu=None, Zw=None):
    k = X.shape[1]
    b = theta[:k]; eps = y - X @ b
    lsv, lsu, lsw = theta[k], theta[k + 1], theta[k + 2]
    sv = np.exp(lsv); su = np.exp(lsu); sw = np.exp(lsw)
    pos = k + 3
    if Zu is not None: nu = Zu.shape[1]; su = np.exp(lsu + Zu @ theta[pos:pos + nu]); pos += nu
    if Zw is not None: nw = Zw.shape[1]; sw = np.exp(lsw + Zw @ theta[pos:pos + nw])
    a = eps / sv - sv / su; bb = -eps / sv - sv / sw
    A = -eps / su + sv ** 2 / (2 * su ** 2) + log_ndtr(a)
    B = eps / sw + sv ** 2 / (2 * sw ** 2) + log_ndtr(bb)
    L = np.logaddexp(A, B); pA = np.exp(A - L); pB = 1 - pA
    lam_a = np.exp(stats.norm.logpdf(a) - log_ndtr(a)); lam_b = np.exp(stats.norm.logpdf(bb) - log_ndtr(bb))
    dA_de = -1 / su + lam_a / sv; dB_de = 1 / sw - lam_b / sv
    dl_de = pA * dA_de + pB * dB_de
    g_b = -(X * dl_de[:, None]).sum(0)
    dA_dsv = sv / su ** 2 + lam_a * (-eps / sv ** 2 - 1 / su); dB_dsv = sv / sw ** 2 + lam_b * (eps / sv ** 2 - 1 / sw)
    g_lsv = np.sum(sv * (pA * dA_dsv + pB * dB_dsv))
    dA_dsu = eps / su ** 2 - sv ** 2 / su ** 3 + lam_a * sv / su ** 2
    dB_dsw = -eps / sw ** 2 - sv ** 2 / sw ** 3 + lam_b * sv / sw ** 2
    dl_dsu = -1 / (su + sw) + pA * dA_dsu; dl_dsw = -1 / (su + sw) + pB * dB_dsw
    g = [g_b, np.array([g_lsv]), np.array([np.sum(su * dl_dsu)]), np.array([np.sum(sw * dl_dsw)])]
    if Zu is not None: g.append((Zu * (su * dl_dsu)[:, None]).sum(0))
    if Zw is not None: g.append((Zw * (sw * dl_dsw)[:, None]).sum(0))
    return np.concatenate(g)


def sfa_2tier_fast(y, X, Zu=None, Zw=None, cluster=None, start=None, compute_se=False, polish=True, ftol=1e-12, gtol=1e-7):
    """Exponential two-tier stochastic frontier y = Xb + v + u - w by ML with the analytic gradient; optional Zu, Zw make ln sigma_u, ln sigma_w linear in covariates."""
    y = np.asarray(y, float); X = np.asarray(X, float); n, k = X.shape
    o = ols(y, X); s = np.sqrt(o['sig2'])
    th0 = np.r_[o['b'], np.log(0.6 * s), np.log(0.4 * s), np.log(0.5 * s)]
    if Zu is not None: th0 = np.r_[th0, np.zeros(Zu.shape[1])]
    if Zw is not None: th0 = np.r_[th0, np.zeros(Zw.shape[1])]
    if start is not None: th0 = start
    nll = lambda th: -np.sum(_tt_loglik_vec(th, y, X, Zu, Zw))
    grad = lambda th: -_tt_grad(th, y, X, Zu, Zw)
    bounds = [(None, None)] * k + [(-8, 4)] * 3 + [(None, None)] * (len(th0) - k - 3)
    res = optimize.minimize(nll, th0, jac=grad, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=20000, maxfun=50000, ftol=ftol, gtol=gtol))
    if polish:
        res2 = optimize.minimize(nll, res.x, jac=grad, method='BFGS', options=dict(maxiter=20000, gtol=1e-6))
        if res2.fun <= res.fun: res = res2
    th = res.x; llf = -res.fun
    if compute_se:
        H = _num_hessian(nll, th)
        try: Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError: Hinv = np.linalg.pinv(H)
        J = _num_jac(lambda t: _tt_loglik_vec(t, y, X, Zu, Zw), th)
        if cluster is not None:
            cl = np.asarray(cluster); uniq, inv = np.unique(cl, return_inverse=True)
            M = np.zeros((len(uniq), len(th))); np.add.at(M, inv, J); S = M.T @ M
        else: S = J.T @ J
        V = Hinv @ S @ Hinv; se = np.sqrt(np.abs(np.diag(V)))
    else: se = np.full(len(th), np.nan)
    Eu, Ew = _tt_cond(th, y, X, Zu, Zw); p = len(th)
    out = dict(theta=th, b=th[:k], se=se, t=th / se, llf=llf, llf_ols=o['llf'], n=n, k=k,
               sv=np.exp(th[k]), su=np.exp(th[k + 1]), sw=np.exp(th[k + 2]), u=Eu, w=Ew, ce=np.exp(-Eu), eps=y - X @ th[:k],
               aic=-2 * llf + 2 * p, bic=-2 * llf + np.log(n) * p, converged=res.success)
    pos = k + 3
    if Zu is not None: out['du'] = th[pos:pos + Zu.shape[1]]; out['du_se'] = se[pos:pos + Zu.shape[1]]; pos += Zu.shape[1]
    if Zw is not None: out['dw'] = th[pos:pos + Zw.shape[1]]; out['dw_se'] = se[pos:pos + Zw.shape[1]]
    return out


def sfa_2tier_sym(y, X, start, multistart=True, profile=False, grid=None):
    """Exponential two-tier fit under the symmetry restriction sigma_u = sigma_w (symmetric Laplace-normal composite).
    Returns llf; 2*(llf_free - llf_sym) ~ chi2(1) tests asymmetry, which is the only feature that distinguishes a
    genuine one-sided tier from symmetric heavy-tailed noise.
    The restricted model has local optima near the boundary sigma_v = 0, so a fit from a single starting point can
    inflate the LR statistic considerably.
    multistart=True : run to full convergence from four starting points (free-model estimates with ln s = mean of
                      ln sigma_u and ln sigma_w; Laplace-dominated, sigma_v = 0.01; normal-dominated, s = 0.01;
                      the free model's sigma_v with the larger of its two scales) and keep the highest likelihood.
    profile=True    : on a grid of the common scale s, fit (b, ln sigma_v) at each grid value and add the two best
                      points to the starting values."""
    y = np.asarray(y, float); X = np.asarray(X, float); k = X.shape[1]
    full = lambda th: np.r_[th[:k + 1], th[k + 1], th[k + 1]]
    nll = lambda th: -np.sum(_tt_loglik_vec(full(th), y, X))
    def grad(th):
        g = -_tt_grad(full(th), y, X); return np.r_[g[:k + 1], g[k + 1] + g[k + 2]]
    bounds = [(None, None)] * (k + 1) + [(-8, 4)]
    b0 = np.asarray(start[:k], float)
    cands = [np.r_[start[:k + 1], 0.5 * (start[k + 1] + start[k + 2])]]
    if multistart or profile:
        sd = np.sqrt(ols(y, X)['sig2'])
    if multistart:
        cands += [np.r_[b0, np.log(0.01), np.log(sd / np.sqrt(2))], np.r_[b0, np.log(sd), np.log(0.01)],
                  np.r_[b0, start[k], max(start[k + 1], start[k + 2])]]
    if profile:
        scored = []
        grid = np.exp(np.linspace(np.log(0.03), np.log(2.0), 10)) if grid is None else grid
        for s_ in grid:
            ls = np.log(s_)
            f_in = lambda t: nll(np.r_[t, ls])
            g_in = lambda t: grad(np.r_[t, ls])[:k + 1]
            for lsv0 in (np.log(0.05), np.log(max(0.5 * sd, 0.05))):
                r = optimize.minimize(f_in, np.r_[b0, lsv0], jac=g_in, method='L-BFGS-B', bounds=bounds[:k + 1], options=dict(maxiter=5000))
                if np.isfinite(r.fun): scored.append((r.fun, np.r_[r.x, ls]))
        scored.sort(key=lambda c: c[0])
        cands += [th for _, th in scored[:2]]
    best = None
    for th0 in cands:   # run each candidate to full convergence (strict L-BFGS-B, then BFGS); keep the highest likelihood
        res = optimize.minimize(nll, th0, jac=grad, method='L-BFGS-B', bounds=bounds, options=dict(maxiter=20000, maxfun=50000, ftol=1e-12, gtol=1e-7))
        res2 = optimize.minimize(nll, res.x, jac=grad, method='BFGS', options=dict(maxiter=20000, gtol=1e-6))
        if np.isfinite(res2.fun) and res2.fun <= res.fun: res = res2
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun): best = res
    return dict(llf=-best.fun, sv=np.exp(best.x[k]), s=np.exp(best.x[k + 1]))


def sfa_2tier_best(y, X, start=None):
    """Fit the unrestricted two-tier model from up to three starting points (the given start, the default OLS-based
    start, and a start with sigma_v = 0.01), run each to full convergence and keep the highest likelihood.
    This avoids local optima in samples and resamples where sigma_v reaches the zero boundary."""
    y = np.asarray(y, float); X = np.asarray(X, float); k = X.shape[1]
    o = ols(y, X); sd = np.sqrt(o['sig2'])
    b0 = o['b'] if start is None else np.asarray(start[:k], float)
    starts = ([np.asarray(start, float)] if start is not None else []) + [
        None,                                                                              # default starting values of sfa_2tier_fast
        np.r_[b0, np.log(0.01), np.log(sd / np.sqrt(2)), np.log(sd / np.sqrt(2))]]
    fits = [sfa_2tier_fast(y, X, start=st) for st in starts]
    fits = [f for f in fits if np.isfinite(f['llf'])]
    return max(fits, key=lambda f: f['llf'])


# ============================================================================
# Helpers for stages 13-16 (load_jrc also in stage 01): public-data download, JRC source tables, fixed-effect residuals,
# placebo partitions, FUA assignment
# ============================================================================
SECTOR_JRC = {"Municipal buildings and facilities": "municipal", "Institutional/tertiary buildings and facilities": "tertiary",
              "Residential buildings and facilities": "residential", "Manufacturing and construction industries": "manufacturing",
              "Transportation": "transport", "Waste/wastewater": "waste"}


def ensure_file(path, url):
    """Download url to path unless the file already exists. If the download fails, stop and print where to save a manually downloaded copy."""
    if os.path.exists(path):
        return path
    import urllib.request
    import time
    last = None
    for attempt in range(1, 6):          # up to five attempts, to ride out transient connection drops
        print(f"downloading {os.path.basename(path)} (attempt {attempt}) ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(path + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(path + ".part", path)
            return path
        except Exception as ex:
            last = ex
            if os.path.exists(path + ".part"):
                os.remove(path + ".part")
            time.sleep(3 * attempt)
    raise SystemExit(f"Could not download {url} ({last}).\nDownload it in a browser and save it as:\n  {path}")


def load_jrc():
    """Table 2 (inventory x sector x emission type) and Table 3 (municipality ancillary data) of JRC GlobalCoM.xlsx. Read once and cached in OUT_DIR/work."""
    cache = wpath("jrc_raw.pkl")
    if os.path.exists(cache):
        return load_obj("jrc_raw.pkl")
    ensure_file(JRC_XLSX, JRC_XLSX_URL)
    t2 = pd.read_excel(JRC_XLSX, sheet_name="Table 2")
    t3 = pd.read_excel(JRC_XLSX, sheet_name="Table 3").drop_duplicates("GCoM_ID")
    save_obj("jrc_raw.pkl", (t2, t3))
    return t2, t3


def jrc_inventory_extra():
    """Additional inventory-level variables (GCoM_ID x inventory type): direct/indirect emissions and activity of the energy sectors,
    activity by sector, NUTS3, LAU2 and FUA codes, and adhesion type.
    Activity data are not collected for the waste sector (JRC Table 1 notes), so energy-sector emissions = total emissions - waste."""
    t2, t3 = load_jrc()
    t2 = t2.copy()
    t2["sec"] = t2.emission_inventory_sector.map(SECTOR_JRC)
    t2["typ"] = np.where(t2.type_of_emissions == "direct_emissions", "dir", "ind")
    key = ["GCoM_ID", "type_of_emission_inventory"]
    tot = t2.groupby(key).agg(jrc_total_emissions=("emissions", "sum"), jrc_total_activity=("activity_data", "sum"))
    en = t2[t2.sec != "waste"]
    di = en.pivot_table(index=key, columns="typ", values=["emissions", "activity_data"], aggfunc="sum")
    di.columns = [f"{a}_{b}" for a, b in di.columns]
    sa = t2.pivot_table(index=key, columns="sec", values="activity_data", aggfunc="sum")
    sa.columns = [f"{c}_activity_MWh" for c in sa.columns]
    ee = en.groupby(key).emissions.sum().rename("energy_emissions_tonnes")
    out = tot.join(di).join(sa).join(ee).reset_index().rename(columns={"type_of_emission_inventory": "inventory_type"})
    t3s = t3.rename(columns={"signatory adhesion type": "adhesion_type", "FUA_ID": "FUA_ID_jrc"})[["GCoM_ID", "NUTS3", "LAU2", "FUA_ID_jrc", "adhesion_type"]]
    return out.merge(t3s, on="GCoM_ID", how="left")


def fe_resid(y, Xb, labels):
    """OLS residuals of y on Xb (structural and reporting variables) plus fixed effects for the categories in labels, and the number of parameters (for the adjusted R2)."""
    D = pd.get_dummies(pd.Series(np.asarray(labels)), drop_first=True).astype(float).values
    X = np.column_stack([Xb, D])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b, X.shape[1]


def moran_value(e, W):
    """Moran's I statistic only (no permutation inference)."""
    z = np.asarray(e, float) - np.mean(e)
    return (len(z) / W.sum()) * (z @ (W @ z)) / (z @ z)


def kmeans_np(P, k, rng, iters=100):
    """k-means (k-means++ initialisation, then Lloyd iterations). P: n x 2 coordinates (km)."""
    n = len(P)
    C = [P[rng.integers(n)]]
    d2 = ((P - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        C.append(P[rng.choice(n, p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((P - C[-1]) ** 2).sum(1))
    C = np.array(C)
    lab = np.zeros(n, int)
    for _ in range(iters):
        D = ((P[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = D.argmin(1)
        if _ > 0 and np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            m = lab == j
            C[j] = P[m].mean(0) if m.any() else P[rng.integers(n)]
    return lab


def voronoi_np(P, k, rng):
    """Nearest-seed partition with k randomly chosen municipalities as seeds (a placebo partition with unequal cell sizes)."""
    seeds = rng.choice(len(P), size=k, replace=False)
    return ((P[:, None, :] - P[seeds][None, :, :]) ** 2).sum(-1).argmin(1)


def xy_km(lat, lon):
    """Latitude/longitude -> local planar coordinates in km (equirectangular approximation). Used only for clustering within a country."""
    la = np.radians(np.asarray(lat, float))
    return np.c_[np.radians(np.asarray(lon, float)) * np.cos(la) * 6371.0, la * 6371.0]


def points_in_fua(lon, lat, path):
    """Code of the functional urban area (FUA) that contains each point (None if none). Point-in-polygon test on the GISCO GeoJSON with matplotlib Path (polygon holes are excluded)."""
    import json
    from matplotlib.path import Path as MPath
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    pts = np.c_[lon, lat]
    out = np.full(len(pts), None, dtype=object)
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        code = feat["properties"]["URAU_CODE"]
        for poly in polys:
            ext = np.asarray(poly[0])
            x0, y0 = ext.min(0); x1, y1 = ext.max(0)
            cand = np.where((out == None) & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1))[0]  # noqa: E711
            if len(cand) == 0:
                continue
            inside = MPath(ext).contains_points(pts[cand])
            for hole in poly[1:]:
                inside &= ~MPath(np.asarray(hole)).contains_points(pts[cand])
            out[cand[inside]] = code
    return out


# ============================================================================
# Helpers for stages 23-24 (check on the MyCovenant 7th release, Supplementary Section S17)
# ============================================================================
EPC_MIN, EPC_MAX = 0.01, 100.0   # per-capita screen: reported emissions per capita outside [0.01, 100] t (Supplementary Section S17)
MIN_DONORS, MIN_SET = 20, 20     # country donor pool if it has >= 20 six-sector inventories (else pooled); sector sets with >= 20 inventories
RELEASE_DTYPES = dict(low_memory=False, dtype={"action_plan_id": str, "emission_inventory_id": str, "GCoM_ID": str, "monitoring_report_id": str})


def prepare_release(df, nuts3):
    """Variables of stage 01 for an extract in the GlobalCoM_SFA_ready.csv schema (either release), plus the sector set,
    reported emissions per capita (epc) and the per-capita screen flag. nuts3: NUTS3 (2016) code of each inventory."""
    df = df.copy()
    df["lnE"] = np.log(df.total_emissions_reported_tonnes)
    df["lnPop"] = np.log(df.population_inventory_year)
    df["lnGDP"] = np.log(df.gdp_per_capita_nuts3_eur)
    df["lnDens"] = np.log(df.population_density_2018_per_km2)
    df["HDDk"] = df.hdd / 1000.0
    df["co2eq"] = df.emission_unit.str.contains("eq", na=False).astype(int)
    df["lca"] = (df.emission_factor_type == "LCA").astype(int)
    df["mon"] = (df.inventory_type == "monitoring_emission_inventory").astype(int)
    df["year_c"] = df.inventory_year - 2008
    df["sc"] = df.reported_sector_count
    for s in [1, 2, 3, 4, 6]:
        df[f"sc{s}"] = (df.sc == s).astype(int)
    df["sc_le3"] = (df.sc <= 3).astype(int)
    df["nonmunicipal"] = df.signatory_name.str.lower().str.contains(NONMUNI_PATTERN, regex=True, na=False).astype(int)
    df["nuts3"] = nuts3
    df["sector_set"] = df[SECTOR_COLS].notna().apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
    need = ["lnE", "lnPop", "lnGDP", "lnDens", "HDDk"]
    df["covars_ok"] = df[need].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).astype(int)
    df["epc"] = df.total_emissions_reported_tonnes / df.population_inventory_year
    df["gross_error"] = ((df.epc < EPC_MIN) | (df.epc > EPC_MAX)).astype(int)
    return df


def release_sample(df, screen=False, exclude_ids=()):
    """Analysis sample as in stage 01 (complete covariates, municipal signatories, NUTS3 known), optionally without the
    signatories in exclude_ids (joint plans of several municipalities) and without inventories outside the per-capita screen."""
    m = (df.covars_ok == 1) & (df.nonmunicipal == 0) & df.nuts3.notna() & ~df.GCoM_ID.isin(exclude_ids)
    if screen:
        m &= df.gross_error == 0
    return df[m].copy().reset_index(drop=True)


def load_2021_release():
    """2021 extract (DATA_PATH) with real NUTS3 (2016) codes from the JRC ancillary table, as in stage 01."""
    _, t3 = load_jrc()
    t3 = t3.drop_duplicates("GCoM_ID").set_index("GCoM_ID")
    d21 = pd.read_csv(DATA_PATH)
    nuts21 = d21.GCoM_ID.map(t3.NUTS3).fillna(d21.country_code + "_" + d21.gdp_per_capita_nuts3_eur.round(2).astype(str))
    return d21, prepare_release(d21, nuts21)


def load_7th_release(earliest=False):
    """Analysis-ready 7th-release extract: the file set in MYCOV7_DATA_PATH (or MYCOV7_EARLIEST_PATH), else the stage-23 copy."""
    path = MYCOV7_EARLIEST_PATH if earliest else MYCOV7_DATA_PATH
    if not os.path.exists(path):
        path = wpath("MyCovenant7_SFA_ready" + ("_earliest" if earliest else "") + ".csv")
    if not os.path.exists(path):
        raise SystemExit("7th-release extract not found: run stage 23 first (python 23_mycovenant7_extract.py).")
    d7 = pd.read_csv(path, **RELEASE_DTYPES)
    return d7, prepare_release(d7, d7.NUTS3)


def ols_V(yv, X, cluster):
    """OLS coefficients, NUTS3-clustered covariance (small-sample factors as in ols) and residuals."""
    X = np.asarray(X, float); n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ yv; e = yv - X @ b
    uniq, inv = np.unique(cluster, return_inverse=True); G = len(uniq)
    M = np.zeros((G, k)); np.add.at(M, inv, X * e[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX_inv @ (M.T @ M) @ XtX_inv
    return b, V, e


def dsg(d, cols, fe="country_code"):
    """design() without all-zero dummy columns (countries absent from a subsample)."""
    X, names = design(d, cols, fe=fe)
    keep = [j for j in range(X.shape[1]) if names[j] == "const" or np.any(X[:, j] != X[0, j])]
    return X[:, keep], [names[j] for j in keep]


def qskew(e, lo, hi):
    """Quantile skewness (q_lo + q_hi - 2 q_50) / (q_hi - q_lo); lo, hi = 0.10, 0.90 gives Kelley's measure."""
    a, m, b = np.quantile(e, [lo, 0.5, hi])
    return (a + b - 2 * m) / (b - a)


def keyed(m):
    """Copy of m with a (signatory, type, year) key used to match inventories between releases."""
    m = m.copy()
    m["key"] = list(zip(m.GCoM_ID.astype(str), m.inventory_type, m.inventory_year.astype(int)))
    return m
