# -*- coding: utf-8 -*-
"""
24_mycovenant7_replication.py - Steps 1, 2 and 4 of the screening protocol on the MyCovenant 7th release and on the 2021
dataset (Supplementary Section S17; Tables S23, S25, S26 and S27).

Same estimators, seeds, replication counts and donor rule as stage 02 (Table 2, model (3)) and stage 20 (Propositions 1-3):
donor shares from the six-sector inventories of each sample (country pool if it has >= 20 donors, otherwise pooled), NUTS3-clustered
standard errors, two-stage bootstrap of the shortfall coefficient (499 replications resampling NUTS3 regions within countries and
the donors), NUTS3 cluster bootstrap within countries for skewness, third-moment shares and the change decomposition (999).
Samples (Table S23 and S25): 2021 dataset (A) and with the per-capita screen (B); 7th release (C), its inventories not in 2021 (D)
and also in 2021 (E); without the screen (F); signatories with 2021 covariates (G); new signatories (H); one BEI and the latest
MEI per municipality (I); earliest entry of duplicated inventories (J); residential buildings only (K); the same 7,011 inventories
with 2021 values (L) and with 7th-release values (M).
Also: influence of single inventories on the third moment, the joint test of the sector-count indicators given the share-implied
shortfall with one data feature changed at a time (Table S26), and Proposition 3 applied between releases (Table S27).
Input : DATA_PATH and JRC_XLSX (2021 dataset, as in stage 01), the 7th-release extracts (MYCOV7_DATA_PATH and MYCOV7_EARLIEST_PATH,
        or the copies written by stage 23)
Output: work/mycovenant7.pkl, tables/MyCovenant7_Tables.xlsx, tables/mycovenant7_numbers.csv,
        tables/check_mycovenant7_numbers.csv (comparison with the numbers printed in the manuscript),
        tables/Tables_S23_S25_S26_S27.tex (LaTeX for Tables S23, S25, S26 and S27)
About 6 minutes on two cores (bootstrap).
"""
import itertools
import warnings
from lib import *
warnings.filterwarnings("ignore")

B, B_LAMBDA, SEED = 999, 499, 20     # replications (cluster bootstrap, two-stage bootstrap) and seed, as in stage 20
CATS = ["addition only", "removal only", "both"]

banner("24 MyCovenant 7th release: Steps 1, 2 and 4 (Supplementary Section S17; Tables S23, S25, S26, S27)")
nums = []


def note(name, value, txt=""):
    nums.append(dict(name=name, value=value, note=txt))


# ------------------------------------------------------------------ data
d21, df21 = load_2021_release()
main21 = release_sample(df21)
assert len(main21) == 7494, len(main21)
main21s = release_sample(df21, screen=True)
d7, df7 = load_7th_release()
joint = set(d7.loc[d7.joint_plan == 1, "GCoM_ID"])
main7_raw = release_sample(df7, exclude_ids=joint)
main7 = release_sample(df7, screen=True, exclude_ids=joint)
d7e, df7e = load_7th_release(earliest=True)
main7f = release_sample(df7e, screen=True, exclude_ids=joint)
d7r = d7.copy()                       # residential buildings only (agriculture and non-allocated buildings excluded)
d7r["residential_emissions_tonnes"] = d7r["residential_raw_emissions_tonnes"]
d7r["total_emissions_reported_tonnes"] = d7r[SECTOR_COLS].sum(axis=1, min_count=1)
d7r["reported_sector_count"] = d7r[SECTOR_COLS].notna().sum(axis=1)
d7r["core5_complete_flag"] = (d7r[SECTOR_COLS[:5]].notna().sum(axis=1) == 5).astype(int)
d7r = d7r[d7r.total_emissions_reported_tonnes > 0]
main7r = release_sample(prepare_release(d7r, d7r.NUTS3), screen=True, exclude_ids=joint)
print(f"2021: {len(main21)} inventories ({len(main21s)} after the per-capita screen); 7th release: {len(main7_raw)} without joint plans, "
      f"{len(main7)} after the screen")


def one_each(m):
    """One BEI (the latest) and the latest MEI per municipality."""
    b = m[m.mon == 0].sort_values("inventory_year").groupby("GCoM_ID").tail(1)
    mm = m[m.mon == 1].sort_values("inventory_year").groupby("GCoM_ID").tail(1)
    return pd.concat([b, mm]).sort_index().reset_index(drop=True)


a_all, b_all = keyed(main21s), keyed(main7)
common = sorted(set(a_all.key) & set(b_all.key))
same21 = a_all.set_index("key").loc[common].reset_index()      # the same inventories with 2021 values ...
same7 = b_all.set_index("key").loc[common].reset_index()       # ... and with 7th-release values
print(f"same inventories in both analysis samples: {len(common)}")


# ------------------------------------------------------------------ Steps 1 and 2 (stage 02 model (3); stage 20 Propositions 1-2)
def mech_w_map(sets, cty, share6, cty6, pooled=False):
    """Share-implied shortfall w_hat for each (sector set, country): mean over donors of -ln(sum of the shares of the reported sectors)."""
    donor_n = pd.Series(cty6).value_counts()

    def mw(setstr, c):
        mask = np.array([ch == "1" for ch in setstr])
        if mask.all():
            return 0.0
        D = share6 if (pooled or donor_n.get(c, 0) < MIN_DONORS) else share6[cty6 == c]
        return float(np.mean(-np.log(np.clip(D[:, mask].sum(1), 1e-6, None))))
    combo = pd.DataFrame({"s": sets, "c": cty}).drop_duplicates()
    return {(r.s, r.c): mw(r.s, r.c) for r in combo.itertuples()}


def steps12(main, label):
    out = dict(sample=label)
    main = main.reset_index(drop=True)
    y = main.lnE.values; cl = main.nuts3.values; cty = main.country_code.values
    sets = main.sector_set.values; sc = main.sc.values
    complete = ((main.sc >= 5) & (main.core5_complete_flag == 1)).values
    out.update(n=len(main), n_muni=int(main.GCoM_ID.nunique()), n_cty=int(main.country_code.nunique()), n_nuts3=int(main.nuts3.nunique()),
               n_bei=int((main.mon == 0).sum()), n_mei=int((main.mon == 1).sum()), median_year=float(main.inventory_year.median()),
               **{f"n_sc{k}": int((sc == k).sum()) for k in range(1, 7)}, n_complete=int(complete.sum()))
    X3, n3 = dsg(main, BASE + Q)
    o3 = ols(y, X3, cluster=cl)
    st = skew_tests(o3["e"])
    for v in BASE + Q:
        if v in n3:
            j = n3.index(v); out[f"b_{v}"] = o3["b"][j]; out[f"se_{v}"] = o3["se"][j]
    out.update(r2=o3["r2"], skew=st["skew"])
    E6 = main[SECTOR_COLS].values.astype(float)
    six = sc == 6
    share6 = E6[six] / E6[six].sum(1, keepdims=True); cty6 = cty[six]
    donor_n = pd.Series(cty6).value_counts()
    out["n_donors"] = int(six.sum())
    out["donor_pools"] = "; ".join(f"{c} {n}" for c, n in donor_n.items() if n >= MIN_DONORS) or "pooled only"
    out["donors_es61"] = int((main.loc[six, "nuts3"].astype(str).str[:4] == "ES61").sum())
    wmap = mech_w_map(sets, cty, share6, cty6)
    wmap_p = mech_w_map(sets, cty, share6, cty6, pooled=True)
    main["w_hat"] = [wmap[(s, c)] for s, c in zip(sets, cty)]
    main["w_hat_pooled"] = [wmap_p[(s, c)] for s, c in zip(sets, cty)]
    w_hat = main.w_hat.values
    Qnc = ["core5_complete_flag", "co2eq", "lca", "mon"]
    XA, nA = dsg(main, BASE + Qnc + ["w_hat"])
    bA, VA, eA = ols_V(y, XA, cl); iA = nA.index("w_hat")
    out.update(lam=bA[iA], lam_se=np.sqrt(VA[iA, iA]), lam_t_m1=(bA[iA] + 1) / np.sqrt(VA[iA, iA]))
    XB, nB = dsg(main, BASE + Q + ["w_hat"])
    bB, VB, eB = ols_V(y, XB, cl)
    idx_sc = [nB.index(c) for c in ["sc1", "sc2", "sc3", "sc4", "sc6"] if c in nB]
    Wd = float(bB[idx_sc] @ np.linalg.solve(VB[np.ix_(idx_sc, idx_sc)], bB[idx_sc]))
    out.update(wald=Wd, wald_df=len(idx_sc), wald_p=1 - stats.chi2.cdf(Wd, len(idx_sc)))
    XAp, nAp = dsg(main, BASE + Qnc + ["w_hat_pooled"])
    bAp, VAp, _ = ols_V(y, XAp, cl); ip = nAp.index("w_hat_pooled")
    out.update(lam_pooled=bAp[ip])
    Xo = np.delete(XA, iA, axis=1)
    r_w = w_hat - Xo @ np.linalg.lstsq(Xo, w_hat, rcond=None)[0]
    out["leverage_1_2"] = float(np.sum(r_w[sc <= 2] ** 2) / np.sum(r_w ** 2))
    # two-stage bootstrap: NUTS3 regions within countries and donors within their pools
    rngL = np.random.default_rng(SEED + 3)
    clu_by_cty_L = {c: np.unique(cl[cty == c]) for c in np.unique(cty)}
    rows_by_clu_L = {k: np.where(cl == k)[0] for k in np.unique(cl)}
    donor_idx = {c: np.where(cty6 == c)[0] for c in donor_n.index}
    combo = pd.DataFrame({"s": sets, "c": cty}).drop_duplicates()
    combo_masks = [(r.s, r.c, np.array([ch == "1" for ch in r.s])) for r in combo.itertuples()]
    key_idx = pd.Series(range(len(combo_masks)), index=pd.MultiIndex.from_tuples([(a, b_) for a, b_, _ in combo_masks]))
    inv_key = key_idx.reindex(pd.MultiIndex.from_arrays([sets, cty])).values
    lam_bs = []
    for _ in range(B_LAMBDA):
        di = np.concatenate([rngL.choice(ix, len(ix), replace=True) for ix in donor_idx.values()])
        sh_b, ct_b = share6[di], cty6[di]
        vals = np.empty(len(combo_masks))
        for j, (s_, c_, msk) in enumerate(combo_masks):
            if msk.all():
                vals[j] = 0.0; continue
            D = sh_b if donor_n.get(c_, 0) < MIN_DONORS else sh_b[ct_b == c_]
            vals[j] = np.mean(-np.log(np.clip(D[:, msk].sum(1), 1e-6, None)))
        wb = vals[inv_key]
        idx = np.concatenate([rows_by_clu_L[k] for c, ks in clu_by_cty_L.items() for k in rngL.choice(ks, len(ks), replace=True)])
        Xb = XA[idx].copy(); Xb[:, iA] = wb[idx]
        lam_bs.append(np.linalg.lstsq(Xb, y[idx], rcond=None)[0][iA])
    lam_bs = np.array(lam_bs)
    out.update(lam_boot_se=lam_bs.std(), lam_boot_lo=np.percentile(lam_bs, 2.5), lam_boot_hi=np.percentile(lam_bs, 97.5))
    # sector-set contrasts against share predictions (sets with >= 20 inventories)
    vc = pd.Series(sets).value_counts()
    big = [s for s in vc.index if vc[s] >= MIN_SET and s != "111111"]
    if "111111" in vc.index and len(big) >= 3:
        setcat = np.where(np.isin(sets, big + ["111111"]), sets, "other")
        Dset = pd.get_dummies(pd.Series(setcat)).astype(float)
        Dset = Dset[[s for s in big] + (["other"] if (setcat == "other").any() else [])]
        XS0, nS0 = dsg(main, BASE + ["co2eq", "lca", "mon"])
        XS = np.column_stack([XS0, Dset.values]); nS = nS0 + list(Dset.columns)
        bS, VS, eS = ols_V(y, XS, cl)
        t_sets = pd.DataFrame([dict(set=s, n=int(vc[s]), coef=bS[nS.index(s)], mechanical=-w_hat[sets == s].mean()) for s in big])
        out.update(sets_n=len(t_sets), sets_slope=np.polyfit(t_sets.mechanical, t_sets.coef, 1, w=np.sqrt(t_sets.n))[0],
                   sets_corr=np.corrcoef(t_sets.mechanical, t_sets.coef)[0, 1])
    # Step 2: residual asymmetry of model (3)
    groups = {"all": np.ones(len(main), bool), "complete": complete, "incomplete": ~complete, "le2": sc <= 2}
    rng = np.random.default_rng(SEED)
    clu_by_cty = {c: np.unique(cl[cty == c]) for c in np.unique(cty)}
    rows_by_clu = {k: np.where(cl == k)[0] for k in np.unique(cl)}
    bs = {g: [] for g in groups}; bs_sh = {g: [] for g in groups}
    e3 = o3["e"]
    for _ in range(B):
        idx = np.concatenate([rows_by_clu[k] for c, ks in clu_by_cty.items() for k in rng.choice(ks, len(ks), replace=True)])
        Xb, yb = X3[idx], y[idx]
        eb = yb - Xb @ np.linalg.lstsq(Xb, yb, rcond=None)[0]
        tot3 = np.sum(eb ** 3)
        for g, m in groups.items():
            mb = m[idx]
            bs[g].append(stats.skew(eb[mb]) if mb.sum() >= 3 else np.nan)
            bs_sh[g].append(np.sum(eb[mb] ** 3) / tot3)
    for g, m in groups.items():
        if m.sum() >= 3:
            out[f"skew_{g}"] = stats.skew(e3[m]); out[f"skew_{g}_lo"] = np.nanpercentile(bs[g], 2.5); out[f"skew_{g}_hi"] = np.nanpercentile(bs[g], 97.5)
            out[f"kelley_{g}"] = qskew(e3[m], 0.10, 0.90)
        out[f"m3share_{g}"] = float(np.sum(e3[m] ** 3) / np.sum(e3 ** 3)); out[f"m3share_{g}_lo"] = np.nanpercentile(bs_sh[g], 2.5)
        out[f"m3share_{g}_hi"] = np.nanpercentile(bs_sh[g], 97.5)
    # influence of the single most extreme residual
    j = int(np.argmax(np.abs(e3)))
    out.update(max_resid_m3share=float(e3[j] ** 3 / np.sum(e3 ** 3)), max_resid_id=f"{main.GCoM_ID[j]} {main.inventory_type[j][:3]} {int(main.inventory_year[j])}")
    keep = np.ones(len(main), bool); keep[j] = False
    out["skew_drop1"] = stats.skew(ols(y[keep], X3[keep])["e"])
    # complete reporters: skewness after removing their ten largest residuals (no re-estimation)
    ec = e3[complete]
    o10 = np.argsort(ec)[-10:]
    epc10 = main.epc.values[complete][o10]
    out.update(skew_complete_drop10=stats.skew(np.delete(ec, o10)), drop10_epc_min=float(epc10.min()), drop10_epc_max=float(epc10.max()),
               drop10_countries="; ".join(f"{c} {n}" for c, n in pd.Series(cty[complete][o10]).value_counts().items()))
    return out


def point12(main):
    """Point estimates of Steps 1-2 without bootstrap: shortfall coefficient and residual skewness of model (3)."""
    main = main.reset_index(drop=True)
    y = main.lnE.values; cl = main.nuts3.values; cty = main.country_code.values; sets = main.sector_set.values; sc = main.sc.values
    complete = ((main.sc >= 5) & (main.core5_complete_flag == 1)).values
    X3, _ = dsg(main, BASE + Q)
    e3 = ols(y, X3)["e"]
    E6 = main[SECTOR_COLS].values.astype(float); six = sc == 6
    share6 = E6[six] / E6[six].sum(1, keepdims=True)
    wm = mech_w_map(sets, cty, share6, cty[six])
    main["w_hat"] = [wm[(s, c)] for s, c in zip(sets, cty)]
    XA, nA = dsg(main, BASE + ["core5_complete_flag", "co2eq", "lca", "mon", "w_hat"])
    bA, VA, _ = ols_V(y, XA, cl); iA = nA.index("w_hat")
    return dict(n=len(main), donors=int(six.sum()), lam=bA[iA], lam_se=np.sqrt(VA[iA, iA]), skew_all=stats.skew(e3),
                skew_complete=stats.skew(e3[complete]), skew_incomplete=stats.skew(e3[~complete]),
                m3share_le2=float(np.sum(e3[sc <= 2] ** 3) / np.sum(e3 ** 3)))


samples = {
    "A": ("2021 dataset", main21),
    "B": ("2021 dataset, per-capita screen", main21s),
    "C": ("7th release", main7),
    "D": ("7th release, not in 2021", main7[main7.new_inventory == True]),
    "E": ("7th release, also in 2021", main7[main7.new_inventory == False]),
    "F": ("7th release, no per-capita screen", main7_raw),
    "G": ("7th release, signatories with 2021 covariates", main7[main7.covariate_source == "jrc2021"]),
    "H": ("7th release, new signatories only", main7[main7.covariate_source == "external"]),
    "I": ("7th release, one BEI and latest MEI per municipality", one_each(main7)),
    "J": ("7th release, earliest of duplicate inventory versions", main7f),
    "K": ("7th release, residential buildings only", main7r),
    "L": ("same 7,011 inventories, 2021 values", same21),
    "M": ("same 7,011 inventories, 7th-release values", same7),
}
res12 = {}
for k, (lab, m) in samples.items():
    o = steps12(m, lab)
    res12[k] = o
    print(f"{k} {lab}: n={o['n']}  coefficient {o['lam']:.3f} [{o['lam_boot_lo']:.2f}, {o['lam_boot_hi']:.2f}]  Wald p {o['wald_p']:.3f}  "
          f"skewness {o['skew_all']:.2f} [{o['skew_all_lo']:.2f}, {o['skew_all_hi']:.2f}]  complete {o['skew_complete']:.2f}  incomplete {o['skew_incomplete']:.2f}", flush=True)
r12 = pd.DataFrame(res12).T

# influence of the inventories removed by the per-capita screen
ge = main7_raw[main7_raw.gross_error == 1]
kr = list(keyed(main7_raw).key)
ids = {r.signatory_name: (str(r.GCoM_ID), r.inventory_type, int(r.inventory_year)) for r in ge.itertuples()}
infl = []
for names in [[n] for n in ids] + [list(c) for c in itertools.combinations(ids, 2)] + [list(ids)]:
    keepk = [(k_ not in ids.values()) or (k_ in [ids[n] for n in names]) for k_ in kr]
    o = point12(main7_raw[np.array(keepk)])
    infl.append(dict(added=" + ".join(names), n=o["n"], skew_all=o["skew_all"], lam=o["lam"]))
infl = pd.DataFrame(infl)
print("7th release, screened sample plus removed inventories:\n" + infl.round(3).to_string(index=False))
same_set = same21.sector_set.values == same7.sector_set.values
ident = dict(n=int(same_set.sum()), skew21=point12(same21[same_set])["skew_all"], skew7=point12(same7[same_set])["skew_all"])
print(f"identical sector sets ({ident['n']}): pooled skewness {ident['skew21']:.2f} (2021 values) and {ident['skew7']:.2f} (7th-release values)")


# ------------------------------------------------------------------ Step 4 (stage 20 Proposition 3)
def decompose(dd, ann=False):
    f = (100 / dd.dy) if ann else 1.0
    T, R_, S_ = dd.dT * f, dd.dR * f, dd.scope * f
    ccm = dd.cat.values == "same coverage"
    mean_cc = T[ccm].mean(); N = len(dd)
    o = dict(mean_all=T.mean(), mean_cc=mean_cc, inflation=T.mean() - mean_cc)
    o["scope_part"] = sum(S_[dd.cat.values == c].sum() for c in CATS) / N
    o["revision_part"] = sum((R_[dd.cat.values == c] - mean_cc).sum() for c in CATS) / N
    return o


def pairs_from(bsel, msel):
    names, cols = SECTOR_NAMES, SECTOR_COLS
    rows = []
    for (_, b), (_, m) in zip(bsel.iterrows(), msel.iterrows()):
        dy = m.inventory_year - b.inventory_year
        if dy <= 0:
            continue
        sbs, sms = b.sector_set, m.sector_set
        keep = [k for k in range(6) if sbs[k] == "1" and sms[k] == "1"]
        rem = [k for k in range(6) if sbs[k] == "1" and sms[k] == "0"]
        add = [k for k in range(6) if sbs[k] == "0" and sms[k] == "1"]
        if not keep:
            continue
        cat = "same coverage" if not rem and not add else ("removal only" if rem and not add else ("addition only" if add and not rem else "both"))
        kb = sum(b[cols[k]] for k in keep); km = sum(m[cols[k]] for k in keep)
        rows.append(dict(id=b.GCoM_ID, cat=cat, dy=dy, dT=m.lnE - b.lnE, dR=np.log(km) - np.log(kb), cty=b.country_code, nuts3=b.nuts3,
                         b_year=b.inventory_year, m_year=m.inventory_year, rem=",".join(names[k] for k in rem), add=",".join(names[k] for k in add)))
    return pd.DataFrame(rows)


def pairs7(m):
    """Latest MEI of each signatory with the BEI of its action plan (fallback: the latest earlier BEI)."""
    mei = m[m.mon == 1].sort_values(["GCoM_ID", "inventory_year", "eid_num"]).groupby("GCoM_ID").tail(1)
    bei = m[m.mon == 0]
    bsel, msel = [], []
    for _, r in mei.iterrows():
        cand = bei[(bei.GCoM_ID == r.GCoM_ID) & (bei.inventory_year < r.inventory_year)]
        if cand.empty:
            continue
        c1 = cand[cand.inventory_year == r.plan_bei_year]
        bsel.append(c1.iloc[-1] if len(c1) else cand.sort_values("inventory_year").iloc[-1]); msel.append(r)
    return pairs_from(pd.DataFrame(bsel), pd.DataFrame(msel))


def step4(d, label):
    out = dict(sample=label, pairs=len(d), **{f"pairs_{c.replace(' ', '_')}": int((d.cat == c).sum()) for c in ["same coverage"] + CATS})
    d = d.copy().reset_index(drop=True); d["scope"] = d.dT - d.dR
    res = {"log": decompose(d), "annual": decompose(d, ann=True)}
    rngp = np.random.default_rng(SEED + 2)
    pc_by_cty = {c: np.unique(d.nuts3[d.cty == c]) for c in d.cty.unique()}
    prow = {k: np.where(d.nuts3.values == k)[0] for k in d.nuts3.unique()}
    bsd = {"log": [], "annual": []}
    for _ in range(B):
        idx = np.concatenate([prow[k] for c, ks in pc_by_cty.items() for k in rngp.choice(ks, len(ks), replace=True)])
        db = d.iloc[idx]
        bsd["log"].append(decompose(db)); bsd["annual"].append(decompose(db, ann=True))
    for k_ in ("log", "annual"):
        bdf = pd.DataFrame(bsd[k_])
        for item in ["mean_all", "mean_cc", "inflation", "scope_part", "revision_part"]:
            out[f"{k_}_{item}"] = res[k_][item]
            out[f"{k_}_{item}_lo"] = bdf[item].quantile(0.025); out[f"{k_}_{item}_hi"] = bdf[item].quantile(0.975)
    dn = d[d.nuts3.astype(str).str[:4] != "ES61"]
    if len(dn) < len(d) and (dn.cat == "same coverage").sum() >= 10:
        r = decompose(dn, ann=True)
        out.update(annual_mean_cc_noES61=r["mean_cc"], annual_revision_part_noES61=r["revision_part"])
    return out


d_pub, _ = build_pairs(main21)
p21 = d_pub[d_pub.dR.notna()].copy()
p21["b_year"] = p21.id.map(main21[main21.mon == 0].set_index("GCoM_ID").inventory_year)
p21["m_year"] = p21.id.map(main21[main21.mon == 1].set_index("GCoM_ID").inventory_year)
p7 = pairs7(main7)
key21 = set(zip(p21.id, p21.b_year, p21.m_year))
p7["same_as_2021"] = [k in key21 for k in zip(p7.id, p7.b_year, p7.m_year)]
res4 = {"A": step4(p21, "2021 dataset"), "C": step4(p7, "7th release, all pairs"),
        "D": step4(p7[~p7.same_as_2021], "7th release, pairs not in 2021"), "E": step4(p7[p7.same_as_2021], "7th release, pairs also in 2021")}
r4 = pd.DataFrame(res4).T
print(r4[["sample", "pairs", "pairs_same_coverage", "pairs_addition_only", "pairs_removal_only", "pairs_both", "annual_mean_cc", "annual_inflation",
          "annual_scope_part", "annual_revision_part"]].round(2).to_string())
# how 2021 pairs in Andalucia appear in the 7th release (same municipality, BEI year and MEI year); waste additions in the 7th release
es = p21[p21.nuts3.astype(str).str[:4] == "ES61"]
k7 = p7.set_index(["id", "b_year", "m_year"])
trans = {}
for c in ["removal only", "same coverage"]:
    e_ = es[es.cat == c]
    found = [k for k in zip(e_.id, e_.b_year, e_.m_year) if k in k7.index]
    trans[c] = dict(n2021=len(e_), found=len(found), cats=k7.loc[found, "cat"].value_counts().to_dict() if found else {})
wa = p7[p7["add"].fillna("").str.contains("waste")]
waste_add = dict(n=len(wa), es61=int((wa.nuts3.astype(str).str[:4] == "ES61").sum()), es51=int((wa.nuts3.astype(str).str[:4] == "ES51").sum()))
print(f"2021 Andalucia pairs in the 7th release: {trans}; 7th-release changed-coverage pairs adding waste: {waste_add}")


# ------------------------------------------------------------------ joint test of the sector-count indicators given w_hat (Table S26)
def recompute(m):
    """Totals, sector counts and sets after changing sector values."""
    m = m.copy()
    E = m[SECTOR_COLS]
    m["total_emissions_reported_tonnes"] = E.sum(axis=1, min_count=1)
    m["reported_sector_count"] = E.notna().sum(axis=1)
    m["sc"] = m.reported_sector_count
    for s in [1, 2, 3, 4, 6]:
        m[f"sc{s}"] = (m.sc == s).astype(int)
    m["core5_complete_flag"] = (E.iloc[:, :5].notna().sum(axis=1) == 5).astype(int)
    m["sector_set"] = E.notna().apply(lambda r: "".join("1" if v else "0" for v in r), axis=1)
    m["lnE"] = np.log(m.total_emissions_reported_tonnes)
    return m


def wald(m, donors=None, split6=None):
    """Joint Wald test of the 1-4 and 6 sector indicators in model (3) with w_hat added; donors: sample whose six-sector inventories
    supply the shares (default: m itself); split6: boolean array splitting the six-sector indicator into two."""
    m = m.reset_index(drop=True).copy()
    y = m.lnE.values; cl = m.nuts3.values; cty = m.country_code.values; sets = m.sector_set.values
    D = m if donors is None else donors.reset_index(drop=True)
    six = D.sc.values == 6
    sh = D[SECTOR_COLS].values[six].astype(float); sh = sh / sh.sum(1, keepdims=True)
    wm = mech_w_map(sets, cty, sh, D.country_code.values[six])
    m["w_hat"] = [wm[(s, c)] for s, c in zip(sets, cty)]
    q = list(Q)
    if split6 is not None:
        m["sc6_a"] = (split6 & (m.sc.values == 6)).astype(int); m["sc6_b"] = (~split6 & (m.sc.values == 6)).astype(int)
        q = ["sc1", "sc2", "sc3", "sc4", "sc6_a", "sc6_b", "core5_complete_flag", "co2eq", "lca", "mon"]
    XB, nB = dsg(m, BASE + q + ["w_hat"])
    bB, VB, _ = ols_V(y, XB, cl)
    cs = [c for c in ["sc1", "sc2", "sc3", "sc4", "sc6"] if c in nB]
    ix = [nB.index(c) for c in cs]
    W = float(bB[ix] @ np.linalg.solve(VB[np.ix_(ix, ix)], bB[ix]))
    X3, _ = dsg(m, BASE + Q)
    out = dict(n=len(m), n6=int((m.sc == 6).sum()), W=W, p=1 - stats.chi2.cdf(W, len(ix)), sd3=float(ols(y, X3)["e"].std()))
    for c in ["sc1", "sc4", "sc6", "sc6_a", "sc6_b"]:
        if c in nB:
            j = nB.index(c); out[c], out[c + "_se"] = bB[j], np.sqrt(VB[j, j])
    return out


wdrop = (same21.waste_emissions_tonnes.notna() & same7.waste_emissions_tonnes.isna()).values
a1 = same21.copy(); a1.loc[wdrop, "waste_emissions_tonnes"] = np.nan; a1 = recompute(a1)
b1 = same7.copy(); b1.loc[wdrop, "waste_emissions_tonnes"] = same21.loc[wdrop, "waste_emissions_tonnes"].values; b1 = recompute(b1)
r21 = main21.nuts3.astype(str).str[:4]; r7 = main7.nuts3.astype(str).str[:4]
W = [("A", "2021 dataset", wald(main21)), ("A", "7th release", wald(main7)),
     ("A", "2021 dataset without Andaluc\\'ia", wald(main21[(r21 != "ES61").values])),
     ("A", "7th release without Andaluc\\'ia", wald(main7[(r7 != "ES61").values])),
     ("B", "2021 values", wald(same21)), ("B", "7th-release values", wald(same7)),
     ("B", "2021 values, waste omitted as in the 7th release", wald(a1)), ("B", "7th-release values, 2021 waste restored", wald(b1)),
     ("B", "2021 values, 7th-release donors", wald(same21, donors=same7)), ("B", "7th-release values, 2021 donors", wald(same7, donors=same21)),
     ("B", "Identical sector sets, 2021 values", wald(same21[same_set])), ("B", "Identical sector sets, 7th-release values", wald(same7[same_set]))]
wtab = pd.DataFrame([dict(panel=p, label=l, **o) for p, l, o in W])
drop6 = ((same21.sc == 6).values & wdrop)
sp = wald(same21, split6=drop6)
ws = (same21.waste_emissions_tonnes / same21.total_emissions_reported_tonnes).values
keep6 = (same21.sc == 6).values & ~wdrop
reg_same = same21.nuts3.astype(str).str[:4].values
mech = dict(n_wdrop=int(wdrop.sum()), wdrop_es61=int((wdrop & (reg_same == "ES61")).sum()), wdrop_es51=int((wdrop & (reg_same == "ES51")).sum()),
            drop_n=int(drop6.sum()), drop_es61=int((drop6 & (reg_same == "ES61")).sum()), keep_n=int(keep6.sum()),
            drop_coef=sp["sc6_a"], drop_se=sp["sc6_a_se"], keep_coef=sp["sc6_b"], keep_se=sp["sc6_b_se"],
            waste_share_drop=float(np.mean(ws[drop6])), waste_share_keep=float(np.mean(ws[keep6])), n_same=int(same_set.sum()))
print(wtab[["panel", "label", "n", "n6", "W", "p", "sc6", "sd3"]].round(3).to_string(index=False))
print("mechanism:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in mech.items()})


# ------------------------------------------------------------------ Proposition 3 between releases (Table S27)
EA = same21[SECTOR_COLS].values.astype(float); EB = same7[SECTOR_COLS].values.astype(float)
pa, pb = ~np.isnan(EA), ~np.isnan(EB)
bothp = pa & pb
RA = np.where(bothp, EA, 0).sum(1); RB = np.where(bothp, EB, 0).sum(1)
dT = np.log(same7.total_emissions_reported_tonnes.values) - np.log(same21.total_emissions_reported_tonnes.values)
nocommon = ~bothp.any(1)          # no sector reported in both releases: retained-sector term undefined
dR = np.where(nocommon, np.nan, np.log(np.where(RB > 0, RB, np.nan)) - np.log(np.where(RA > 0, RA, np.nan)))
scope = dT - dR
rem = (pa & ~pb).any(1); add = (pb & ~pa).any(1)
rcat = np.where(~rem & ~add, "same coverage", np.where(rem & ~add, "removal only", np.where(add & ~rem, "addition only", "both")))
rreg = np.where(reg_same == "ES61", "Andalucia", np.where(reg_same == "ES51", "Catalonia", "other"))
rd = pd.DataFrame(dict(cat=rcat, region=rreg, dT=dT, dR=dR, scope=scope,
                       rem=[",".join(SECTOR_NAMES[k] for k in range(6) if pa[i, k] and not pb[i, k]) for i in range(len(rcat))],
                       add=[",".join(SECTOR_NAMES[k] for k in range(6) if pb[i, k] and not pa[i, k]) for i in range(len(rcat))]))


rd["changed"] = rd.cat != "same coverage"


def rsumm(g):
    return pd.Series(dict(n=len(g), mean_dT=g.dT.mean(), mean_scope=g.scope.mean(), mean_dR=g.dR.mean(),
                          ident=(g.dT.abs() < 1e-4).mean(), big=(g.dT.abs() > 0.10).mean(), changed=g.changed.mean()))


order = ["same coverage", "removal only", "addition only", "both"]
tcat = pd.concat([rd.groupby("cat").apply(rsumm).reindex(order), rsumm(rd).to_frame("all").T])
treg = pd.concat([rd.groupby("region").apply(rsumm).reindex(["Andalucia", "Catalonia", "other"]), rsumm(rd).to_frame("all").T])
cv = np.cov(rd.scope, rd.dR)
bothE = bothp[:, :5]
RAe = np.where(bothE, EA[:, :5], 0).sum(1); RBe = np.where(bothE, EB[:, :5], 0).sum(1)
ok = (RAe > 0) & (RBe > 0)
dRe = np.log(RBe[ok]) - np.log(RAe[ok])
wb_ = bothp[:, 5]
rw = EB[wb_, 5] / EA[wb_, 5]
rel = dict(var_dT=float(np.var(dT, ddof=1)), var_scope=float(cv[0, 0]), var_dR=float(cv[1, 1]), cov2=float(2 * cv[0, 1]),
           energy_identical=float(np.mean(np.abs(dRe) < 1e-4)), energy_big=float(np.mean(np.abs(dRe) > 0.10)),
           waste_both=int(wb_.sum()), waste_0885=float(np.mean(np.abs(rw - 0.885) < 0.001)),
           removed_any=int((rd.rem != "").sum()), removed_waste_only=int((rd.rem == "waste").sum()),
           added_any=int((rd["add"] != "").sum()), added_manufacturing_only=int((rd["add"] == "manufacturing").sum()),
           added_residential_only=int((rd["add"] == "residential").sum()))
print(tcat.round(3).to_string()); print(treg.round(3).to_string()); print({k: round(v, 4) for k, v in rel.items()})
print(f"same inventories without a sector reported in both releases: {int(nocommon.sum())}")
print("removed sectors:", rd.loc[rd.rem != "", "rem"].value_counts().head(5).to_dict(), "| added sectors:", rd.loc[rd["add"] != "", "add"].value_counts().head(5).to_dict())

save_obj("mycovenant7.pkl", dict(steps12=r12, step4=r4, pairs7=p7, influence=infl, identical_sets=ident, transitions=trans, waste_add=waste_add,
                                 wald=wtab, mechanism=mech, release_by_category=tcat, release_by_region=treg, release=rel))


# ------------------------------------------------------------------ LaTeX for Tables S23, S25, S26 and S27
def n_(x, d=2):
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "$-$")


def ci(x, c, d=2):
    return f"{n_(x[c], d)} [{n_(x[c + '_lo'], d)}, {n_(x[c + '_hi'], d)}]"


def num(v):
    return f"{int(v):,}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


R, S4 = r12, r4
K = ["A", "B", "C", "D"]
K4 = ["A", "C", "D", "E"]


def row(lab, f, keys=K, src=R):
    return lab + " & " + " & ".join(f(src.loc[k]) for k in keys) + r" \\"


T = [r"\begin{table}[width=\linewidth,pos=htbp]",
     r"\caption{Screening tests in the 2021 dataset and in the MyCovenant 7th release (January 2026)}\label{tab:S23}",
     r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular*}{\tblwidth}{@{}LRRRR@{}}", r"\toprule",
     r"& 2021 dataset & 2021, same & 7th release & 7th release, \\", r"& (main analysis) & screen & & not in 2021 \\", r"\midrule",
     r"\multicolumn{5}{@{}l}{\textit{Panel A. Sample}}\\",
     row("Inventories (municipalities)", lambda x: f"{num(x.n)} ({num(x.n_muni)})"),
     row("BEI / MEI", lambda x: f"{num(x.n_bei)} / {num(x.n_mei)}"),
     row("Median inventory year", lambda x: f"{int(x.median_year)}"),
     row("One- and two-sector inventories", lambda x: f"{int(x.n_sc1 + x.n_sc2)}"),
     row("Six-sector donors (Andaluc\\'ia)", lambda x: f"{int(x.n_donors)} ({int(x.donors_es61)})"),
     r"\midrule", r"\multicolumn{5}{@{}l}{\textit{Panel B. Step 1: levels (Proposition~1)}}\\",
     row("Coefficient on $\\hat w_i$", lambda x: n_(x.lam, 3)),
     row("\\quad clustered SE", lambda x: n_(x.lam_se, 3)),
     row("\\quad two-stage bootstrap 95\\%", lambda x: f"[{n_(x.lam_boot_lo)}, {n_(x.lam_boot_hi)}]"),
     row("Wald test of counts given $\\hat w_i$ ($p$)", lambda x: f"{x.wald_p:.3f}"),
     row("Sector-set slope (sets)", lambda x: f"{n_(x.sets_slope)} ({int(x.sets_n)})"),
     r"\midrule", r"\multicolumn{5}{@{}l}{\textit{Panel C. Step 2: residual asymmetry (Proposition~2), model (3)}}\\",
     row("Skewness, all", lambda x: ci(x, "skew_all")),
     row("\\quad core-five-complete", lambda x: ci(x, "skew_complete")),
     row("\\quad incomplete", lambda x: ci(x, "skew_incomplete")),
     row("Third-moment share, 1--2 sectors", lambda x: (f"{x.m3share_le2 * 100:.0f}\\% [{x.m3share_le2_lo * 100:.0f}, {x.m3share_le2_hi * 100:.0f}]" if x.skew_all_hi < 0 else "n.r.")),
     row("Largest residual: share of third moment", lambda x: f"{x.max_resid_m3share * 100:.0f}\\%"),
     row("Skewness without that inventory", lambda x: n_(x.skew_drop1)),
     row("Quantile skewness (10/90), all", lambda x: n_(x.kelley_all, 3)),
     r"\midrule", r"\multicolumn{5}{@{}l}{\textit{Panel D. Step 4: changes (Proposition~3), \% per year}}\\",
     r"& 2021 dataset & 7th release, & 7th release, & 7th release, \\", r"& & all pairs & not in 2021 & also in 2021 \\",
     row("Pairs (changed coverage)", lambda x: f"{num(x.pairs)} ({num(x.pairs - x.pairs_same_coverage)})", K4, S4),
     row("Constant-coverage mean", lambda x: ci(x, "annual_mean_cc"), K4, S4),
     row("All pairs minus constant coverage", lambda x: ci(x, "annual_inflation"), K4, S4),
     row("\\quad scope term", lambda x: ci(x, "annual_scope_part"), K4, S4),
     row("\\quad retained-sector term", lambda x: ci(x, "annual_revision_part"), K4, S4),
     row("Without ES61: constant coverage", lambda x: n_(x.annual_mean_cc_noES61), K4, S4),
     row("Without ES61: retained-sector term", lambda x: n_(x.annual_revision_part_noES61), K4, S4),
     r"\bottomrule", r"\end{tabular*}",
     r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize Notes: same code, seeds and replication counts as Tables~2, S14, S16 and S17; donor shares are re-estimated from the six-sector inventories of each sample (country pools with at least 20 donors, otherwise pooled). "
     f"2021, same screen: inventories outside 0.01--100~t per capita removed ({len(main21) - len(main21s)} inventories), as in both 7th-release columns ({len(main7_raw) - len(main7)} removed). "
     r"Not in 2021: no 2021 inventory with the same signatory, type and year. Intervals: 999 replications resampling NUTS3 regions within countries (499 for the two-stage bootstrap, which also resamples donors). n.r.: not reported because the pooled third-moment interval includes zero. ES61: Andaluc\'ia. Data processing: Section~S17 and Table~S24.\end{minipage}",
     r"\end{table}"]
lab25 = {"F": "No gross-error screen", "G": "Signatories with 2021 covariates", "H": "New signatories only (external covariates)",
         "I": "One BEI and latest MEI per municipality", "J": "Earliest of duplicate inventory versions", "K": "Residential buildings only"}


def srow(nm, x):
    return (f"{nm} & {num(x.n)} & {num(x.n_donors)} & {n_(x.lam)} [{n_(x.lam_boot_lo)}, {n_(x.lam_boot_hi)}] & {pval(x.wald_p)} & "
            f"{ci(x, 'skew_all')} & {n_(x.skew_complete)} & {n_(x.skew_incomplete)} \\\\")


T += ["", r"\begin{table}[width=\linewidth,pos=htbp]",
      r"\caption{Same-inventory comparison and sensitivity of the 7th-release replication}\label{tab:S25}",
      r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular*}{\tblwidth}{@{}LRRRRRRR@{}}", r"\toprule",
      r"& Inven- & Six-sector & Coef. on $\hat w_i$ & Wald & Skewness, all & Com- & Incom- \\", r"& tories & donors & & $p$ & & plete & plete \\",
      r"\midrule", r"\multicolumn{8}{@{}l}{\textit{Panel A. The same inventories in both analysis samples}}\\",
      srow("2021 values", R.loc["L"]), srow("7th-release values", R.loc["M"]),
      r"\midrule", r"\multicolumn{8}{@{}l}{\textit{Panel B. Processing choices, 7th release}}\\", srow("Main specification (Table~S23)", R.loc["C"])]
T += [srow(lab25[k], R.loc[k]) for k in "FGHIJK"]
T += [r"\bottomrule", r"\end{tabular*}",
      r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize Notes: intervals as in Table~S23; Wald: test of the sector-count indicators given $\hat w_i$. "
      f"Panel A: the {len(common):,} inventories with the same signatory, type and year in both analysis samples, after the same per-capita screen; donor shares are re-estimated from each release's six-sector inventories, which are fewer in the 7th release because {mech['n_wdrop']} of these inventories lack the waste reported in 2021. "
      r"Residential buildings only: agriculture and non-allocated buildings excluded instead of assigned to the residential sector as in the 2021 dataset (Table~S24).\end{minipage}",
      r"\end{table}"]


def c_(x, k):
    return f"{n_(x[k])} ({x[k + '_se']:.2f})"


T += ["", r"\begin{table}[width=\linewidth,pos=htbp]",
      r"\caption{Joint test of the sector-count indicators given $\hat w_i$ in the two releases}\label{tab:S26}",
      r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular*}{\tblwidth}{@{}LRRRRRRRR@{}}", r"\toprule",
      r"& Inven- & Six- & Wald & & \multicolumn{3}{c}{Contrast given $\hat w_i$ (SE)} & Residual \\",
      r"& tories & sector & (5 df) & $p$ & Six sectors & Four sectors & One sector & SD \\", r"\midrule",
      r"\multicolumn{9}{@{}l}{\textit{Panel A. Analysis samples}}\\"]
for r in wtab.itertuples():
    if r.panel == "B" and r.label == "2021 values":
        T += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{Panel B. The same {len(common):,} inventories; one data feature changed at a time}}}}\\"]
    x = wtab.loc[r.Index]
    T.append(f"{r.label} & {num(r.n)} & {num(r.n6)} & {r.W:.1f} & {pval(r.p)} & {c_(x, 'sc6')} & {c_(x, 'sc4')} & {c_(x, 'sc1')} & {r.sd3:.2f} \\\\")
T += [r"\bottomrule", r"\end{tabular*}",
      r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize Notes: contrasts are the coefficients on the sector-count indicators in model (3) of Table~2 with $\hat w_i$ added (reference: five sectors, with the core-five indicator; NUTS3-clustered SE); Wald tests the one- to four- and six-sector indicators jointly. Residual SD: model (3) residuals. The six-sector column counts such inventories in the analysis sample; rows swapping donor pools use the other release's donors. Donor shares are otherwise re-estimated within each sample. "
      f"Waste omitted as in the 7th release: waste is removed from the {mech['n_wdrop']} inventories that report it in 2021 but not in the 7th release ({mech['wdrop_es61']} in Andaluc\\'ia, {mech['wdrop_es51']} in Catalonia), and totals, sector sets and donors are recomputed; 2021 waste restored does the reverse. "
      f"With 2021 values, the six-sector contrast is {n_(mech['drop_coef'])} ({mech['drop_se']:.2f}) for the {mech['drop_n']} six-sector inventories that lack waste in the 7th release ({mech['drop_es61']} in Andaluc\\'ia) and {n_(mech['keep_coef'])} ({mech['keep_se']:.2f}) for the other {mech['keep_n']}; their mean waste shares are {mech['waste_share_drop'] * 100:.1f}\\% and {mech['waste_share_keep'] * 100:.1f}\\%. "
      f"Identical sector sets: the {mech['n_same']:,} inventories with the same sector set in both releases.\\end{{minipage}}",
      r"\end{table}"]


def pc_(x):
    return f"{x * 100:.1f}\\%"


lab27 = {"same coverage": "Same sectors", "removal only": "Sectors removed", "addition only": "Sectors added", "both": "Removed and added", "all": "All"}
T += ["", r"\begin{table}[width=\linewidth,pos=htbp]",
      rf"\caption{{Differences between the 2021 dataset and the 7th release for the same {len(common):,} inventories, decomposed by Proposition~3}}\label{{tab:S27}}",
      r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\par\medskip\noindent\textit{Panel A. By coverage between releases}\par\smallskip",
      r"\begin{tabular*}{\tblwidth}{@{}LRRRRRR@{}}", r"\toprule",
      r"Coverage in the 7th release & Inventories & Mean difference & Scope term & Retained-sector & Identical & $|$difference$|$ \\",
      r"& & (ln points) & & term & totals & $>$0.10 \\", r"\midrule"]
for k in order + ["all"]:
    x = tcat.loc[k]
    if k == "all":
        T.append(r"\midrule")
    T.append(f"{lab27[k]} & {num(x.n)} & {n_(x.mean_dT, 3)} & {n_(x.mean_scope, 3)} & {n_(x.mean_dR, 3)} & {pc_(x.ident)} & {pc_(x.big)} \\\\")
T += [r"\bottomrule", r"\end{tabular*}", r"\par\medskip\noindent\textit{Panel B. By region}\par\smallskip",
      r"\begin{tabular*}{\tblwidth}{@{}LRRRRRR@{}}", r"\toprule",
      r"Region & Inventories & Coverage & Mean difference & Scope term & Retained-sector & Identical \\",
      r"& & changed & (ln points) & & term & totals \\", r"\midrule"]
rl = {"Andalucia": "Andaluc\\'ia (ES61)", "Catalonia": "Catalonia (ES51)", "other": "Other regions", "all": "All"}
for k in ["Andalucia", "Catalonia", "other", "all"]:
    x = treg.loc[k]
    if k == "all":
        T.append(r"\midrule")
    T.append(f"{rl[k]} & {num(x.n)} & {pc_(x.changed)} & {n_(x.mean_dT, 3)} & {n_(x.mean_scope, 3)} & {n_(x.mean_dR, 3)} & {pc_(x.ident)} \\\\")
T += [r"\bottomrule", r"\end{tabular*}",
      r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize Notes: for each inventory, the difference $\ln E^{7\mathrm{th}}-\ln E^{2021}$ is split exactly into a retained-sector term (log change of the sectors reported in both releases) and a scope term (the remainder, from sectors reported in only one release), as in Proposition~3 with releases in place of inventory years. "
      f"Waste is the only removed sector in {rel['removed_waste_only']:,} of the {rel['removed_any']:,} inventories that lose a sector; manufacturing ({rel['added_manufacturing_only']}) and residential ({rel['added_residential_only']}) are the most common single additions among the {rel['added_any']} that gain one. "
      f"Among energy sectors reported in both releases, {rel['energy_identical'] * 100:.1f}\\% of inventories have identical values and {rel['energy_big'] * 100:.1f}\\% differ by more than 0.10 ln points; "
      f"of the {rel['waste_both']} inventories with waste in both releases, {rel['waste_0885'] * 100:.1f}\\% have a 7th-release value 0.885 times the 2021 value. "
      f"Variance of the difference: scope term {rel['var_scope']:.4f}, retained-sector term {rel['var_dR']:.4f}, twice their covariance {rel['cov2']:.4f}. "
      f"Largest single-inventory share of the pooled residual third moment: {R.loc['A', 'max_resid_m3share'] * 100:.0f}\\% in the 2021 dataset (Table~S23), "
      f"{R.loc['F', 'max_resid_m3share'] * 100:.0f}\\% in the 7th release before the per-capita screen and {R.loc['C', 'max_resid_m3share'] * 100:.0f}\\% after it.\\end{{minipage}}",
      r"\end{table}"]
with open(os.path.join(TAB_DIR, "Tables_S23_S25_S26_S27.tex"), "w", encoding="utf-8") as f:
    f.write("% Tables S23, S25, S26 and S27 generated by 24_mycovenant7_replication.py (cas-sc LaTeX)\n" + "\n".join(T) + "\n")


# ------------------------------------------------------------------ numbers for Section S17 and check against the manuscript
A_, Bs, C_, D_, E_, F_ = (R.loc[k] for k in "ABCDEF")
chk = []


def check(where, item, printed, value, digits):
    """Compare a computed value, rounded as printed in the manuscript, with the printed number."""
    v = float(value)
    chk.append(dict(where=where, item=item, manuscript=printed, this_run=round(v, digits), status="OK" if round(v, digits) == printed else "DIFF"))


check("S17 Step 1", "7th release: six-sector donors", 489, C_.n_donors, 0)
check("S17 Step 1", "7th release: donors in Andalucia", 122, C_.donors_es61, 0)
check("S17 Step 1", "not in 2021: donors", 182, D_.n_donors, 0)
check("S17 Step 1", "7th release: shortfall coefficient", -1.11, C_.lam, 2)
check("S17 Step 1", "7th release: two-stage interval, lower", -1.29, C_.lam_boot_lo, 2)
check("S17 Step 1", "7th release: two-stage interval, upper", -0.89, C_.lam_boot_hi, 2)
check("S17 Step 1", "7th release: clustered SE", 0.105, C_.lam_se, 3)
check("S17 Step 1", "7th release: t against -1", -1.08, C_.lam_t_m1, 2)
check("S17 Step 1", "7th release: sector-set slope", 1.18, C_.sets_slope, 2)
check("S17 Step 1", "same inventories, 2021 values: coefficient", -1.03, R.loc["L", "lam"], 2)
check("S17 Step 1", "same inventories, 7th values: coefficient", -0.95, R.loc["M", "lam"], 2)
check("S17 Step 1", "not in 2021: coefficient", -1.34, D_.lam, 2)
check("S17 Step 1", "not in 2021: clustered SE", 0.090, D_.lam_se, 3)
check("S17 Step 1", "not in 2021: t against -1", -3.8, D_.lam_t_m1, 1)
check("S17 Step 1", "not in 2021: sector sets", 9, D_.sets_n, 0)
check("S17 Step 1", "not in 2021: sector-set slope", -0.06, D_.sets_slope, 2)
check("S17 Step 1", "not in 2021: identifying variation from 1-2 sectors, %", 78, D_.leverage_1_2 * 100, 0)
check("S17 joint test", "2021: Wald", 18.2, W[0][2]["W"], 1)
check("S17 joint test", "7th release: Wald", 6.4, W[1][2]["W"], 1)
check("S17 joint test", "7th release: p", 0.266, W[1][2]["p"], 3)
check("S17 joint test", "same inventories, 2021 values: Wald", 23.5, W[4][2]["W"], 1)
check("S17 joint test", "same inventories, 7th values: Wald", 7.8, W[5][2]["W"], 1)
check("S17 joint test", "same inventories, 7th values: p", 0.170, W[5][2]["p"], 3)
check("S17 joint test", "residual SD, 2021 values", 0.428, W[4][2]["sd3"], 3)
check("S17 joint test", "residual SD, 7th values", 0.429, W[5][2]["sd3"], 3)
check("S17 joint test", "six-sector inventories losing waste", 481, mech["drop_n"], 0)
check("S17 joint test", "of which in Andalucia", 401, mech["drop_es61"], 0)
check("S17 joint test", "their six-sector contrast", 0.37, mech["drop_coef"], 2)
check("S17 joint test", "other six-sector inventories", 293, mech["keep_n"], 0)
check("S17 joint test", "their six-sector contrast", 0.09, mech["keep_coef"], 2)
check("S17 joint test", "mean waste share, losing waste, %", 5.2, mech["waste_share_drop"] * 100, 1)
check("S17 joint test", "mean waste share, others, %", 7.2, mech["waste_share_keep"] * 100, 1)
check("S17 joint test", "waste omitted as in 7th: Wald", 12.7, W[6][2]["W"], 1)
check("S17 joint test", "waste omitted as in 7th: p", 0.026, W[6][2]["p"], 3)
check("S17 joint test", "waste omitted: six-sector contrast", -0.06, W[6][2]["sc6"], 2)
check("S17 joint test", "2021 waste restored: Wald", 15.5, W[7][2]["W"], 1)
check("S17 joint test", "2021 waste restored: p", 0.008, W[7][2]["p"], 3)
check("S17 joint test", "2021 waste restored: six-sector contrast", 0.19, W[7][2]["sc6"], 2)
check("S17 joint test", "7th values: six-sector contrast", -0.02, W[5][2]["sc6"], 2)
check("S17 joint test", "2021 values: six-sector contrast", 0.24, W[4][2]["sc6"], 2)
check("S17 joint test", "donor swap: Wald (2021 values)", 21.0, W[8][2]["W"], 1)
check("S17 joint test", "donor swap: Wald (7th values)", 4.9, W[9][2]["W"], 1)
check("S17 joint test", "identical sector sets: Wald (2021 values)", 12.2, W[10][2]["W"], 1)
check("S17 joint test", "identical sector sets: Wald (7th values)", 7.3, W[11][2]["W"], 1)
check("S17 joint test", "2021 without Andalucia: six-sector contrast", 0.08, W[2][2]["sc6"], 2)
check("S17 joint test", "2021 without Andalucia: p", 0.033, W[2][2]["p"], 3)
check("S17 Step 2", "7th release: skewness", -0.67, C_.skew_all, 2)
check("S17 Step 2", "7th release: skewness interval, lower", -1.17, C_.skew_all_lo, 2)
check("S17 Step 2", "7th release: skewness interval, upper", -0.22, C_.skew_all_hi, 2)
check("S17 Step 2", "7th release: core-complete skewness", -0.17, C_.skew_complete, 2)
check("S17 Step 2", "7th release: incomplete skewness", -0.98, C_.skew_incomplete, 2)
check("S17 Step 2", "7th release: 1-2-sector third-moment share, %", 66, C_.m3share_le2 * 100, 0)
check("S17 Step 2", "same inventories, 2021 values: skewness", -1.07, R.loc["L", "skew_all"], 2)
check("S17 Step 2", "same inventories, 7th values: skewness", -0.95, R.loc["M", "skew_all"], 2)
check("S17 Step 2", "same inventories, 2021 values: core-complete", -0.16, R.loc["L", "skew_complete"], 2)
check("S17 Step 2", "same inventories, 7th values: core-complete", -0.33, R.loc["M", "skew_complete"], 2)
check("S17 Step 2", "same inventories, 2021 values: incomplete", -1.52, R.loc["L", "skew_incomplete"], 2)
check("S17 Step 2", "same inventories, 7th values: incomplete", -1.25, R.loc["M", "skew_incomplete"], 2)
check("S17 Step 2", "same inventories, 2021 values: 1-2-sector share, %", 90, R.loc["L", "m3share_le2"] * 100, 0)
check("S17 Step 2", "same inventories, 7th values: 1-2-sector share, %", 83, R.loc["M", "m3share_le2"] * 100, 0)
check("S17 Step 2", "identical sector sets, 2021 values: skewness", -0.85, ident["skew21"], 2)
check("S17 Step 2", "identical sector sets, 7th values: skewness", -0.97, ident["skew7"], 2)
check("S17 Step 2", "2021: largest residual's third-moment share, %", 42, A_.max_resid_m3share * 100, 0)
check("S17 Step 2", "2021: skewness without it", -1.31, A_.skew_drop1, 2)
check("S17 Step 2", "2021 with the screen: skewness", -1.27, Bs.skew_all, 2)
check("S17 Step 2", "2021 with the screen: 1-2-sector share, %", 78, Bs.m3share_le2 * 100, 0)
check("S17 Step 2", "not in 2021: skewness", -0.14, D_.skew_all, 2)
check("S17 Step 2", "not in 2021: core-complete skewness", 0.56, D_.skew_complete, 2)
check("S17 Step 2", "not in 2021: core-complete without its ten largest residuals", 0.07, D_.skew_complete_drop10, 2)
check("S17 Step 2", "those ten: lowest t per capita", 18.3, D_.drop10_epc_min, 1)
check("S17 Step 2", "those ten: highest t per capita", 67.7, D_.drop10_epc_max, 1)
check("S17 Step 2", "not in 2021: core-complete quantile skewness", -0.05, D_.kelley_complete, 2)
check("S17 screen", "screened sample plus Lingewaard and Orebro: skewness", 0.40,
      infl.loc[infl.added.str.contains("Lingewaard") & infl.added.str.contains("rebro") & ~infl.added.str.contains("Omobono"), "skew_all"].iloc[0], 2)
check("S17 screen", "screened sample plus Lingewaard: skewness", 0.09, infl.loc[infl.added == "Lingewaard", "skew_all"].iloc[0], 2)
check("S17 screen", "all three restored: skewness", 0.44, F_.skew_all, 2)
check("S17 Step 4", "7th release: pairs", 2634, S4.loc["C", "pairs"], 0)
check("S17 Step 4", "7th release: changed coverage", 858, S4.loc["C", "pairs"] - S4.loc["C", "pairs_same_coverage"], 0)
check("S17 Step 4", "addition only", 524, S4.loc["C", "pairs_addition_only"], 0)
check("S17 Step 4", "removal only", 130, S4.loc["C", "pairs_removal_only"], 0)
check("S17 Step 4", "both", 204, S4.loc["C", "pairs_both"], 0)
check("S17 Step 4", "7th release: constant-coverage mean", -2.15, S4.loc["C", "annual_mean_cc"], 2)
check("S17 Step 4", "2021: constant-coverage mean", -2.24, S4.loc["A", "annual_mean_cc"], 2)
check("S17 Step 4", "7th release: all pairs minus constant coverage", -0.60, S4.loc["C", "annual_inflation"], 2)
check("S17 Step 4", "7th release: scope term", -0.16, S4.loc["C", "annual_scope_part"], 2)
check("S17 Step 4", "7th release: retained-sector term", -0.44, S4.loc["C", "annual_revision_part"], 2)
check("S17 Step 4", "7th without Andalucia: constant coverage", -1.69, S4.loc["C", "annual_mean_cc_noES61"], 2)
check("S17 Step 4", "7th without Andalucia: retained-sector term", -0.16, S4.loc["C", "annual_revision_part_noES61"], 2)
check("S17 Step 4", "2021 without Andalucia: constant coverage", -1.61, S4.loc["A", "annual_mean_cc_noES61"], 2)
check("S17 Step 4", "2021 without Andalucia: retained-sector term", -0.24, S4.loc["A", "annual_revision_part_noES61"], 2)
check("S17 Step 4", "Andalucia removal-only pairs in 2021", 232, trans["removal only"]["n2021"], 0)
check("S17 Step 4", "of which matched in the 7th release", 225, trans["removal only"]["found"], 0)
check("S17 Step 4", "of which constant coverage in the 7th release", 133, trans["removal only"]["cats"].get("same coverage", 0), 0)
check("S17 Step 4", "of which both additions and removals", 75, trans["removal only"]["cats"].get("both", 0), 0)
check("S17 Step 4", "Andalucia constant-coverage pairs matched", 99, trans["same coverage"]["found"], 0)
check("S17 Step 4", "of which addition only in the 7th release", 95, trans["same coverage"]["cats"].get("addition only", 0), 0)
check("S17 Step 4", "7th release: changed-coverage pairs adding waste", 597, waste_add["n"], 0)
check("S17 Step 4", "of which in Andalucia", 195, waste_add["es61"], 0)
check("S17 Step 4", "of which in Catalonia", 306, waste_add["es51"], 0)
check("S17 processing", "signatories with 2021 covariates: coefficient", -1.10, R.loc["G", "lam"], 2)
check("S17 processing", "one BEI and latest MEI: coefficient", -1.06, R.loc["I", "lam"], 2)
check("S17 processing", "earliest entry: coefficient", -1.11, R.loc["J", "lam"], 2)
check("S17 processing", "new signatories: coefficient", -1.08, R.loc["H", "lam"], 2)
check("S17 processing", "new signatories: clustered SE", 0.37, R.loc["H", "lam_se"], 2)
check("S17 processing", "new signatories: one-sector inventories", 0, R.loc["H", "n_sc1"], 0)
check("S17 processing", "residential buildings only: coefficient", -1.26, R.loc["K", "lam"], 2)
check("S17 processing", "residential buildings only: t against -1", -2.7, R.loc["K", "lam_t_m1"], 1)
check("S17 releases", "same inventories with changed sector sets, %", 17.8, treg.loc["all", "changed"] * 100, 1)
check("S17 releases", "changed sector sets in Andalucia, %", 53.1, treg.loc["Andalucia", "changed"] * 100, 1)
check("S17 releases", "changed sector sets in Catalonia, %", 56.0, treg.loc["Catalonia", "changed"] * 100, 1)
check("S17 releases", "changed sector sets elsewhere, %", 7.1, treg.loc["other", "changed"] * 100, 1)
check("S17 releases", "identical totals in Andalucia, %", 4.3, treg.loc["Andalucia", "ident"] * 100, 1)
check("S17 releases", "identical totals in Catalonia, %", 4.1, treg.loc["Catalonia", "ident"] * 100, 1)
check("S17 releases", "identical totals elsewhere, %", 52.1, treg.loc["other", "ident"] * 100, 1)
check("S17 releases", "Andalucia: retained-sector term", 0.124, treg.loc["Andalucia", "mean_dR"], 3)
check("S17 releases", "Andalucia: scope term", -0.032, treg.loc["Andalucia", "mean_scope"], 3)
check("S17 releases", "variance of the difference", 0.0555, rel["var_dT"], 4)
check("S17 releases", "scope variance", 0.0199, rel["var_scope"], 4)
check("S17 releases", "retained-sector variance", 0.0331, rel["var_dR"], 4)
check("S17 releases", "twice the covariance", 0.0024, rel["cov2"], 4)
check("Table S27", "inventories losing only waste", 988, rel["removed_waste_only"], 0)
check("Table S27", "inventories losing a sector", 1104, rel["removed_any"], 0)
check("Table S27", "inventories gaining only manufacturing", 81, rel["added_manufacturing_only"], 0)
check("Table S27", "inventories gaining only residential", 27, rel["added_residential_only"], 0)
check("Table S27", "inventories gaining a sector", 156, rel["added_any"], 0)
check("Table S27", "largest single-inventory third-moment share, 7th release before the screen, %", 170, R.loc["F", "max_resid_m3share"] * 100, 0)
check("Table S27", "largest single-inventory third-moment share, 7th release, %", 10, R.loc["C", "max_resid_m3share"] * 100, 0)
check("S17 releases", "inventories with waste in both releases", 890, rel["waste_both"], 0)
check("S17 releases", "of which 0.885 times the 2021 value, %", 77.4, rel["waste_0885"] * 100, 1)
chk = pd.DataFrame(chk)
chk.to_csv(os.path.join(TAB_DIR, "check_mycovenant7_numbers.csv"), index=False)
print(f"\ncheck against the manuscript: {int((chk.status == 'OK').sum())} OK, {int((chk.status == 'DIFF').sum())} DIFF")
if (chk.status == "DIFF").any():
    print(chk[chk.status == "DIFF"].to_string(index=False))

with pd.ExcelWriter(os.path.join(TAB_DIR, "MyCovenant7_Tables.xlsx")) as xw:
    r12.to_excel(xw, sheet_name="Steps12_by_sample")
    r4.to_excel(xw, sheet_name="Step4_by_sample")
    wtab.to_excel(xw, sheet_name="S26_joint_test", index=False)
    pd.DataFrame([mech]).to_excel(xw, sheet_name="S26_mechanism", index=False)
    tcat.to_excel(xw, sheet_name="S27_by_coverage")
    treg.to_excel(xw, sheet_name="S27_by_region")
    pd.DataFrame([rel]).to_excel(xw, sheet_name="S27_numbers", index=False)
    infl.to_excel(xw, sheet_name="Screen_influence", index=False)
    chk.to_excel(xw, sheet_name="Check_manuscript", index=False)
for k_, v_ in rel.items():
    note(f"release_{k_}", v_)
pd.DataFrame(nums).to_csv(os.path.join(TAB_DIR, "mycovenant7_numbers.csv"), index=False)
print("saved: work/mycovenant7.pkl, tables/MyCovenant7_Tables.xlsx, tables/mycovenant7_numbers.csv, tables/check_mycovenant7_numbers.csv, "
      "tables/Tables_S23_S25_S26_S27.tex")
