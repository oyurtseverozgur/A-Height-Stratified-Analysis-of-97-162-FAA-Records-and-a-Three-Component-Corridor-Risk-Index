"""
Height-stratified wildlife strike analysis for the low-altitude UAM envelope
and the exposure-normalised UAM Corridor Risk Index (UCRI), revision R1.

Inputs
  - FAA National Wildlife Strike Database export (Public.xlsx, or the same
    sheet saved as CSV)
  - FAA ATADS annual airport operations, 2010-2025 (atads_ops_2010_2025.csv;
    columns: airport, year, total, ...)

Usage
  python uam_wildlife_r1.py --nwsd Public.xlsx --ops atads_ops_2010_2025.csv --out results

Runs end-to-end in about 15-25 min on a laptop (bootstrap loops dominate).
Set --fast to cut bootstrap replicates for a quick check.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
BANDS = [0, 50, 200, 500, 1500, 3500, np.inf]
ZONES = ["A", "B", "C", "D", "E", "F"]
ZONE_LABEL = {"A": "1–50 ft", "B": "51–200 ft", "C": "201–500 ft",
              "D": "501–1,500 ft", "E": "1,501–3,500 ft", "F": ">3,500 ft"}
GROUND = ["Take-off Run", "Landing Roll", "Taxi", "Parked"]
AIRBORNE = ["Approach", "Climb", "Descent", "En Route", "Local", "Departure", "Arrival"]

# metropolitan corridor anchors (FAA location identifiers of the anchor airports)
CORRIDORS = {
    "New York": ["JFK", "LGA", "EWR", "TEB"],
    "Miami": ["MIA", "FLL"],
    "San Francisco–Oakland": ["SFO", "OAK"],
    "Los Angeles": ["LAX"],
    "Chicago": ["ORD", "MDW"],
    "Houston": ["IAH", "HOU"],
    "Seattle": ["SEA"],
    "Dallas–Fort Worth": ["DFW"],
    "Atlanta": ["ATL"],
    "Denver": ["DEN"],
}
BUFFER_KM = 40.0
SHRINK_K = 20
INDEX_YEARS = (2010, 2024)
TRAIN_YEARS, TEST_YEARS = (2010, 2017), (2018, 2024)

GREY = ["#1a1a1a", "#4d4d4d", "#808080", "#a6a6a6", "#cccccc", "#e6e6e6"]


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "savefig.dpi": 300,
    })


def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_nwsd(path):
    cache = os.path.splitext(path)[0] + ".pkl"
    if os.path.exists(cache):
        d = pd.read_pickle(cache)
    else:
        if path.lower().endswith((".xlsx", ".xls")):
            d = pd.read_excel(path, engine="openpyxl")
        else:
            d = pd.read_csv(path, low_memory=False)
        d.to_pickle(cache)
    for c in ["AC_CLASS", "TYPE_ENG", "STATE", "AIRPORT_ID", "SPECIES", "SIZE",
              "PHASE_OF_FLIGHT", "OPERATOR", "TIME_OF_DAY", "FAAREGION"]:
        d[c] = d[c].astype("object").where(d[c].notna(), None)
        d[c] = d[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    for c in ["HEIGHT", "SPEED", "LATITUDE", "LONGITUDE", "AC_MASS", "INCIDENT_YEAR"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    dmg = d["INDICATED_DAMAGE"]
    if dmg.dtype == object:
        dmg = dmg.astype(str).str.lower().isin(["true", "1", "yes"])
    d["dmg"] = dmg.astype(int)
    d["hc"] = d["HEIGHT"] > 0
    d["zone"] = pd.cut(d["HEIGHT"].where(d["hc"]), BANDS, labels=ZONES)
    d["month"] = pd.to_datetime(d["INCIDENT_DATE"], errors="coerce").dt.month
    d["yr5"] = (d["INCIDENT_YEAR"].clip(upper=2025) // 5 * 5).astype("Int64").astype(str)
    d["SIZE"] = d["SIZE"].fillna("NA")
    d["dmgS"] = d["dmg"].astype(str) + "_" + d["SIZE"]
    return d


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def cramers_v(chi2, n, shape):
    return float(np.sqrt(chi2 / (n * (min(shape) - 1))))


# ----------------------------------------------------------------------------
# selection-bias correction: post-stratification weights
# ----------------------------------------------------------------------------
def population(d):
    """Records that could carry a positive height: airborne or unknown phase,
    height not recorded as zero."""
    return d[~d["PHASE_OF_FLIGHT"].isin(GROUND) & ~(d["HEIGHT"] == 0)].copy()


def ps_weights(P, cells=("dmgS", "yr5")):
    key = P[list(cells)].astype(str).agg("|".join, axis=1)
    sel = P["hc"].values
    n_pop = key.value_counts()
    n_sel = key[sel].value_counts()
    return (n_pop / n_sel).reindex(key[sel]).values


def zone_rates(S, w):
    out = {}
    for z in ZONES:
        k = (S["zone"] == z).values
        out[z] = float(np.average(S["dmg"].values[k], weights=w[k]))
    k = S["zone"].isin(list("ABC")).values
    out["A–C"] = float(np.average(S["dmg"].values[k], weights=w[k]))
    out["All"] = float(np.average(S["dmg"].values, weights=w))
    return out


def ps_bootstrap(P, B, seed=7):
    rng = np.random.default_rng(seed)
    key = P[["dmgS", "yr5"]].astype(str).agg("|".join, axis=1).values
    sel = P["hc"].values
    dm = P["dmg"].values
    zn = P["zone"].astype(str).values
    reps = {k: [] for k in ZONES + ["A–C", "All"]}
    for _ in range(B):
        i = rng.integers(0, len(P), len(P))
        kb, sb = key[i], sel[i]
        npop = pd.Series(kb).value_counts()
        nsel = pd.Series(kb[sb]).value_counts()
        wb = (npop / nsel).reindex(kb[sb]).values
        db, zb = dm[i][sb], zn[i][sb]
        for z in ZONES:
            m = zb == z
            reps[z].append(np.average(db[m], weights=wb[m]))
        m = np.isin(zb, list("ABC"))
        reps["A–C"].append(np.average(db[m], weights=wb[m]))
        reps["All"].append(np.average(db, weights=wb))
    return {k: np.array(v) for k, v in reps.items()}


# ----------------------------------------------------------------------------
# adjusted damage model
# ----------------------------------------------------------------------------
def adjusted_model(h, ref="A"):
    x = h.copy()
    x["zone"] = x["zone"].astype(str)
    x["mass"] = x["AC_MASS"].fillna(0).astype(int).astype(str)
    x["eng"] = x["TYPE_ENG"].fillna("NA")
    x["phase"] = x["PHASE_OF_FLIGHT"].fillna("NA").replace(
        {"Arrival": "Other", "Departure": "Other", "Unknown": "Other", "Taxi": "Other",
         "Local": "Other", "Descent": "Other"})
    x["acc"] = x["AC_CLASS"].fillna("NA")
    x["tod"] = x["TIME_OF_DAY"].fillna("NA")
    f = (f"dmg ~ C(zone, Treatment('{ref}')) + C(mass) + C(eng) + C(SIZE) + C(phase)"
         " + C(yr5) + C(acc) + C(tod)")
    m = smf.logit(f, data=x).fit(disp=0, cov_type="HC1")
    ci = m.conf_int()
    out = {}
    for k in m.params.index:
        if "zone" in k:
            z = k.split("T.")[1][0]
            out[z] = (float(np.exp(m.params[k])), float(np.exp(ci.loc[k, 0])),
                      float(np.exp(ci.loc[k, 1])), float(m.pvalues[k]))
    return out, int(m.nobs), float(m.prsquared)


# ----------------------------------------------------------------------------
# UCRI
# ----------------------------------------------------------------------------
def haversine_km(lat, lon, lat0, lon0):
    r = np.pi / 180
    a = (np.sin((lat - lat0) * r / 2) ** 2
         + np.cos(lat * r) * np.cos(lat0 * r) * np.sin((lon - lon0) * r / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(a))


def species_damage_prob(d, years, w, k=SHRINK_K):
    """Empirical-Bayes damage probability per species for strikes at 1-500 ft,
    shrunk toward the weighted damage rate of the species' size class."""
    s = d[(d["HEIGHT"] > 0) & (d["HEIGHT"] <= 500) & d["INCIDENT_YEAR"].between(*years)].copy()
    s["w"] = s["INDEX_NR"].map(w).fillna(1.0)
    s["SP"] = s["SPECIES"].fillna("Unknown bird")
    s["wd"] = s["w"] * s["dmg"]
    size_rate = s.groupby("SIZE").apply(lambda g: g["wd"].sum() / g["w"].sum())
    g = s.groupby("SP").agg(n=("w", "size"), wn=("w", "sum"), wd=("wd", "sum"),
                            SIZE=("SIZE", lambda v: v.mode().iloc[0]))
    prior = g["SIZE"].map(size_rate)
    g["p"] = (g["wd"] + k * prior) / (g["wn"] + k)
    return g, float(np.average(s["dmg"], weights=s["w"]))


def centroid(d, anchors):
    ids = ["K" + a for a in anchors]
    return d[d["AIRPORT_ID"].isin(ids)].groupby("AIRPORT_ID")[["LATITUDE", "LONGITUDE"]].first().mean()


def ucri_components(d, ops, anchors, years, sp, pbar, radius=BUFFER_KM):
    yrs = list(range(years[0], years[1] + 1))
    ids = ["K" + a for a in anchors]
    a = d[d["AIRPORT_ID"].isin(ids) & d["INCIDENT_YEAR"].between(*years)]
    n_ops = ops[ops["airport"].isin(anchors) & ops["year"].isin(yrs)]["total"].sum()
    low = a[(a["HEIGHT"] > 0) & (a["HEIGHT"] <= 500)]
    q = len(low) / max(a["HEIGHT"].notna().sum(), 1)
    E = len(a) / n_ops * 1e5 * q
    c = centroid(d, anchors)
    b = d[(d["HEIGHT"] > 0) & (d["HEIGHT"] <= 500) & d["INCIDENT_YEAR"].between(*years)
          & d["LATITUDE"].notna()]
    b = b[haversine_km(b["LATITUDE"].values, b["LONGITUDE"].values, c["LATITUDE"], c["LONGITUDE"]) <= radius]
    f = b["SPECIES"].fillna("Unknown bird").value_counts(normalize=True)
    C = float((f * sp["p"].reindex(f.index).fillna(pbar)).sum())
    m = b["month"].value_counts().reindex(range(1, 13), fill_value=0).values.astype(float)
    roll = np.array([m[[i, (i + 1) % 12, (i + 2) % 12]].sum() for i in range(12)])
    S = 4 * roll.max() / m.sum()
    return dict(E=E, C=C, S=S, UCRI=E * C, UCRI_peak=E * C * S, q=q,
                peak_start=int(np.argmax(roll)) + 1, n_anchor=len(low), n_buffer=len(b),
                ops=int(n_ops), obs_dmg=low["dmg"].sum() / n_ops * 1e5,
                obs_dmg_all=a["dmg"].sum() / n_ops * 1e5)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(a):
    set_style()
    os.makedirs(a.out, exist_ok=True)
    B_PS, B_RANK, B_VAL = (60, 60, 300) if a.fast else (500, 500, 2000)
    R = {}
    d = load_nwsd(a.nwsd)
    ops = pd.read_csv(a.ops)
    h = d[d["hc"]].copy()
    R["n_total"], R["n_hc"] = len(d), len(h)
    R["n_zero"] = int((d["HEIGHT"] == 0).sum())
    R["date_range"] = [str(d["INCIDENT_DATE"].min())[:10], str(d["INCIDENT_DATE"].max())[:10]]
    R["dmg_full"], R["dmg_hc"] = float(d["dmg"].mean()), float(h["dmg"].mean())
    R["share_le500"] = float((h["HEIGHT"] <= 500).mean())
    R["share_le1500"] = float((h["HEIGHT"] <= 1500).mean())
    R["share_le3500"] = float((h["HEIGHT"] <= 3500).mean())
    R["zero_phase"] = d[d["HEIGHT"] == 0]["PHASE_OF_FLIGHT"].value_counts(normalize=True).round(4).to_dict()
    R["heap_100"] = float((h["HEIGHT"] % 100 == 0).mean())

    # ---- Table 1: representativeness (test of independence, Cramer's V)
    rep = {}
    for var, sub in [("dmg", d), ("SIZE", d[d["SIZE"] != "NA"]),
                     ("PHASE_OF_FLIGHT", d[d["PHASE_OF_FLIGHT"].isin(AIRBORNE)])]:
        t = pd.crosstab(sub[var], sub["hc"])
        c2, p, df, _ = stats.chi2_contingency(t, correction=False)
        rep[var] = dict(chi2=float(c2), df=int(df), p=float(p), V=cramers_v(c2, t.values.sum(), t.shape),
                        n=int(t.values.sum()),
                        share_hc=(t[True] / t[True].sum()).round(4).to_dict(),
                        share_rest=(t[False] / t[False].sum()).round(4).to_dict())
    R["representativeness"] = rep

    # ---- post-stratification weights and corrected zone rates
    P = population(d).reset_index(drop=True)
    S = P[P["hc"]].reset_index(drop=True)
    w = ps_weights(P)
    R["ps_population_n"], R["ps_selected_n"] = len(P), len(S)
    R["ps_population_dmg"] = float(P["dmg"].mean())
    R["ps_ess"] = float(w.sum() ** 2 / (w ** 2).sum())
    corr = zone_rates(S, w)
    boot = ps_bootstrap(P, B_PS)
    R["zone_raw"], R["zone_ps"] = {}, {}
    for z in ZONES:
        x = h[h["zone"] == z]
        k, n = int(x["dmg"].sum()), len(x)
        lo, hi = wilson(k, n)
        R["zone_raw"][z] = dict(n=n, pct=n / len(h) * 100, k=k, rate=k / n * 100, lo=lo * 100, hi=hi * 100,
                                speed_med=float(x["SPEED"].median()),
                                large=float((x["SIZE"] == "Large").mean() * 100),
                                night=float((x["TIME_OF_DAY"] == "Night").mean() * 100))
    for z in ZONES + ["A–C", "All"]:
        R["zone_ps"][z] = dict(rate=corr[z] * 100, lo=float(np.percentile(boot[z], 2.5) * 100),
                               hi=float(np.percentile(boot[z], 97.5) * 100))
    R["p_boot_C_gt_A"] = float(np.mean(boot["C"] > boot["A"]))
    R["p_boot_C_gt_B"] = float(np.mean(boot["C"] > boot["B"]))
    R["p_boot_D_gt_C"] = float(np.mean(boot["D"] > boot["C"]))
    # alternative cell definition (damage x size only) as a robustness check
    w2 = ps_weights(P, ("dmgS",))
    R["zone_ps_alt"] = {k: v * 100 for k, v in zone_rates(S, w2).items()}
    # alternative boundaries
    alt = {}
    Sw = S.assign(w=w)
    for cuts in [(50, 200, 500), (100, 300, 500), (30, 150, 400), (75, 250, 600)]:
        zz = pd.cut(Sw["HEIGHT"], [0, *cuts], labels=["A", "B", "C"])
        alt[str(cuts)] = {k: float(np.average(g["dmg"], weights=g["w"]) * 100) for k, g in Sw.groupby(zz, observed=True)}
    R["boundary_sensitivity"] = alt

    # ---- adjusted logistic model
    R["or_vs_A"], R["model_n"], R["model_r2"] = adjusted_model(h, "A")
    R["or_vs_B"], _, _ = adjusted_model(h, "B")

    # ---- rotorcraft subset
    r = d[d["AC_CLASS"] == "B"]
    rh = r[r["hc"]]
    R["rotor"] = dict(n=len(r), n_hc=len(rh), completion=float(r["HEIGHT"].notna().mean()),
                      fixed_completion=float(d[d["AC_CLASS"] == "A"]["HEIGHT"].notna().mean()),
                      dmg=float(rh["dmg"].mean() * 100),
                      phase=rh["PHASE_OF_FLIGHT"].value_counts(normalize=True).round(3).to_dict(), zones={})
    for z in ZONES:
        x = rh[rh["zone"] == z]
        k, n = int(x["dmg"].sum()), len(x)
        lo, hi = wilson(k, n)
        R["rotor"]["zones"][z] = dict(n=n, k=k, rate=k / n * 100, lo=lo * 100, hi=hi * 100,
                                      speed=float(x["SPEED"].median()),
                                      large=float((x["SIZE"] == "Large").mean() * 100))
    for p_ in [("C", "A"), ("C", "B"), ("C", "D")]:
        t = np.array([[rh[rh["zone"] == z]["dmg"].sum(), (rh["zone"] == z).sum() - rh[rh["zone"] == z]["dmg"].sum()] for z in p_])
        R[f"rotor_p_{p_[0]}{p_[1]}"] = float(stats.chi2_contingency(t, correction=False)[1])

    # ---- sub-500 ft environment
    low = h[h["HEIGHT"] <= 500]
    R["low_n"] = len(low)
    R["low_phase"] = low["PHASE_OF_FLIGHT"].fillna("Not reported").value_counts().to_dict()
    R["low_size"] = low["SIZE"].value_counts().to_dict()
    R["low_size_dmg"] = (low.groupby("SIZE")["dmg"].mean() * 100).round(2).to_dict()
    sp_low = low[~low["SPECIES"].fillna("").str.contains("Unknown")].groupby("SPECIES")["dmg"].agg(["size", "mean"]).sort_values("size", ascending=False)
    R["low_species_top"] = sp_low.head(10).round(4).to_dict(orient="index")

    # ---- seasonality
    seas = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    R["season"], R["month_ratio"] = {}, {}
    for z in "ABCD":
        mc = h[h["zone"] == z]["month"].value_counts().sort_index()
        rr = mc / mc.mean()
        R["month_ratio"][z] = rr.round(3).to_dict()
        R["season"][z] = rr.groupby(rr.index.map(seas)).mean().round(3).to_dict()

    # ---- temporal trend 1990-2025
    yr = d[d["INCIDENT_YEAR"] <= 2025].groupby("INCIDENT_YEAR").agg(n=("dmg", "size"), hc=("hc", "sum"))
    yr["low"] = low.groupby("INCIDENT_YEAR").size()
    yr["share"] = yr["low"] / yr["hc"]
    yr["completion"] = yr["hc"] / yr["n"]
    sl = stats.linregress(yr.index, yr["share"])
    R["trend"] = dict(slope_pp_decade=sl.slope * 1000, p=sl.pvalue,
                      share_1990s=float(yr.loc[1990:1999, "share"].mean()),
                      share_2020_25=float(yr.loc[2020:2025, "share"].mean()),
                      compl_1990s=float(yr.loc[1990:1999, "completion"].mean()),
                      compl_2020_25=float(yr.loc[2020:2025, "completion"].mean()))
    yr.to_csv(os.path.join(a.out, "annual_trend.csv"))

    # ---- UCRI
    wmap = pd.Series(w, index=S["INDEX_NR"].values)
    wmap = wmap[~wmap.index.duplicated()]
    sp, pbar = species_damage_prob(d, INDEX_YEARS, wmap)
    sp.sort_values("n", ascending=False).to_csv(os.path.join(a.out, "species_damage_probability.csv"))
    T = pd.DataFrame({c: ucri_components(d, ops, an, INDEX_YEARS, sp, pbar) for c, an in CORRIDORS.items()}).T
    T = T.astype(float)
    T["rank"] = T["UCRI"].rank(ascending=False).astype(int)
    T["rank_peak"] = T["UCRI_peak"].rank(ascending=False).astype(int)
    # bootstrap rank uncertainty
    rng = np.random.default_rng(11)
    nat = d[(d["HEIGHT"] > 0) & (d["HEIGHT"] <= 500) & d["INCIDENT_YEAR"].between(*INDEX_YEARS)].reset_index(drop=True)
    pre = {}
    for c, an in CORRIDORS.items():
        cc = centroid(d, an)
        bb = nat[nat["LATITUDE"].notna()]
        bb = bb[haversine_km(bb["LATITUDE"].values, bb["LONGITUDE"].values, cc["LATITUDE"], cc["LONGITUDE"]) <= BUFFER_KM]
        pre[c] = (T.loc[c, "E"] * T.loc[c, "ops"] / 1e5, T.loc[c, "ops"],
                  bb["SPECIES"].fillna("Unknown bird").values, bb["month"].values)
    ranks, ranks_pk = [], []
    for _ in range(B_RANK):
        spb, pb = species_damage_prob(nat.iloc[rng.integers(0, len(nat), len(nat))], INDEX_YEARS, wmap)
        ann, pk = {}, {}
        for c, (na, no, spc, mo) in pre.items():
            E = rng.poisson(na) / no * 1e5
            i = rng.integers(0, len(spc), len(spc))
            f = pd.Series(spc[i]).value_counts(normalize=True)
            C = float((f * spb["p"].reindex(f.index).fillna(pb)).sum())
            m = np.bincount(mo[i].astype(int), minlength=13)[1:].astype(float)
            roll = np.array([m[[k, (k + 1) % 12, (k + 2) % 12]].sum() for k in range(12)])
            ann[c] = E * C
            pk[c] = E * C * 4 * roll.max() / m.sum()
        ranks.append(pd.Series(ann).rank(ascending=False))
        ranks_pk.append(pd.Series(pk).rank(ascending=False))
    Rk, Rp = pd.DataFrame(ranks), pd.DataFrame(ranks_pk)
    T["rank_lo"], T["rank_hi"] = Rk.quantile(0.025), Rk.quantile(0.975)
    T["p_top3"] = (Rk <= 3).mean()
    T["peak_rank_lo"], T["peak_rank_hi"] = Rp.quantile(0.025), Rp.quantile(0.975)
    T.sort_values("UCRI", ascending=False).to_csv(os.path.join(a.out, "ucri_corridors.csv"))

    # ---- temporal hold-out validation on 45 ATADS airports
    sp_tr, pb_tr = species_damage_prob(d, TRAIN_YEARS, wmap)
    rows = []
    for ap in sorted(ops["airport"].unique()):
        tr = ucri_components(d, ops, [ap], TRAIN_YEARS, sp_tr, pb_tr)
        te = ucri_components(d, ops, [ap], TEST_YEARS, sp_tr, pb_tr)
        st = d[d["AIRPORT_ID"] == "K" + ap]["STATE"].mode()
        rows.append(dict(airport=ap, state=st.iloc[0] if len(st) else None, E=tr["E"], C=tr["C"], S=tr["S"],
                         UCRI=tr["UCRI"], UCRI_peak=tr["UCRI_peak"], obs=te["obs_dmg"], obs_all=te["obs_dmg_all"],
                         n_train=tr["n_anchor"], n_test=te["n_anchor"]))
    V = pd.DataFrame(rows)
    st_cnt = d[(d["HEIGHT"] > 0) & (d["HEIGHT"] <= 500) & d["INCIDENT_YEAR"].between(*TRAIN_YEARS)]["STATE"].value_counts()
    V["state_density"] = V["state"].map(st_cnt)
    mm = lambda x: (x - x.min()) / (x.max() - x.min())
    V["additive"] = 0.4 * mm(V["E"]) + 0.4 * mm(V["C"]) + 0.2 * mm(V["S"])
    cands = ["UCRI", "UCRI_peak", "E", "C", "S", "additive", "state_density"]
    val = {}
    rng = np.random.default_rng(3)
    bs = {k: [] for k in cands}
    for _ in range(B_VAL):
        v = V.iloc[rng.integers(0, len(V), len(V))]
        for k in cands:
            bs[k].append(stats.spearmanr(v[k], v["obs"])[0])
    for k in cands:
        rho, p = stats.spearmanr(V[k], V["obs"], nan_policy="omit")
        rho2, p2 = stats.spearmanr(V[k], V["obs_all"], nan_policy="omit")
        val[k] = dict(rho=float(rho), p=float(p), lo=float(np.nanpercentile(bs[k], 2.5)),
                      hi=float(np.nanpercentile(bs[k], 97.5)), rho_all=float(rho2), p_all=float(p2))
    diff = np.array(bs["UCRI"]) - np.array(bs["state_density"])
    val["UCRI_minus_state"] = [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))]
    diff = np.array(bs["UCRI"]) - np.array(bs["E"])
    val["UCRI_minus_E"] = [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))]
    R["validation"], R["validation_n"] = val, len(V)
    V.to_csv(os.path.join(a.out, "validation_airports.csv"), index=False)

    # ---- weighting sensitivity for additive alternatives (answers the weight question)
    base = T["UCRI"]
    top3 = set(base.sort_values(ascending=False).index[:3])
    grid = []
    g = np.round(np.arange(0, 1.0001, 0.05), 2)
    for w1 in g:
        for w2 in g:
            w3 = round(1 - w1 - w2, 2)
            if w3 < -1e-9:
                continue
            A_ = w1 * mm(T["E"]) + w2 * mm(T["C"]) + w3 * mm(T["S"])
            Av = w1 * mm(V["E"]) + w2 * mm(V["C"]) + w3 * mm(V["S"])
            grid.append(dict(w1=w1, w2=w2, w3=w3, rho_vs_ucri=stats.spearmanr(A_, base)[0],
                             top3_overlap=len(set(A_.sort_values(ascending=False).index[:3]) & top3),
                             rho_valid=stats.spearmanr(Av, V["obs"])[0]))
    G = pd.DataFrame(grid)
    G.to_csv(os.path.join(a.out, "weight_grid.csv"), index=False)
    R["weight_grid"] = dict(n=len(G), f_rho90=float((G["rho_vs_ucri"] >= 0.9).mean()),
                            f_top3=float((G["top3_overlap"] == 3).mean()),
                            valid_max=float(G["rho_valid"].max()),
                            valid_best=G.loc[G["rho_valid"].idxmax(), ["w1", "w2", "w3"]].to_dict(),
                            valid_median=float(G["rho_valid"].median()),
                            valid_0402=float(G[(G.w1 == 0.4) & (G.w2 == 0.4)]["rho_valid"].iloc[0]),
                            f_valid_below_ucri=float((G["rho_valid"] < val["UCRI"]["rho"]).mean()))
    sel = [(0.40, 0.40, 0.20), (0.30, 0.30, 0.40), (0.45, 0.45, 0.10), (1 / 3, 1 / 3, 1 / 3),
           (0.50, 0.30, 0.20), (0.30, 0.50, 0.20), (0.50, 0.50, 0.00)]
    rows = []
    for ww in sel:
        A_ = ww[0] * mm(T["E"]) + ww[1] * mm(T["C"]) + ww[2] * mm(T["S"])
        Av = ww[0] * mm(V["E"]) + ww[1] * mm(V["C"]) + ww[2] * mm(V["S"])
        rows.append(dict(weights="/".join(f"{x:.2f}" for x in ww), rho_vs_ucri=stats.spearmanr(A_, base)[0],
                         top3=" > ".join(A_.sort_values(ascending=False).index[:3]),
                         rho_valid=stats.spearmanr(Av, V["obs"])[0]))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "weight_selected.csv"), index=False)
    # exponent (weighted geometric) and buffer/shrinkage sensitivity
    ex = []
    for e1 in [0.5, 0.75, 1, 1.25, 1.5]:
        for e2 in [0.5, 0.75, 1, 1.25, 1.5]:
            X = T["E"] ** e1 * T["C"] ** e2
            ex.append(dict(a=e1, b=e2, rho=stats.spearmanr(X, base)[0],
                           top3=len(set(X.sort_values(ascending=False).index[:3]) & top3)))
    EX = pd.DataFrame(ex)
    R["exponent"] = dict(n=len(EX), f_rho90=float((EX["rho"] >= 0.9).mean()), rho_min=float(EX["rho"].min()),
                         f_top3_2=float((EX["top3"] >= 2).mean()))
    alt = []
    for rad in [25, 40, 60]:
        for k in [5, 20, 50]:
            spk, pbk = species_damage_prob(d, INDEX_YEARS, wmap, k=k)
            u = pd.Series({c: ucri_components(d, ops, an, INDEX_YEARS, spk, pbk, radius=rad)["UCRI"] for c, an in CORRIDORS.items()})
            alt.append(dict(radius_km=rad, k=k, rho=stats.spearmanr(u, base)[0],
                            top3=" > ".join(u.sort_values(ascending=False).index[:3])))
    pd.DataFrame(alt).to_csv(os.path.join(a.out, "buffer_shrinkage.csv"), index=False)
    R["buffer_shrinkage_rho_min"] = float(pd.DataFrame(alt)["rho"].min())

    json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)

    # ---------------------------------------------------------------- figures
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    xl = [ZONE_LABEL[z].replace(" ft", "") for z in ZONES]

    # Figure 1: counts and raw vs corrected damage rates
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    n = [R["zone_raw"][z]["n"] for z in ZONES]
    ax[0].bar(range(6), n, color=[GREY[1]] * 3 + [GREY[3]] * 3, edgecolor="black", linewidth=0.5)
    ax[0].set_xticks(range(6)); ax[0].set_xticklabels(xl, rotation=30, ha="right")
    ax[0].set_ylabel("Height-coded strikes (n)"); ax[0].set_xlabel("Height band (ft AGL)")
    ax[0].yaxis.grid(True, linestyle="--", alpha=0.2); ax[0].set_axisbelow(True)
    ax[0].text(-0.18, 1.02, "(a)", transform=ax[0].transAxes, fontsize=9)
    raw = np.array([R["zone_raw"][z]["rate"] for z in ZONES])
    rlo = raw - np.array([R["zone_raw"][z]["lo"] for z in ZONES]); rhi = np.array([R["zone_raw"][z]["hi"] for z in ZONES]) - raw
    cor = np.array([R["zone_ps"][z]["rate"] for z in ZONES])
    clo = cor - np.array([R["zone_ps"][z]["lo"] for z in ZONES]); chi_ = np.array([R["zone_ps"][z]["hi"] for z in ZONES]) - cor
    xs = np.arange(6)
    ax[1].errorbar(xs - 0.1, raw, yerr=[rlo, rhi], fmt="o", color="black", mfc="white", ms=4, capsize=2, lw=0.8, label="Raw (height-coded subset)")
    ax[1].errorbar(xs + 0.1, cor, yerr=[clo, chi_], fmt="s", color="black", ms=4, capsize=2, lw=0.8, label="Selection-corrected")
    ax[1].axhline(R["ps_population_dmg"] * 100, color=GREY[2], lw=0.8, linestyle=":")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(xl, rotation=30, ha="right")
    ax[1].set_ylabel("Damage rate (%)"); ax[1].set_xlabel("Height band (ft AGL)"); ax[1].set_ylim(0, 18)
    ax[1].yaxis.grid(True, linestyle="--", alpha=0.2); ax[1].set_axisbelow(True)
    ax[1].legend(frameon=False, loc="upper left")
    ax[1].text(-0.15, 1.02, "(b)", transform=ax[1].transAxes, fontsize=9)
    fig.tight_layout(); save(fig, os.path.join(fig_dir, "Figure1.png"))

    # Figure 3: sub-500 ft phase and species
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1, 1.4]})
    ph = pd.Series(R["low_phase"]).sort_values(ascending=True)
    ax[0].barh(range(len(ph)), ph.values, color=GREY[2], edgecolor="black", linewidth=0.5)
    ax[0].set_yticks(range(len(ph))); ax[0].set_yticklabels(ph.index); ax[0].set_xscale("log")
    ax[0].set_xlabel("Strikes at 1–500 ft AGL (n, log scale)")
    ax[0].text(-0.38, 1.02, "(a)", transform=ax[0].transAxes, fontsize=9)
    spt = sp_low.head(10).iloc[::-1]
    ax[1].barh(range(len(spt)), spt["size"].values, color=GREY[3], edgecolor="black", linewidth=0.5)
    ax[1].set_yticks(range(len(spt))); ax[1].set_yticklabels(spt.index)
    ax[1].set_xlabel("Strikes at 1–500 ft AGL (n)")
    ax2 = ax[1].twiny(); ax2.plot(spt["mean"].values * 100, range(len(spt)), "kD", ms=3.5)
    ax2.set_xlabel("Raw damage rate (%, diamonds)"); ax2.spines["top"].set_visible(True)
    ax[1].text(-0.55, 1.02, "(b)", transform=ax[1].transAxes, fontsize=9)
    fig.tight_layout(); save(fig, os.path.join(fig_dir, "Figure3.png"))

    # Figure 2: rotorcraft analogue
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    rz = R["rotor"]["zones"]
    rr = np.array([rz[z]["rate"] for z in ZONES])
    ax[0].errorbar(xs, rr, yerr=[rr - np.array([rz[z]["lo"] for z in ZONES]), np.array([rz[z]["hi"] for z in ZONES]) - rr],
                   fmt="o", color="black", ms=4, capsize=2, lw=0.8)
    for i, z in enumerate(ZONES):
        ax[0].annotate(f"n={rz[z]['n']:,}", (i, rz[z]["lo"]), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=6.5)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(xl, rotation=30, ha="right"); ax[0].set_ylim(0, 35)
    ax[0].set_ylabel("Damage rate (%)"); ax[0].set_xlabel("Height band (ft AGL)")
    ax[0].yaxis.grid(True, linestyle="--", alpha=0.2)
    ax[0].text(-0.15, 1.02, "(a)", transform=ax[0].transAxes, fontsize=9)
    sr = [rz[z]["speed"] for z in ZONES]; sa = [R["zone_raw"][z]["speed_med"] for z in ZONES]
    ax[1].plot(xs, sr, "o-", color="black", ms=4, lw=0.9, label="Rotorcraft")
    ax[1].plot(xs, sa, "s--", color=GREY[2], ms=4, lw=0.9, label="All aircraft")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(xl, rotation=30, ha="right")
    ax[1].set_ylabel("Median reported speed (kt)"); ax[1].set_xlabel("Height band (ft AGL)")
    ax[1].legend(frameon=False); ax[1].yaxis.grid(True, linestyle="--", alpha=0.2)
    ax[1].text(-0.15, 1.02, "(b)", transform=ax[1].transAxes, fontsize=9)
    fig.tight_layout(); save(fig, os.path.join(fig_dir, "Figure2.png"))

    # Figure 4: size and seasonal matrix
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
    sz = pd.Series(R["low_size"]).reindex(["Small", "Medium", "Large"])
    ax[0].bar(range(3), sz.values, color=[GREY[4], GREY[2], GREY[0]], edgecolor="black", linewidth=0.5)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(sz.index); ax[0].set_ylabel("Strikes at 1–500 ft AGL (n)")
    for i, s_ in enumerate(sz.index):
        ax[0].annotate(f"{R['low_size_dmg'][s_]:.1f}%", (i, sz[s_]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax[0].text(-0.3, 1.02, "(a)", transform=ax[0].transAxes, fontsize=9)
    M = np.array([[R["month_ratio"][z].get(m, np.nan) for m in range(1, 13)] for z in "ABCD"])
    im = ax[1].imshow(M, cmap="cividis", aspect="auto", vmin=0.3, vmax=2.2)
    ax[1].set_yticks(range(4)); ax[1].set_yticklabels([ZONE_LABEL[z] for z in "ABCD"])
    ax[1].set_xticks(range(12)); ax[1].set_xticklabels(list("JFMAMJJASOND"))
    for i in range(4):
        for j in range(12):
            ax[1].text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < 1.2 else "black")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.04, pad=0.02); cb.set_label("Monthly / mean monthly count", fontsize=7)
    ax[1].text(-0.25, 1.02, "(b)", transform=ax[1].transAxes, fontsize=9)
    fig.tight_layout(); save(fig, os.path.join(fig_dir, "Figure4.png"))

    # Figure 5: trend and ECDF
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].plot(yr.index, yr["share"] * 100, "o-", color="black", ms=2.5, lw=0.9, label="Share of height-coded strikes at 1–500 ft")
    ax[0].plot(yr.index, yr["completion"] * 100, "s--", color=GREY[2], ms=2.5, lw=0.9, label="Height field completion (positive heights)")
    ax[0].set_ylabel("Percent"); ax[0].set_xlabel("Year"); ax[0].set_ylim(0, 70)
    ax[0].legend(frameon=False, loc="lower left"); ax[0].yaxis.grid(True, linestyle="--", alpha=0.2)
    ax[0].text(-0.15, 1.02, "(a)", transform=ax[0].transAxes, fontsize=9)
    hs = np.sort(h["HEIGHT"].values); cdf = np.arange(1, len(hs) + 1) / len(hs)
    ax[1].plot(hs, cdf * 100, color="black", lw=1.0); ax[1].set_xscale("log")
    for x_, lab in [(500, "500"), (1500, "1,500"), (3500, "3,500")]:
        ax[1].axvline(x_, color=GREY[3], lw=0.7, linestyle=":")
    ax[1].set_xlabel("Strike height (ft AGL, log scale)"); ax[1].set_ylabel("Cumulative share (%)")
    ax[1].yaxis.grid(True, linestyle="--", alpha=0.2)
    ax[1].text(-0.15, 1.02, "(b)", transform=ax[1].transAxes, fontsize=9)
    fig.tight_layout(); save(fig, os.path.join(fig_dir, "Figure5.png"))

    # Figure 6: UCRI components and rank intervals
    Ts = T.sort_values("UCRI", ascending=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].scatter(T["E"], T["C"] * 100, s=12 + 30 * T["S"], facecolor="white", edgecolor="black", linewidth=0.8)
    offs = {"Los Angeles": (-6, -11), "Miami": (5, 5), "Dallas–Fort Worth": (-30, 8), "New York": (5, 4)}
    for c in T.index:
        ax[0].annotate(c, (T.loc[c, "E"], T.loc[c, "C"] * 100), textcoords="offset points",
                       xytext=offs.get(c, (5, 3)), fontsize=6.5)
    ee = np.linspace(T["E"].min() * 0.9, T["E"].max() * 1.05, 50)
    for lvl in [0.3, 0.5, 0.7]:
        ax[0].plot(ee, lvl / ee * 100, color=GREY[3], lw=0.6, linestyle="--")
    ax[0].set_xlabel("E: strikes at 1–500 ft per 10$^5$ operations"); ax[0].set_ylabel("C: expected damage probability (%)")
    ax[0].set_ylim(1.5, 7); ax[0].text(-0.15, 1.02, "(a)", transform=ax[0].transAxes, fontsize=9)
    yy = np.arange(len(Ts))
    ax[1].hlines(yy, Ts["rank_lo"], Ts["rank_hi"], color="black", lw=1.0)
    ax[1].plot(Ts["rank"], yy, "o", color="black", ms=4)
    ax[1].set_yticks(yy); ax[1].set_yticklabels(Ts.index); ax[1].invert_xaxis()
    ax[1].set_xticks(range(1, 11)); ax[1].set_xlabel("UCRI rank (95% bootstrap interval)")
    ax[1].xaxis.grid(True, linestyle="--", alpha=0.2)
    ax[1].text(-0.5, 1.02, "(b)", transform=ax[1].transAxes, fontsize=9)
    fig.tight_layout(); save(fig, os.path.join(fig_dir, "Figure6.png"))

    # Figure 7: validation
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax[0].scatter(V["UCRI"], V["obs"], s=12, facecolor="white", edgecolor="black", linewidth=0.7)
    lab_ap = {"JFK": (4, -3), "SFO": (4, 2), "DEN": (4, -6), "DFW": (4, -6), "ORD": (5, -7), "MIA": (4, 2),
              "SEA": (4, 2), "LAX": (-20, 4), "IAH": (4, 2), "ATL": (-6, 5)}
    for _, rw in V.iterrows():
        if rw["airport"] in lab_ap:
            ax[0].annotate(rw["airport"], (rw["UCRI"], rw["obs"]), textcoords="offset points",
                           xytext=lab_ap[rw["airport"]], fontsize=6)
    ax[0].set_xscale("log"); ax[0].set_ylim(-0.03, 1.05)
    ax[0].yaxis.grid(True, linestyle="--", alpha=0.2)
    ax[0].set_xlabel("UCRI, 2010–2017 (damaging strikes per 10$^5$ ops)")
    ax[0].set_ylabel("Observed 2018–2024 damaging\nstrikes at 1–500 ft per 10$^5$ ops")
    ax[0].text(-0.2, 1.02, "(a)", transform=ax[0].transAxes, fontsize=9)
    labs = {"UCRI": "UCRI (E×C)", "UCRI_peak": "UCRI peak (E×C×S)", "C": "C only", "additive": "Additive 0.4/0.4/0.2",
            "E": "E only", "S": "S only", "state_density": "State density (R0 D$_c$)"}
    order = ["UCRI", "UCRI_peak", "C", "additive", "E", "S", "state_density"]
    vals = [val[k]["rho"] for k in order]
    ax[1].barh(range(len(order))[::-1], vals, xerr=[[val[k]["rho"] - val[k]["lo"] for k in order], [val[k]["hi"] - val[k]["rho"] for k in order]],
               color=[GREY[1]] * 2 + [GREY[3]] * 5, edgecolor="black", linewidth=0.5, capsize=2, error_kw=dict(lw=0.7))
    ax[1].set_yticks(range(len(order))[::-1]); ax[1].set_yticklabels([labs[k] for k in order])
    ax[1].axvline(0, color="black", lw=0.6); ax[1].set_xlabel("Spearman ρ with 2018–2024 outcome (n = 45)")
    ax[1].set_xlim(-0.5, 1.0); ax[1].text(-0.55, 1.02, "(b)", transform=ax[1].transAxes, fontsize=9)
    fig.tight_layout(); save(fig, os.path.join(fig_dir, "Figure7.png"))
    print("done:", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nwsd", required=True)
    ap.add_argument("--ops", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true")
    main(ap.parse_args())
