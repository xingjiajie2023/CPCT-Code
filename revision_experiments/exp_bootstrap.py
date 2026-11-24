# -*- coding: utf-8 -*-
"""Exp 4: patient-level paired bootstrap (B=2000) on real outer predictions.

Unit of resampling = patient; all drug records of a sampled patient are carried
together.  Repeated outer splits are NOT treated as independent cohorts: for
each patient-drug we first average the out-of-fold HARD scores across the three
partition seeds (label verified identical), giving one prediction per
patient-drug.  We then resample patients with replacement and compute, per
drug, CPCT - Clinical-MTL paired differences in AUROC / AUPR / Brier with 95%
percentile CIs.  A per-seed analysis is also emitted as a secondary check.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "source_data")
CPCT_OUT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "cpct_select", "outputs"))
DRUGS = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", "Doxorubicin",
         "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]
B = 2000
SEED = 12345


def metrics(y, pc, pk):
    ok = np.isfinite(pc) & np.isfinite(pk)
    y, pc, pk = y[ok], pc[ok], pk[ok]
    if len(np.unique(y)) < 2:
        return (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)
    return (roc_auc_score(y, pc), average_precision_score(y, pc), np.mean((pc - y) ** 2)), \
           (roc_auc_score(y, pk), average_precision_score(y, pk), np.mean((pk - y) ** 2))


def main():
    pp = pd.read_csv(os.path.join(CPCT_OUT, "per_patient_predictions.csv"))
    pp = pp[pp.method.isin(["CPCT_HARD", "Clinical_HARD"])].copy()
    # label consistency across seeds per patient-drug
    chk = pp.groupby(["patient_id", "drug"])["label"].nunique()
    print("patient-drug label inconsistency across seeds:", int((chk > 1).sum()))
    # average scores across seeds -> one out-of-fold prediction per patient-drug
    agg = pp.groupby(["method", "patient_id", "drug"]).agg(score=("score", "mean"),
                                                           label=("label", "first")).reset_index()
    c = agg[agg.method == "CPCT_HARD"].set_index(["patient_id", "drug"])
    k = agg[agg.method == "Clinical_HARD"].set_index(["patient_id", "drug"])
    common = c.index.intersection(k.index)
    c, k = c.loc[common].reset_index(), k.loc[common].reset_index()
    assert np.array_equal(c.patient_id, k.patient_id) and np.array_equal(c.drug, k.drug)
    assert np.allclose(c.label, k.label)
    patients = np.sort(c.patient_id.unique())
    pidx = {p: i for i, p in enumerate(patients)}
    pi = c.patient_id.map(pidx).to_numpy()
    y = c.label.to_numpy(float)
    pc, pk = c.score.to_numpy(float), k.score.to_numpy(float)

    rng = np.random.default_rng(SEED)
    rows = []
    point = {}
    for j, drug in enumerate(DRUGS):
        m = (c.drug == drug).to_numpy()
        yy, a, b = y[m], pc[m], pk[m]
        # point estimates on full data
        (auc_c, ap_c, bs_c), (auc_k, ap_k, bs_k) = metrics(yy, a, b)
        point[drug] = dict(AUROC_CPCT=auc_c, AUROC_Clinical=auc_k, AUROC_diff=auc_c - auc_k,
                           AUPR_CPCT=ap_c, AUPR_Clinical=ap_k, AUPR_diff=ap_c - ap_k,
                           Brier_CPCT=bs_c, Brier_Clinical=bs_k, Brier_diff=bs_c - bs_k,
                           n=len(yy), n_pos=int(yy.sum()))
        # patient blocks for this drug
        pats = np.unique(pi[m])
        rec = {p: np.flatnonzero(pi[m] == p) for p in pats}
        dA = np.empty(B); dP = np.empty(B); dB = np.empty(B)
        for bi in range(B):
            draw = rng.choice(pats, size=len(pats), replace=True)
            ii = np.concatenate([rec[p] for p in draw])
            (ac, apc, bc), (ak, apk, bk) = metrics(yy[ii], a[ii], b[ii])
            dA[bi], dP[bi], dB[bi] = ac - ak, apc - apk, bc - bk
        for name, arr in [("AUROC", dA), ("AUPR", dP), ("Brier", dB)]:
            rows.append(dict(drug=drug, metric=name, point=point[drug][name + "_diff"],
                             ci_lo=float(np.percentile(arr, 2.5)),
                             ci_hi=float(np.percentile(arr, 97.5)),
                             sd=float(np.std(arr, ddof=1)), B=B))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bootstrap_paired_ci.csv"), index=False)
    pd.DataFrame(point).T.to_csv(os.path.join(OUT, "bootstrap_point_estimates.csv"))
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)

    # secondary: per-seed bootstrap (kept separate, not pooled as independent cohorts)
    srows = []
    for seed in (2024, 2025, 2026):
        sub = pp[pp.seed == seed]
        cc = sub[sub.method == "CPCT_HARD"].set_index(["patient_id", "drug"])
        kk = sub[sub.method == "Clinical_HARD"].set_index(["patient_id", "drug"])
        ci = cc.index.intersection(kk.index)
        cc, kk = cc.loc[ci].reset_index(), kk.loc[ci].reset_index()
        pats = np.sort(cc.patient_id.unique()); pm = {p: i for i, p in enumerate(pats)}
        pii = cc.patient_id.map(pm).to_numpy(); yy = cc.label.to_numpy(float)
        aa = cc.score.to_numpy(float); bb = kk.score.to_numpy(float)
        for drug in DRUGS:
            m = (cc.drug == drug).to_numpy()
            y_, a_, b_ = yy[m], aa[m], bb[m]
            ps = np.unique(pii[m]); rec = {p: np.flatnonzero(pii[m] == p) for p in ps}
            dA = np.empty(B)
            for bi in range(B):
                draw = rng.choice(ps, size=len(ps), replace=True)
                ii = np.concatenate([rec[p] for p in draw])
                (x1, _, _), (x2, _, _) = metrics(y_[ii], a_[ii], b_[ii])
                dA[bi] = x1 - x2
            srows.append(dict(seed=seed, drug=drug, metric="AUROC",
                              point=float(np.mean(dA)), ci_lo=float(np.percentile(dA, 2.5)),
                              ci_hi=float(np.percentile(dA, 97.5)), B=B))
    pd.DataFrame(srows).to_csv(os.path.join(OUT, "bootstrap_paired_per_seed.csv"), index=False)
    print("BOOTSTRAP DONE", flush=True)


if __name__ == "__main__":
    main()
