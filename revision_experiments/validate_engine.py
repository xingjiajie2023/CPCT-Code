# -*- coding: utf-8 -*-
"""Validate a from-scratch clinical engine against the archived HARD readout.

The engine mirrors cpct_ablation_final/code/run_variants.py (fit_general),
recomputing inner losses on the SAME 36-config pool and outer gene/profile
scalers/prior as the archive.  If HARD == archived Contrast_drugwise, the
engine is faithful and can be reused for inductive / crossfit / prior variants.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
CPCT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "cpct_revision"))
ABLC = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "cpct_ablation_final", "code"))
SEL = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "cpct_select", "code"))
sys.path.insert(0, ABLC)
sys.path.insert(0, SEL)
from common import fit_general, predict_general  # noqa
from splits import patient_splits  # noqa

ROOT = os.environ["DCIT_DATA_ROOT"]
ARCH = os.path.join(os.path.dirname(SEL), "archive")

CC = (0.0001, 0.001, 0.01)
CP = (0.0, 0.0001, 0.001)
CS = (0.0, 0.1, 1.0, 10.0)
GRID = [(a, b, c) for a in CC for b in CP for c in CS]
assert len(GRID) == 36


def load():
    a = np.load(os.path.join(ROOT, "dcit/cache/patients.npz"))
    prof = np.load(os.path.join(SEL, os.pardir, "outputs/source/profiles.npz"))
    prior = np.load(os.path.join(SEL, os.pardir, "outputs/source/source_prior.npz"))["weights"].astype(float)
    phi = np.column_stack([prof["target_pan"], prof["target_private"]])
    return dict(x=a["X"].astype(float), y=a["Y"].astype(float), m=a["mask"].astype(bool),
                ids=a["patient_ids"], folds={s: a["folds_%d" % s] for s in (2024, 2025, 2026)},
                phi=phi, V=prior)


def cell(d, seed, fold, phi, V):
    x, y, m = d["x"], d["y"], d["m"]
    train = np.flatnonzero(d["folds"][seed] != fold)
    test = np.flatnonzero(d["folds"][seed] == fold)
    losses = np.zeros((36, 3, 9))
    for part, (it, iv) in enumerate(patient_splits(y[train], m[train], 3, 50000 + seed + fold)):
        tr, va = train[it], train[iv]
        xs = StandardScaler().fit(x[tr]); xx = xs.transform(x)
        ps = StandardScaler().fit(phi[tr]); ss = ps.transform(phi)[:, None, :] * np.sqrt(V)[None, :, :]
        for ci, (cc, cp, cs) in enumerate(GRID):
            model = fit_general(xx[tr], y[tr], m[tr], ss[tr], cc, cp, cs,
                                use_shared=True, use_private=True, use_source=True)
            pp = np.clip(predict_general(model, xx[va], ss[va]), 1e-12, 1 - 1e-12)
            yy = np.where(m[va], y[va], 0.)
            losses[ci, part] = ((-yy * np.log(pp) - (1 - yy) * np.log1p(-pp)) * m[va]).sum(axis=0) / m[va].sum(axis=0)
    choice = losses.mean(axis=1).argmin(axis=0)
    xs = StandardScaler().fit(x[train]); xx = xs.transform(x)
    ps = StandardScaler().fit(phi[train]); ss = ps.transform(phi)[:, None, :] * np.sqrt(V)[None, :, :]
    probs = {}
    for ci in np.unique(choice):
        cc, cp, cs = GRID[ci]
        model = fit_general(xx[train], y[train], m[train], ss[train], cc, cp, cs,
                            use_shared=True, use_private=True, use_source=True)
        probs[int(ci)] = predict_general(model, xx[test], ss[test])
    pred = np.stack([probs[int(choice[j])][:, j] for j in range(9)], axis=1)
    return pred, losses, choice


if __name__ == "__main__":
    d = load()
    arch = pd.read_csv(os.path.join(ARCH, "nested/per_patient.csv"))
    arch = arch[arch.method == "Contrast_drugwise"]
    archc = pd.read_csv(os.path.join(ARCH, "nested/per_patient.csv"))
    archc = archc[archc.method == "Clinical_drugwise"]
    pos = {p: i for i, p in enumerate(np.unique(d["ids"]))}
    rows = []
    for seed in (2024, 2025, 2026):
        for fold in range(3):
            pred, losses, choice = cell(d, seed, fold, d["phi"], d["V"])
            test = np.flatnonzero(d["folds"][seed] == fold)
            ids = d["ids"][test]
            idx = {p: i for i, p in enumerate(ids)}
            diff = []
            for j in range(9):
                sub = arch[(arch.seed == seed) & (arch.fold == fold)]
                for _, r in sub[sub.drug == d["ids"][0][:0] if False else (sub.drug.notna())].iterrows():
                    pass
            # compare via drug index using DRUGS order
            DRUGS = ["5-Fluorouracil","Cisplatin","Cyclophosphamide","Docetaxel","Doxorubicin",
                     "Etoposide","Gemcitabine","Paclitaxel","Temozolomide"]
            for j, drug in enumerate(DRUGS):
                sub = arch[(arch.seed == seed) & (arch.fold == fold) & (arch.drug == drug)]
                ii = np.array([idx[p] for p in sub.patient_id])
                diff.append(float(np.max(np.abs(pred[ii, j] - sub.score.to_numpy(float)))))
            rows.append(dict(seed=seed, fold=fold, max_abs_vs_archive=max(diff)))
            print("done", seed, fold, max(diff), flush=True)
    print(pd.DataFrame(rows).to_string(index=False))
