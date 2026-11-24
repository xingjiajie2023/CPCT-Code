# -*- coding: utf-8 -*-
"""Shared, validated engine for CPCT revision experiments.

Reproduces the archived CPCT HARD readout bit-for-bit (max|diff|=1.1e-16).
All new experiments call `run_cell` with a modified source view phi and/or a
modified prior V while keeping the outer splits, inner folds, candidate grid,
weights and optimizer identical to the archived pipeline.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ABLC = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "cpct_ablation_final", "code"))
_SELC = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "cpct_select", "code"))
for _p in (_ABLC, _SELC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from common import fit_general, predict_general  # noqa: E402
from splits import patient_splits  # noqa: E402

ROOT = os.environ["DCIT_DATA_ROOT"]
CPCT = os.path.abspath(os.path.join(_HERE, os.pardir))
CPCT_OUT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "cpct_select", "outputs"))

CC = (0.0001, 0.001, 0.01)
CP = (0.0, 0.0001, 0.001)
CS = (0.0, 0.1, 1.0, 10.0)
GRID = [(a, b, c) for a in CC for b in CP for c in CS]
assert len(GRID) == 36

DRUGS = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", "Doxorubicin",
         "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]
SEEDS = (2024, 2025, 2026)
FOLDS = (0, 1, 2)


def load_data():
    a = np.load(os.path.join(ROOT, "dcit/cache/patients.npz"))
    return dict(x=a["X"].astype(float), y=a["Y"].astype(float), m=a["mask"].astype(bool),
                ids=a["patient_ids"],
                folds={s: a["folds_%d" % s] for s in SEEDS})


def load_archive_views():
    p = np.load(os.path.join(CPCT_OUT, "source/profiles.npz"))
    phi = np.column_stack([p["target_pan"], p["target_private"]])
    V = np.load(os.path.join(CPCT_OUT, "source/source_prior.npz"))["weights"].astype(float)
    return phi, V


def run_cell(d, seed, fold, phi, V, grid=GRID, inner_seed_base=50000, pool=None):
    """Full from-scratch CPCT/refit cell. Returns per-drug test predictions,
    inner loss table, chosen config index, and the achieved metrics.
    `pool` optionally restricts the HARD choice to a subset of grid indices
    (used for the source-free Clinical-MTL control: cs == 0)."""
    x, y, m = d["x"], d["y"], d["m"]
    train = np.flatnonzero(d["folds"][seed] != fold)
    test = np.flatnonzero(d["folds"][seed] == fold)
    G = len(grid)
    losses = np.zeros((G, 3, 9))
    for part, (it, iv) in enumerate(patient_splits(y[train], m[train], 3, inner_seed_base + seed + fold)):
        tr, va = train[it], train[iv]
        xs = StandardScaler().fit(x[tr]); xx = xs.transform(x)
        ps = StandardScaler().fit(phi[tr]); ss = ps.transform(phi)[:, None, :] * np.sqrt(V)[None, :, :]
        for ci, (cc, cp, cs) in enumerate(grid):
            model = fit_general(xx[tr], y[tr], m[tr], ss[tr], cc, cp, cs,
                                use_shared=True, use_private=True, use_source=True)
            pp = np.clip(predict_general(model, xx[va], ss[va]), 1e-12, 1 - 1e-12)
            yy = np.where(m[va], y[va], 0.)
            losses[ci, part] = ((-yy * np.log(pp) - (1 - yy) * np.log1p(-pp)) * m[va]).sum(axis=0) / m[va].sum(axis=0)
    mean_loss = losses.mean(axis=1)
    if pool is not None:
        mask = np.zeros(G, bool); mask[list(pool)] = True
        mean_loss = np.where(mask[:, None], mean_loss, np.inf)
    choice = mean_loss.argmin(axis=0)
    xs = StandardScaler().fit(x[train]); xx = xs.transform(x)
    ps = StandardScaler().fit(phi[train]); ss = ps.transform(phi)[:, None, :] * np.sqrt(V)[None, :, :]
    probs = {}
    for ci in np.unique(choice):
        cc, cp, cs = grid[ci]
        model = fit_general(xx[train], y[train], m[train], ss[train], cc, cp, cs,
                            use_shared=True, use_private=True, use_source=True)
        probs[int(ci)] = predict_general(model, xx[test], ss[test])
    pred = np.stack([probs[int(choice[j])][:, j] for j in range(9)], axis=1)
    return dict(seed=seed, fold=fold, test_index=test, pred=pred, losses=losses,
                choice=choice, grid=np.array(grid, float))


def cell_metrics(res, d):
    rows = []
    for j, drug in enumerate(DRUGS):
        te = res["test_index"]
        valid = d["m"][te, j]
        pp = res["pred"][valid, j]
        yy = d["y"][te, j][valid]
        rows.append(dict(seed=res["seed"], fold=res["fold"], drug=drug,
                         AUC=float(roc_auc_score(yy, pp)) if len(np.unique(yy)) > 1 else np.nan,
                         AUPR=float(average_precision_score(yy, pp)),
                         Brier=float(np.mean((pp - yy) ** 2)), n=int(valid.sum())))
    return rows


def patient_rows(res, d, method):
    te = res["test_index"]
    rows = []
    for j, drug in enumerate(DRUGS):
        for k, i in enumerate(te):
            if d["m"][i, j]:
                rows.append(dict(method=method, seed=res["seed"], fold=res["fold"],
                                 patient_id=d["ids"][i], drug=drug,
                                 label=int(d["y"][i, j]), score=float(res["pred"][k, j])))
    return rows
