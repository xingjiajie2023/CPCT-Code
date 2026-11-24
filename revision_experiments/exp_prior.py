# -*- coding: utf-8 -*-
"""Exp 3: drug-relation prior sensitivity.

Mappings of the source residual correlation r_de -> rho_de in [0,1]:
  square      : rho = r^2                      (archived method, control)
  absolute    : rho = |r|
  pos_trunc   : rho = r^2 if r>0 else 0        (positive-correlation truncation)
  keep_sign   : rho = (1+r)/2                  (direction-preserving non-negative map)
Prior lower bound lambda in v_de = (lambda + (1-lambda) rho_de) eta_e:
  lambda in {0, 0.05, 0.1, 0.2}                (0.1 = archived control)
kappa (common-source column) and eta are recovered unchanged from the archive.
Everything else (phi, splits, inner selection, grid, optimizer) is fixed.
"""
import os, sys, time, itertools
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_engine as E
from verify_source_rebuild import build
from rev_source import DRUGS

OUT = os.path.join(HERE, os.pardir, "source_data")
MAPPINGS = ("square", "absolute", "pos_trunc", "keep_sign")
LAMBDAS = (0.0, 0.05, 0.1, 0.2)


def signed_correlation(d):
    tr = d["train"]
    resid = np.zeros((len(d["target"]), 9))
    hs = d["hs"]
    for j in range(9):
        t = tr & d["mask"][:, j]
        nuis = np.linalg.lstsq(hs[t], d["target"][t, j], rcond=None)[0]
        resid[:, j] = d["target"][:, j] - hs @ nuis
    R = np.eye(9)
    for j in range(9):
        for k in range(j):
            t = tr & d["mask"][:, j] & d["mask"][:, k]
            R[j, k] = R[k, j] = np.corrcoef(resid[t, j], resid[t, k])[0, 1]
    return R


def build_all_priors():
    d = build()
    phi, V_orig = E.load_archive_views()
    R = signed_correlation(d)
    rel_orig = V_orig[:, 1:]                 # (0.1 + 0.9 r^2) * eta
    # V[d,e] = (0.1 + 0.9 r2[d,e]) * eta[e]  -- eta indexed by SOURCE drug e (column)
    r2 = R ** 2
    eta = np.mean(rel_orig / (0.1 + 0.9 * r2), axis=0)
    kappa = V_orig[:, 0].copy()
    Vrec = np.column_stack([kappa, (0.1 + 0.9 * r2) * eta[None, :]])
    print("prior reconstruction max|diff| =", float(np.max(np.abs(Vrec - V_orig))), flush=True)

    configs = {}
    rows = []
    for mp in MAPPINGS:
        if mp == "square":
            rho = R ** 2
        elif mp == "absolute":
            rho = np.abs(R)
        elif mp == "pos_trunc":
            rho = np.where(R > 0, R ** 2, 0.0)
        else:
            rho = (1 + R) / 2
        for lam in LAMBDAS:
            V = np.column_stack([kappa, (lam + (1 - lam) * rho) * eta[:, None]])
            key = "%s_lam%.2f" % (mp, lam)
            configs[key] = V
            rows.append(dict(config=key, mapping=mp, lam=lam,
                             prior_min_private=float(V[:, 1:].min()),
                             prior_max_private=float(V[:, 1:].max()),
                             prior_mean_private=float(V[:, 1:].mean()),
                             prior_common_sum=float(V[:, 0].sum())))
            np.savez_compressed(os.path.join(OUT, "prior_config_%s.npz" % key), weights=V)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "prior_config_summary.csv"), index=False)
    np.save(os.path.join(OUT, "prior_correlation_matrix.npy"), R)
    return configs


def _worker(args):
    seed, fold, key = args
    d = E.load_data()
    phi, _ = E.load_archive_views()
    V = np.load(os.path.join(OUT, "prior_config_%s.npz" % key))["weights"]
    res = E.run_cell(d, seed, fold, phi, V)
    sel = [dict(config=key, seed=seed, fold=fold, drug=DRUGS[j],
                source_on=bool(E.GRID[int(res["choice"][j])][2] > 0),
                common_C=E.GRID[int(res["choice"][j])][0],
                private_C=E.GRID[int(res["choice"][j])][1],
                source_C=E.GRID[int(res["choice"][j])][2]) for j in range(9)]
    rows = E.cell_metrics(res, d)
    for r in rows:
        r["config"] = key
    return rows, sel


def main():
    t0 = time.time()
    configs = build_all_priors()
    tasks = [(s, f, k) for k in configs for s in E.SEEDS for f in E.FOLDS]
    rows, sels = [], []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs)):
            r, sel = fu.result(); rows += r; sels += sel
            if (i + 1) % 24 == 0:
                print("done", i + 1, "/", len(tasks), round(time.time() - t0, 1), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "prior_sensitivity_per_fold.csv"), index=False)
    final = df.groupby(["seed", "config", "drug"])[["AUC", "AUPR", "Brier"]].mean().reset_index()
    final = final.groupby(["config", "drug"])[["AUC", "AUPR", "Brier"]].mean().reset_index()
    final.to_csv(os.path.join(OUT, "prior_sensitivity_per_drug.csv"), index=False)
    pd.DataFrame(sels).to_csv(os.path.join(OUT, "prior_sensitivity_selection.csv"), index=False)
    print("EXP3 DONE configs=%d seconds=%d" % (len(configs), round(time.time() - t0, 1)), flush=True)


if __name__ == "__main__":
    main()
