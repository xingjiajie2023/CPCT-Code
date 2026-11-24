# -*- coding: utf-8 -*-
"""Exp 2: source-domain reliability cross-fitting.

Problem: the archived source validation set is used twice -- to select the
residual view / Ridge alpha AND to estimate residual prediction reliability
(residual_R2 -> eta) and pan reliability (pan_only_R2 -> kappa) in the prior.

Fix: K-fold cross-fitting over the source validation cells of each drug.
  inner : select view + alpha by MSE on validation \\ fold k
  outer : estimate residual_R2 and pan_only_R2 on the held-out fold k
  cross-fitted eta/kappa = mean held-out R2 over folds
The relation rho_de is recovered unchanged from the archived prior, so the ONLY
change is the reliability estimates.  Clinical phi, outer splits, inner folds,
grid, weights and optimizer are unchanged.
"""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_engine as E
from verify_source_rebuild import build, select
from rev_source import DRUGS

OUT = os.path.join(HERE, os.pardir, "source_data")
K = 3


def build_crossfit_prior():
    d = build()
    phi, V_orig = E.load_archive_views()
    sv = pd.read_csv(os.path.join(E.CPCT_OUT, "source/source_validation.csv")).sort_values("drug_index")
    eta_orig = np.clip(sv["residual_R2"].to_numpy(float), 0, .95)
    eta_orig = eta_orig / np.maximum(1 - eta_orig, 1e-8)
    rel = V_orig[:, 1:] / np.maximum(eta_orig[None, :], 1e-12)   # = 0.1 + 0.9 rho
    rho = (rel - 0.1) / 0.9

    rows = []
    r2_cf = np.zeros(9)
    pan_cf = np.zeros(9)
    for j in range(9):
        src = d
        y = src["target"][:, j]
        tr = src["train"] & src["mask"][:, j]
        va = np.flatnonzero((~src["train"]) & src["mask"][:, j])
        hs, ht = src["hs"], src["ht"]
        nuis_y = np.linalg.lstsq(hs[tr], y[tr], rcond=None)[0]
        resid = y - hs @ nuis_y
        base_src = hs @ nuis_y

        # precompute per-view transformed features and ridge machinery on source train
        views = {}
        for name, (sx, tx) in src["features"].items():
            nx = np.linalg.lstsq(hs[tr], sx[tr], rcond=None)[0]
            rx, rt = sx - hs @ nx, tx - ht @ nx
            x = rx[tr]
            gram = x @ x.T
            eig, uu = np.linalg.eigh((gram + gram.T) / 2)
            proj = uu.T @ resid[tr]
            views[name] = dict(rx=rx, rt=rt, x=x, eig=eig, uu=uu, proj=proj)

        kf = KFold(n_splits=K, shuffle=True, random_state=9000 + j)
        held_r2, held_pan = [], []
        for itr, ite in kf.split(va):
            inner, outer = va[itr], va[ite]
            # inner: select view/alpha by MSE on inner part of validation
            best = None
            for name, vw in views.items():
                cross_in = vw["rx"][inner] @ vw["x"].T
                for alpha in (0.1, 1., 10., 100., 1000., 10000.):
                    dual = vw["uu"] @ (vw["proj"] / (np.maximum(vw["eig"], 0) + alpha))
                    pred_in = cross_in @ dual
                    mse = float(np.mean((resid[inner] - pred_in) ** 2))
                    if best is None or mse < best[0]:
                        best = (mse, name, alpha, dual, vw["rx"])
            _, name, alpha, dual, rx_best = best
            pred_out = (rx_best[outer] @ rx_best[tr].T) @ dual
            held_r2.append(float(r2_score(resid[outer], pred_out)))
            held_pan.append(float(r2_score(y[outer], base_src[outer])))
            rows.append(dict(drug=DRUGS[j], fold=len(held_r2) - 1, n_inner=len(inner), n_outer=len(outer),
                             selected_view=name, alpha=alpha, residual_R2_outer=held_r2[-1],
                             pan_R2_outer=held_pan[-1]))
        r2_cf[j] = float(np.mean(held_r2))
        pan_cf[j] = float(np.mean(held_pan))

    eta_cf = np.clip(r2_cf, 0, .95) / np.maximum(1 - np.clip(r2_cf, 0, .95), 1e-8)
    kappa_cf = np.clip(pan_cf, 0, .95) / np.maximum(1 - np.clip(pan_cf, 0, .95), 1e-8)
    V_cf = np.empty_like(V_orig)
    V_cf[:, 0] = kappa_cf
    V_cf[:, 1:] = (0.1 + 0.9 * rho) * eta_cf[None, :]

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "crossfit_source_folds.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "crossfit_prior.npz"), weights=V_cf,
                        residual_R2=r2_cf, pan_R2=pan_cf, eta=eta_cf, kappa=kappa_cf)
    cmp = pd.DataFrame(dict(drug=DRUGS,
                            residual_R2_orig=sv["residual_R2"].to_numpy(float),
                            residual_R2_crossfit=r2_cf,
                            pan_R2_orig=sv["pan_only_R2"].to_numpy(float),
                            pan_R2_crossfit=pan_cf))
    cmp.to_csv(os.path.join(OUT, "crossfit_reliability_comparison.csv"), index=False)
    print(cmp.round(4).to_string(index=False), flush=True)
    return V_cf


def _worker(args):
    seed, fold, tag = args
    d = E.load_data()
    phi, V_orig = E.load_archive_views()
    if tag == "crossfit":
        V = np.load(os.path.join(OUT, "crossfit_prior.npz"))["weights"]
    else:
        V = V_orig
    res = E.run_cell(d, seed, fold, phi, V)
    rows = E.cell_metrics(res, d)
    for r in rows:
        r["method"] = "CPCT_crossfit" if tag == "crossfit" else "CPCT_original"
    return rows


def main():
    t0 = time.time()
    build_crossfit_prior()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = [(s, f, tag) for tag in ("original", "crossfit") for s in E.SEEDS for f in E.FOLDS]
    rows = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fu in as_completed(futs):
            rows += fu.result()
            print("done", futs[fu], round(time.time() - t0, 1), "s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "crossfit_per_fold.csv"), index=False)
    final = df.groupby(["method", "seed", "drug"])[["AUC", "AUPR", "Brier"]].mean().reset_index()
    final = final.groupby(["method", "drug"])[["AUC", "AUPR", "Brier"]].mean().reset_index()
    final.to_csv(os.path.join(OUT, "crossfit_per_drug.csv"), index=False)
    print("EXP2 DONE seconds=", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
