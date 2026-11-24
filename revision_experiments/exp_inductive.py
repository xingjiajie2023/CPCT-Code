# -*- coding: utf-8 -*-
"""P0: strict inductive generalization (CPCT vs Clinical-MTL, plus transductive ref).

For each outer (seed,fold) the source views phi are rebuilt using ONLY the outer
training patients for every patient-expression fit.  Test patients never enter
any transform fitting / alignment / adaptation; labels used only for metrics.
CPCT = full 36-candidate pool + HARD readout; Clinical-MTL = source-free 9-cand
pool (cs=0).  Transductive reference = same engine on the archived all-patient
phi (recomputed here, not taken from archive, so the delta isolates the leak).
"""
import os, sys, time, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev_engine as E
from rev_source import SourceCtx, SEEDS, DRUGS

OUT = os.path.join(HERE, os.pardir, "source_data")


def build_all_phis():
    ctx = SourceCtx()
    d = E.load_data()
    phis = {}
    metas = []
    for seed in SEEDS:
        for fold in range(3):
            train = np.flatnonzero(d["folds"][seed] != fold)
            phi, meta = ctx.build_phi(train)
            phis[(seed, fold)] = phi
            for m in meta:
                metas.append(dict(seed=seed, fold=fold, **m))
    np.savez_compressed(os.path.join(OUT, "inductive_phi_folds.npz"),
                       **{"phi_%d_%d" % (s, f): phis[(s, f)] for s in SEEDS for f in range(3)})
    pd.DataFrame(metas).to_csv(os.path.join(OUT, "inductive_source_selection.csv"), index=False)
    ref = np.column_stack([ctx.p["target_pan"], ctx.p["target_private"]])
    diffs = [float(np.max(np.abs(phis[(s, f)] - ref))) for s in SEEDS for f in range(3)]
    print("inductive phi built; max|diff| vs transductive phi = %.4g" % max(diffs), flush=True)


def _worker(args):
    seed, fold, mode = args
    d = E.load_data()
    if mode == "transductive":
        phi, V = E.load_archive_views()
        pool = None
        method = "CPCT_transductive"
    elif mode == "clinical":
        phi, V = E.load_archive_views()
        pool = [i for i, g in enumerate(E.GRID) if g[2] == 0]
        method = "Clinical_MTL"
    else:
        z = np.load(os.path.join(OUT, "inductive_phi_folds.npz"))
        phi = z["phi_%d_%d" % (seed, fold)]
        V = E.load_archive_views()[1]
        pool = None
        method = "CPCT_inductive"
    res = E.run_cell(d, seed, fold, phi, V, pool=pool)
    rows = E.cell_metrics(res, d)
    prows = E.patient_rows(res, d, method)
    sel = [dict(seed=seed, fold=fold, method=method, drug=DRUGS[j],
                common_C=E.GRID[int(res["choice"][j])][0],
                private_C=E.GRID[int(res["choice"][j])][1],
                source_C=E.GRID[int(res["choice"][j])][2],
                source_on=bool(E.GRID[int(res["choice"][j])][2] > 0))
           for j in range(9)]
    return rows, prows, sel


def summarize(fold_rows):
    df = pd.DataFrame(fold_rows)
    mean3 = df.groupby(["method", "seed", "drug"])[["AUC", "AUPR", "Brier"]].mean().reset_index()
    final = mean3.groupby(["method", "drug"])[["AUC", "AUPR", "Brier"]].mean().reset_index()
    return final


def main():
    t0 = time.time()
    if not os.path.exists(os.path.join(OUT, "inductive_phi_folds.npz")):
        build_all_phis()
    tasks = [(s, f, m) for m in ("transductive", "clinical", "inductive")
             for s in SEEDS for f in range(3)]
    allrows, allpat, allsel = [], [], []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fu in as_completed(futs):
            r, p, sel = fu.result()
            allrows += r; allpat += p; allsel += sel
            print("done", futs[fu], round(time.time() - t0, 1), "s", flush=True)
    pd.DataFrame(allrows).to_csv(os.path.join(OUT, "inductive_per_fold.csv"), index=False)
    pd.DataFrame(allpat).to_csv(os.path.join(OUT, "inductive_per_patient.csv"), index=False)
    pd.DataFrame(allsel).to_csv(os.path.join(OUT, "inductive_selected_configs.csv"), index=False)
    final = summarize(allrows)
    final.to_csv(os.path.join(OUT, "inductive_per_drug.csv"), index=False)
    # difference table: inductive - transductive and inductive - clinical
    piv = final.pivot(index="drug", columns="method", values=["AUC", "AUPR", "Brier"])
    out = pd.DataFrame(index=DRUGS)
    for metric in ("AUC", "AUPR", "Brier"):
        out[metric + "_transductive"] = piv[(metric, "CPCT_transductive")]
        out[metric + "_inductive"] = piv[(metric, "CPCT_inductive")]
        out[metric + "_delta_ind_minus_trans"] = piv[(metric, "CPCT_inductive")] - piv[(metric, "CPCT_transductive")]
        out[metric + "_ClinicalMTL"] = piv[(metric, "Clinical_MTL")]
        out[metric + "_delta_ind_minus_clinical"] = piv[(metric, "CPCT_inductive")] - piv[(metric, "Clinical_MTL")]
    out.to_csv(os.path.join(OUT, "inductive_differences.csv"))
    print(out[["AUC_transductive", "AUC_inductive", "AUC_delta_ind_minus_trans",
               "AUC_ClinicalMTL", "AUC_delta_ind_minus_clinical"]].round(4).to_string(), flush=True)
    print("P0 DONE seconds=", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
