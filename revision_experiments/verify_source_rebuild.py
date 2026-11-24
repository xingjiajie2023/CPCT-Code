# -*- coding: utf-8 -*-
"""Rebuild source residual views from raw arrays + archived preprocessing and
confirm the selected view/alpha and residual_R2 reproduce source_validation.csv.

This is the basis for the cross-fitting experiment: once the source ridge
pipeline is reproduced, we can re-run view/alpha selection on a fraction of the
source validation cells and estimate reliability on a disjoint outer fold.
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.special import logit
from sklearn.metrics import r2_score

ROOT = os.environ["DCIT_DATA_ROOT"]
CPCT_OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        os.pardir, os.pardir, "cpct_select", "outputs"))
PRE = os.path.join(CPCT_OUT, "source", "preprocessing.pkl")
ALPHAS = (0.1, 1., 10., 100., 1000., 10000.)


def nuisance_basis(u, mean, scale):
    z = np.clip((u - mean) / scale, -4., 4.)
    return np.column_stack([np.ones(len(z)), z, z * z, z * z * z])


def build():
    a = np.load(os.path.join(ROOT, "dcit/cache/patients.npz"))
    s = np.load(os.path.join(ROOT, "benchmark_dcit/cache/source_raw_v2.npz"))
    b = np.load(os.path.join(ROOT, "dcit/cache/source_latents.npz"))
    r = np.load(os.path.join(ROOT, "representations/transdrp.npz"))
    with open(PRE, "rb") as f:
        pre = pickle.load(f)
    train = s["train_mask"].astype(bool)
    source_pan = logit(np.clip(r["src_native_score"].astype(float).mean(axis=1), 1e-6, 1 - 1e-6))
    target_pan = logit(np.clip(r["tgt_native_score"].astype(float).mean(axis=1), 1e-6, 1 - 1e-6))
    hs = nuisance_basis(source_pan, pre["pan_mean"], pre["pan_scale"])
    ht = nuisance_basis(target_pan, pre["pan_mean"], pre["pan_scale"])
    gsc = pre["gene"]
    bsc = pre["base"]
    usc = pre["UDA"]
    gene = (gsc.transform(s["X"]), gsc.transform(a["X"]))
    base = (bsc.transform(b["B"]), bsc.transform(a["B"]))
    uda = (usc.transform(r["src_repr"]), usc.transform(r["tgt_repr"]))
    pv = pre["PV20"]
    xs = pv["source_scaler"].transform(s["X"])
    xt = pv["target_scaler"].transform(a["X"])
    cs = xs @ pv["source_projection"]
    ct = xt @ pv["target_projection"]
    pvfeat = (pv["source_coord_scaler"].transform(cs), pv["target_coord_scaler"].transform(ct))
    features = {"gene": gene, "base": base, "UDA": uda, "PV20": pvfeat}
    return dict(features=features, hs=hs, ht=ht, target=-s["Z"].astype(float),
                mask=s["mask"].astype(bool), train=train, drug_names=s["drug_names"])


def select(d, j, val_mask=None, alphas=ALPHAS):
    """Reproduce source.py task(j) view/alpha selection; return best row dict."""
    y = d["target"][:, j]
    train = d["train"] & d["mask"][:, j]
    valid = (val_mask if val_mask is not None else (~d["train"])) & d["mask"][:, j]
    hs, ht = d["hs"], d["ht"]
    nuisance = np.linalg.lstsq(hs[train], y[train], rcond=None)[0]
    residual = y - hs @ nuisance
    baseline_source = hs @ nuisance
    best = None
    rows = []
    for name, (sx, tx) in d["features"].items():
        nx = np.linalg.lstsq(hs[train], sx[train], rcond=None)[0]
        rx, rt = sx - hs @ nx, tx - ht @ nx
        x = rx[train]
        gram = x @ x.T
        eig, u = np.linalg.eigh((gram + gram.T) / 2)
        proj = u.T @ residual[train]
        cross = rx[valid] @ x.T
        for alpha in alphas:
            dual = u @ (proj / (np.maximum(eig, 0) + alpha))
            predicted = cross @ dual
            mse = float(np.mean((residual[valid] - predicted) ** 2))
            row = dict(drug_index=j, view=name, alpha=alpha, source_validation_MSE=mse,
                       residual_R2=float(r2_score(residual[valid], predicted)),
                       total_R2=float(r2_score(y[valid], baseline_source[valid] + predicted)))
            rows.append(row)
            if best is None or mse < best[0]:
                best = (mse, row.copy())
    best[1]["pan_only_R2"] = float(r2_score(y[valid], baseline_source[valid]))
    best[1]["drug"] = str(d["drug_names"][j])
    return best[1], rows


if __name__ == "__main__":
    d = build()
    ref = pd.read_csv(os.path.join(CPCT_OUT, "source", "source_validation.csv")).sort_values("drug_index")
    out = []
    for j in range(9):
        best, rows = select(d, j)
        r = ref.iloc[j]
        out.append(dict(j=j, view=best["view"], ref_view=r["view"],
                        alpha=best["alpha"], ref_alpha=float(r["alpha"]),
                        r2=best["residual_R2"], ref_r2=float(r["residual_R2"]),
                        pan=best["pan_only_R2"], ref_pan=float(r["pan_only_R2"])))
    print(pd.DataFrame(out).to_string(index=False))
