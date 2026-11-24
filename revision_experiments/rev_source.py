# -*- coding: utf-8 -*-
"""Fold-specific source-view construction for the strict-inductive experiment.

Every transform that touches patient expression (target scaler, target PCA,
PV20 alignment, target coordinate scaler) is fit ONLY on the outer training
patients.  Source-side fitting uses source cells only.  Frozen native scores
(target_pan) involve no patient-data fitting.  The selected UDA view is never
used by any drug, so the frozen transductive domain-adapted representation does
not enter CPCT's final source views.

When train = all patients, this reproduces the archived target_private exactly.
"""
import os, sys, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from scipy.special import logit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

ROOT = os.environ["DCIT_DATA_ROOT"]
_HERE = os.path.dirname(os.path.abspath(__file__))
CPCT_OUT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, "cpct_select", "outputs"))
ALPHAS = (0.1, 1., 10., 100., 1000., 10000.)
SEEDS = (2024, 2025, 2026)
DRUGS = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", "Doxorubicin",
         "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]


def nuisance_basis(u, mean, scale):
    z = np.clip((u - mean) / scale, -4., 4.)
    return np.column_stack([np.ones(len(z)), z, z * z, z * z * z])


class SourceCtx:
    def __init__(self):
        self.a = np.load(os.path.join(ROOT, "dcit/cache/patients.npz"))
        self.s = np.load(os.path.join(ROOT, "benchmark_dcit/cache/source_raw_v2.npz"))
        self.b = np.load(os.path.join(ROOT, "dcit/cache/source_latents.npz"))
        self.r = np.load(os.path.join(ROOT, "representations/transdrp.npz"))
        with open(os.path.join(CPCT_OUT, "source/preprocessing.pkl"), "rb") as f:
            self.pre = pickle.load(f)
        with open(os.path.join(CPCT_OUT, "source/source_heads.pkl"), "rb") as f:
            self.heads = pickle.load(f)
        self.p = np.load(os.path.join(CPCT_OUT, "source/profiles.npz"))
        self.str = self.s["train_mask"].astype(bool)
        self.hs = nuisance_basis(logit(np.clip(self.r["src_native_score"].astype(float).mean(axis=1), 1e-6, 1 - 1e-6)),
                                 self.pre["pan_mean"], self.pre["pan_scale"])
        self.ht = nuisance_basis(logit(np.clip(self.r["tgt_native_score"].astype(float).mean(axis=1), 1e-6, 1 - 1e-6)),
                                 self.pre["pan_mean"], self.pre["pan_scale"])
        # source-only fixed views
        pre = self.pre
        self.gene_s, self.gene_t = pre["gene"].transform(self.s["X"]), pre["gene"].transform(self.a["X"])
        self.base_s, self.base_t = pre["base"].transform(self.b["B"]), pre["base"].transform(self.a["B"])
        self.uda_s, self.uda_t = pre["UDA"].transform(self.r["src_repr"]), pre["UDA"].transform(self.r["tgt_repr"])
        xs = pre["PV20"]["source_scaler"].transform(self.s["X"])
        self.xs = xs
        self.ps = PCA(100, svd_solver="randomized", random_state=2024).fit(xs[self.str])

    def fold_features(self, train_idx):
        X = self.a["X"]
        ts = StandardScaler().fit(X[train_idx])
        xt = ts.transform(X)
        pt = PCA(100, svd_solver="randomized", random_state=2024).fit(xt[train_idx])
        u, _, vh = np.linalg.svd(self.ps.components_ @ pt.components_.T)
        ws = self.ps.components_.T @ u[:, :20]
        wt = pt.components_.T @ vh.T[:, :20]
        cs = self.xs @ ws
        ct = xt @ wt
        ss = StandardScaler().fit(cs[self.str])
        st = StandardScaler().fit(ct[train_idx])
        return {"gene": (self.gene_s, self.gene_t),
                "base": (self.base_s, self.base_t),
                "UDA": (self.uda_s, self.uda_t),
                "PV20": (ss.transform(cs), st.transform(ct))}

    def build_phi(self, train_idx):
        feats = self.fold_features(train_idx)
        y = -self.s["Z"].astype(float)
        m = self.s["mask"].astype(bool)
        target_private = np.zeros((len(self.a["patient_ids"]), 9))
        meta = []
        for j in range(9):
            tr = self.str & m[:, j]
            va = (~self.str) & m[:, j]
            yj = y[:, j]
            nuis = np.linalg.lstsq(self.hs[tr], yj[tr], rcond=None)[0]
            resid = yj - self.hs @ nuis
            best = None
            for name, (sx, tx) in feats.items():
                nx = np.linalg.lstsq(self.hs[tr], sx[tr], rcond=None)[0]
                rx, rt = sx - self.hs @ nx, tx - self.ht @ nx
                x = rx[tr]
                gram = x @ x.T
                eig, uu = np.linalg.eigh((gram + gram.T) / 2)
                proj = uu.T @ resid[tr]
                cross = rx[va] @ x.T
                for alpha in ALPHAS:
                    dual = uu @ (proj / (np.maximum(eig, 0) + alpha))
                    pred = cross @ dual
                    mse = float(np.mean((resid[va] - pred) ** 2))
                    if best is None or mse < best[0]:
                        coef = x.T @ dual
                        best = (mse, rt @ coef, name, alpha)
            target_private[:, j] = best[1]
            meta.append(dict(drug=DRUGS[j], view=best[2], alpha=best[3], mse=best[0]))
        phi = np.column_stack([self.p["target_pan"], target_private])
        return phi, meta


if __name__ == "__main__":
    ctx = SourceCtx()
    allidx = np.arange(len(ctx.a["patient_ids"]))
    phi_all, meta = ctx.build_phi(allidx)
    ref = np.column_stack([ctx.p["target_pan"], ctx.p["target_private"]])
    print("SANITY train=ALL: phi max|diff| =", float(np.max(np.abs(phi_all - ref))))
    print(pd_meta := "\n".join(str(m) for m in meta))
