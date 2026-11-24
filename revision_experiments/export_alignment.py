# -*- coding: utf-8 -*-
"""Export PV20 shared-subspace coordinates (before/after) for the alignment plot.
Writes a small npz into source_data/.  All values computed from raw project data.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from rev_source import SourceCtx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, os.pardir, "source_data"))

ctx = SourceCtx()
allidx = np.arange(len(ctx.a["patient_ids"]))
X = ctx.a["X"]
ts = StandardScaler().fit(X[allidx]); xt = ts.transform(X)
pt = PCA(100, svd_solver="randomized", random_state=2024).fit(xt[allidx])
u, _, vh = np.linalg.svd(ctx.ps.components_ @ pt.components_.T)
ws = ctx.ps.components_.T @ u[:, :20]
wt = pt.components_.T @ vh.T[:, :20]
cs = ctx.xs @ ws; ct = xt @ wt
ss = StandardScaler().fit(cs[ctx.str]); st = StandardScaler().fit(ct[allidx])
cs_z = ss.transform(cs); ct_z = st.transform(ct)
# BEFORE: both domains in the SAME gene space (one shared gene scaler fit on
# source expression), reduced by a PCA fit on the pooled gene-scaled data.
# Using one common coordinate system is essential for an honest before/after.
gene_s = ctx.gene_s   # (715, D) pre['gene'].transform(source X)
gene_t = ctx.gene_t   # (1186, D) SAME scaler applied to target X
pg = PCA(20, svd_solver="randomized", random_state=2024).fit(np.vstack([gene_s, gene_t]))
bef_s = pg.transform(gene_s)
bef_t = pg.transform(gene_t)

# per-drug response-like native scores (frozen, no patient fitting) for coloring
src_native = ctx.r["src_native_score"].astype(float).mean(axis=1)   # (715,)
tgt_native = ctx.r["tgt_native_score"].astype(float).mean(axis=1)   # (1186,)

np.savez_compressed(
    os.path.join(DATA, "alignment_coords.npz"),
    source_before=bef_s, target_before=bef_t,
    source_raw=cs, target_raw=ct,
    source_aligned=cs_z, target_aligned=ct_z,
    src_native=src_native, tgt_native=tgt_native,
    source_ids=ctx.s_ids if hasattr(ctx, "s_ids") else ctx.p["source_ids"],
    patient_ids=ctx.a["patient_ids"],
    drug_names=ctx.a["drug_names"],
)
print("saved alignment_coords.npz")
print("shapes", bef_s.shape, bef_t.shape, cs_z.shape, ct_z.shape)
