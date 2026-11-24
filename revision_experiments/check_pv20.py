# -*- coding: utf-8 -*-
"""Check that PV20 source/target projections are reproducible from raw arrays."""
import os, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = os.environ["DCIT_DATA_ROOT"]
PRE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, "cpct_select", "outputs", "source", "preprocessing.pkl"))
a = np.load(os.path.join(ROOT, "dcit/cache/patients.npz"))
s = np.load(os.path.join(ROOT, "benchmark_dcit/cache/source_raw_v2.npz"))
with open(PRE, "rb") as f:
    pre = pickle.load(f)
pv = pre["PV20"]
tr = s["train_mask"].astype(bool)
source_scaler = StandardScaler().fit(s["X"][tr])
xs = source_scaler.transform(s["X"])
target_scaler = StandardScaler().fit(a["X"])
xt = target_scaler.transform(a["X"])
ps = PCA(100, svd_solver="randomized", random_state=2024).fit(xs[tr])
pt = PCA(100, svd_solver="randomized", random_state=2024).fit(xt)
u, sing, vh = np.linalg.svd(ps.components_ @ pt.components_.T)
ws = ps.components_.T @ u[:, :20]
wt = pt.components_.T @ vh.T[:, :20]
print("source_proj max|diff|", np.max(np.abs(ws - pv["source_projection"])))
print("target_proj max|diff|", np.max(np.abs(wt - pv["target_projection"])))
print("source_scaler mean maxdiff", np.max(np.abs(source_scaler.mean_ - pv["source_scaler"].mean_)))
print("target_scaler mean maxdiff", np.max(np.abs(target_scaler.mean_ - pv["target_scaler"].mean_)))
