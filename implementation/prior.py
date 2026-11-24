import pickle
import numpy as np,pandas as pd
from paths import PROFILE_DIR
def source_prior_weights(force=False):
    cached=PROFILE_DIR/'source_prior.npz'
    if cached.exists() and not force:
        return np.load(cached)['weights']
    with (PROFILE_DIR/'prepared.pkl').open('rb') as stream:a=pickle.load(stream)
    n,d=a['target'].shape
    residual=np.zeros((n,d))
    for j in range(d):
        tr=a['train']&a['mask'][:,j]
        coef=np.linalg.lstsq(a['hs'][tr],a['target'][tr,j],rcond=None)[0]
        residual[:,j]=a['target'][:,j]-a['hs']@coef
    relation=np.eye(d)
    for j in range(d):
        for k in range(j):
            valid=a['train']&a['mask'][:,j]&a['mask'][:,k]
            rho=np.corrcoef(residual[valid,j],residual[valid,k])[0,1]
            relation[j,k]=relation[k,j]=rho*rho
    val=pd.read_csv(PROFILE_DIR/'source_validation.csv').sort_values('drug_index')
    r=np.clip(val['residual_R2'].to_numpy(),0,.95);q=r/np.maximum(1-r,1e-8)
    private=(.1+.9*relation)*q[None,:]
    pan=np.clip(val['pan_only_R2'].to_numpy(),0,.95);pan=pan/np.maximum(1-pan,1e-8)
    return np.column_stack([pan,private])
