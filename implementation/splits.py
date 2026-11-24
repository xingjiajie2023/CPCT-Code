import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
def patient_splits(y,mask,folds,seed):
    strata=np.clip(((y==1)&mask).sum(axis=1),0,2)
    counts=np.unique(strata,return_counts=True)[1]
    if counts.min()>=folds:
        return list(StratifiedKFold(folds,shuffle=True,random_state=seed).split(np.zeros(len(y)),strata))
    return list(KFold(folds,shuffle=True,random_state=seed).split(y))
