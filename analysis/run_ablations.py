"""Matched development ablations; original main results are read-only.

NoShared removes the shared clinical coefficient block. TotalViews substitutes
reconstructed total source responses for residual-response views. NoRelations
removes drug-conditioned residual correlations while preserving each drug's
total private-source prior variance and source-quality weighting.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,sys,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score


def task(seed,fold,variant,data_root,cpct_root):
    sys.path.insert(0,cpct_root)
    from model import fit_multitask,predict
    from splits import patient_splits
    a=np.load(Path(data_root)/'dcit/cache/patients.npz')
    p=np.load(Path(cpct_root)/'outputs/source/profiles.npz')
    prior=np.load(Path(cpct_root)/'outputs/source/source_prior.npz')['weights'].copy()
    x,y,m=a['X'].astype(float),a['Y'].astype(float),a['mask'].astype(bool)
    private=p['target_total'] if variant=='TotalViews' else p['target_private']
    phi=np.column_stack([p['target_pan'],private])
    if variant=='NoRelations':
        val=pd.read_csv(Path(cpct_root)/'outputs/source/source_validation.csv').sort_values('drug_index')
        r=np.clip(val.residual_R2.to_numpy(),0,.95);q=r/(1-r)
        total=prior[:,1:].sum(axis=1,keepdims=True)
        prior[:,1:]=q[None,:]*total/q.sum()
    train=np.flatnonzero(a['folds_'+str(seed)]!=fold)
    test=np.flatnonzero(a['folds_'+str(seed)]==fold)
    cc_grid=(0.,) if variant=='NoShared' else (.0001,.001,.01)
    grid=[(cc,cp,cs) for cc in cc_grid for cp in (0.,.0001,.001) for cs in (0.,.1,1.,10.)]
    losses=np.zeros((len(grid),3,9))
    for part,(it,iv) in enumerate(patient_splits(y[train],m[train],3,50000+seed+fold)):
        tr,va=train[it],train[iv]
        xs=StandardScaler().fit(x[tr]);xx=xs.transform(x)
        ps=StandardScaler().fit(phi[tr]);ss=ps.transform(phi)[:,None,:]*np.sqrt(prior)[None,:,:]
        for ci,(cc,cp,cs) in enumerate(grid):
            model=fit_multitask(xx[tr],y[tr],m[tr],ss[tr],cc,cp,cs)
            if not model['success'] and model['gradient']>1e-3:raise RuntimeError((variant,seed,fold,part,model['gradient']))
            pp=np.clip(predict(model,xx[va],ss[va]),1e-12,1-1e-12)
            yy=np.where(m[va],y[va],0.)
            point=-yy*np.log(pp)-(1-yy)*np.log1p(-pp)
            losses[ci,part]=(point*m[va]).sum(axis=0)/m[va].sum(axis=0)
    choice=losses.mean(axis=1).argmin(axis=0)
    xs=StandardScaler().fit(x[train]);xx=xs.transform(x)
    ps=StandardScaler().fit(phi[train]);ss=ps.transform(phi)[:,None,:]*np.sqrt(prior)[None,:,:]
    probabilities={}
    for ci in np.unique(choice):
        cc,cp,cs=grid[ci]
        model=fit_multitask(xx[train],y[train],m[train],ss[train],cc,cp,cs)
        probabilities[ci]=predict(model,xx[test],ss[test])
    rows,raw,params=[],[],[]
    for d,drug in enumerate(a['drug_names'].astype(str)):
        good=m[test,d];pp=probabilities[choice[d]][:,d];yy=y[test,d][good]
        rows.append({'method':variant,'seed':seed,'fold':fold,'drug':drug,
                     'AUC':roc_auc_score(yy,pp[good]),'AUPR':average_precision_score(yy,pp[good])})
        cc,cp,cs=grid[choice[d]]
        params.append({'method':variant,'seed':seed,'fold':fold,'drug':drug,
                       'common_C':cc,'private_C':cp,'source_C':cs})
        for i in np.flatnonzero(good):raw.append({'method':variant,'seed':seed,'fold':fold,
            'patient_id':str(a['patient_ids'][test[i]]),'drug':drug,'label':int(y[test[i],d]),'score':float(pp[i])})
    return rows,raw,params


def main(args):
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    variants=['NoShared','TotalViews','NoRelations']
    (out/'config.json').write_text(json.dumps({'variants':variants,'seeds':[2024,2025,2026],
        'outer_folds':3,'inner_folds':3,'source_prior_trace_preserved_NoRelations':True,
        'main_results_retrained':False,'status':'additional development ablations for manuscript'},indent=2))
    rows,raw,params=[],[],[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        todo={pool.submit(task,s,f,v,args.data_root,args.cpct_root):(v,s,f)
              for v in variants for s in (2024,2025,2026) for f in range(3)}
        for future in as_completed(todo):
            a,b,c=future.result();rows+=a;raw+=b;params+=c
            for name,items in [('per_fold',rows),('per_patient',raw),('parameters',params)]:
                pd.DataFrame(items).to_csv(out/(name+'.csv'),index=False)
            print('completed',todo[future],flush=True)
    summary=pd.DataFrame(rows).groupby(['method','seed','drug'])[['AUC','AUPR']].mean().reset_index()
    summary.to_csv(out/'per_seed.csv',index=False)
    print(summary.groupby(['drug','method']).AUC.mean().unstack().round(4).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True)
    p.add_argument('--cpct-root',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
