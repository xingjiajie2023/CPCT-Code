"""Source-only learning of pan-response and drug-specific response contrasts.

Representation/penalty selection uses only fixed source validation responses.
The shared principal-vector view uses unlabeled patient expression (transductive,
as permitted by the existing protocol). Clinical response labels are never read.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,pickle
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from paths import ROOT,PROFILE_DIR
OUT=PROFILE_DIR
ALPHAS=(.1,1.,10.,100.,1000.,10000.)


def nuisance_basis(u,mean,scale):
    z=np.clip((u-mean)/scale,-4.,4.)
    return np.column_stack([np.ones(len(z)),z,z*z,z*z*z])


def prepare():
    a=np.load(ROOT/'dcit/cache/patients.npz')
    s=np.load(ROOT/'benchmark_dcit/cache/source_raw_v2.npz')
    b=np.load(ROOT/'dcit/cache/source_latents.npz')
    r=np.load(ROOT/'representations/transdrp.npz')
    assert np.array_equal(r['tgt_ids'],a['patient_ids'])
    assert np.array_equal(r['src_ids'],s['cell_ids'])
    assert np.array_equal(r['source_train_mask'],s['train_mask'])
    train=s['train_mask'].astype(bool)
    source_pan=logit(np.clip(r['src_native_score'].astype(float).mean(axis=1),1e-6,1-1e-6))
    target_pan=logit(np.clip(r['tgt_native_score'].astype(float).mean(axis=1),1e-6,1-1e-6))
    pan_mean,pan_scale=source_pan[train].mean(),source_pan[train].std()
    hs=nuisance_basis(source_pan,pan_mean,pan_scale)
    ht=nuisance_basis(target_pan,pan_mean,pan_scale)
    features={}
    preprocessing={}
    for name,sx,tx in [('gene',s['X'],a['X']),('base',b['B'],a['B']),
                      ('UDA',r['src_repr'],r['tgt_repr'])]:
        scaler=StandardScaler().fit(sx[train])
        features[name]=(scaler.transform(sx),scaler.transform(tx))
        preprocessing[name]=scaler
    # Aligned source/target principal directions; source scale is learned from
    # source training cells, target scale from expression only, no response.
    source_scaler=StandardScaler().fit(s['X'][train])
    target_scaler=StandardScaler().fit(a['X'])
    xs=source_scaler.transform(s['X'])
    xt=target_scaler.transform(a['X'])
    ps=PCA(100,svd_solver='randomized',random_state=2024).fit(xs[train])
    pt=PCA(100,svd_solver='randomized',random_state=2024).fit(xt)
    u,singular,vh=np.linalg.svd(ps.components_@pt.components_.T)
    ws,wt=ps.components_.T@u[:,:20],pt.components_.T@vh.T[:,:20]
    cs,ct=xs@ws,xt@wt
    ss=StandardScaler().fit(cs[train]);st=StandardScaler().fit(ct)
    features['PV20']=(ss.transform(cs),st.transform(ct))
    preprocessing['PV20']={'source_scaler':source_scaler,'target_scaler':target_scaler,
        'source_projection':ws,'target_projection':wt,'source_coord_scaler':ss,'target_coord_scaler':st}
    preprocessing['pan_mean']=pan_mean;preprocessing['pan_scale']=pan_scale
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'preprocessing.pkl').open('wb') as stream:pickle.dump(preprocessing,stream)
    with (OUT/'prepared.pkl').open('wb') as stream:
        pickle.dump({'features':features,'hs':hs,'ht':ht,'target':-s['Z'].astype(float),
                     'mask':s['mask'].astype(bool),'train':train,'source_pan':source_pan,
                     'target_pan':target_pan,'patient_ids':a['patient_ids'],
                     'source_ids':s['cell_ids'],'drug_names':s['drug_names']},stream)
    (OUT/'config.json').write_text(json.dumps({'source_fold':0,'source_penalties':ALPHAS,
        'source_views':list(features),'source_target':'negative continuous Z_SCORE',
        'source_nuisance':'cubic functions of common native susceptibility score',
        'uses_patient_response':False,'PV20':'source/target expression-only principal-vector alignment',
        'target_expression_scope':'all 1186 patients, no clinical labels'},indent=2))


def task(j):
    with (OUT/'prepared.pkl').open('rb') as stream:a=pickle.load(stream)
    y=a['target'][:,j]
    train=a['train']&a['mask'][:,j]
    valid=(~a['train'])&a['mask'][:,j]
    hs,ht=a['hs'],a['ht']
    nuisance=np.linalg.lstsq(hs[train],y[train],rcond=None)[0]
    residual=y-hs@nuisance
    baseline_source,baseline_target=hs@nuisance,ht@nuisance
    candidates=[];best=None
    for name,(sx,tx) in a['features'].items():
        nuisance_x=np.linalg.lstsq(hs[train],sx[train],rcond=None)[0]
        rx,rt=sx-hs@nuisance_x,tx-ht@nuisance_x
        x=rx[train]
        gram=x@x.T
        eig,u=np.linalg.eigh((gram+gram.T)/2)
        proj=u.T@residual[train]
        cross=rx[valid]@x.T
        for alpha in ALPHAS:
            dual=u@(proj/(np.maximum(eig,0)+alpha))
            predicted=cross@dual
            mse=float(np.mean((residual[valid]-predicted)**2))
            entry={'drug_index':j,'view':name,'alpha':alpha,'source_validation_MSE':mse,
                   'residual_R2':float(r2_score(residual[valid],predicted)),
                   'total_R2':float(r2_score(y[valid],baseline_source[valid]+predicted))}
            candidates.append(entry)
            if best is None or mse<best[0]:
                coef=x.T@dual
                best=(mse,rx@coef,rt@coef,entry.copy(),coef,nuisance_x)
    best[3]['pan_only_R2']=float(r2_score(y[valid],baseline_source[valid]))
    best[3]['drug']=str(a['drug_names'][j])
    fitted={'view':best[3]['view'],'alpha':best[3]['alpha'],'coef':best[4],
            'nuisance_x':best[5],'nuisance_y':nuisance}
    return j,best[1],best[2],baseline_source,baseline_target,best[3],candidates,fitted


def main():
    prepare()
    with (OUT/'prepared.pkl').open('rb') as stream:a=pickle.load(stream)
    d=len(a['drug_names'])
    source_private=np.zeros((len(a['source_ids']),d))
    target_private=np.zeros((len(a['patient_ids']),d))
    source_base,target_base=np.zeros_like(source_private),np.zeros_like(target_private)
    selected,allrows=[],[]
    heads={}
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(task,j) for j in range(d)]
        for t in as_completed(tasks):
            j,ss,tt,bs,bt,selection,rows,fitted=t.result()
            heads[j]=fitted
            source_private[:,j],target_private[:,j]=ss,tt
            source_base[:,j],target_base[:,j]=bs,bt
            selected.append(selection);allrows+=rows
            print(selection,flush=True)
    with (OUT/'source_heads.pkl').open('wb') as stream:pickle.dump(heads,stream)
    np.savez_compressed(OUT/'profiles.npz',source_private=source_private,target_private=target_private,
        source_total=source_base+source_private,target_total=target_base+target_private,
        source_baseline=source_base,target_baseline=target_base,source_pan=a['source_pan'],
        target_pan=a['target_pan'],patient_ids=a['patient_ids'],source_ids=a['source_ids'],
        drug_names=a['drug_names'])
    pd.DataFrame(selected).sort_values('drug_index').to_csv(OUT/'source_validation.csv',index=False)
    pd.DataFrame(allrows).to_csv(OUT/'all_source_candidates.csv',index=False)
    from prior import source_prior_weights
    np.savez_compressed(OUT/'source_prior.npz',weights=source_prior_weights(force=True))
    print('profiles_saved',str(OUT/'profiles.npz'),flush=True)


if __name__=='__main__':main()
