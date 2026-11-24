"""Nested patient-CV for common/private clinical contrast transfer.

All choices use inner training patients. Global and drug-wise selection are
reported separately. The outer response labels are used for metrics only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,pickle,time
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
from model import fit_multitask,predict,observation_weights
from paths import ROOT,PROFILE_DIR,OUTPUT
from splits import patient_splits
from prior import source_prior_weights

COMMON=(.0001,.001,.01)
PRIVATE=(0.,.0001,.001)
SOURCE=(.1,1.,10.)
FAMILIES=('Clinical','Pan','Contrast')


def settings():
    values=[]
    for common in COMMON:
        for private in PRIVATE:
            values.append(('Clinical',common,private,0.))
            for family in ('Pan','Contrast'):
                for source in SOURCE:values.append((family,common,private,source))
    return values


def loss_by_drug(y,m,p):
    p=np.clip(p,1e-12,1-1e-12)
    losses=-(np.where(m,y,0.)*np.log(p)+(1-np.where(m,y,0.))*np.log1p(-p))
    return np.sum(m*losses,axis=0)/np.maximum(m.sum(axis=0),1)


def task(seed,fold,prior_file,out):
    start=time.time()
    data=np.load(ROOT/'dcit/cache/patients.npz')
    profile=np.load(PROFILE_DIR/'profiles.npz')
    prior=np.load(prior_file)['weights']
    x,y,m=data['X'].astype(float),data['Y'].astype(float),data['mask'].astype(bool)
    assert np.array_equal(data['patient_ids'],profile['patient_ids'])
    phi=np.column_stack([profile['target_pan'],profile['target_private']])
    train=np.flatnonzero(data['folds_'+str(seed)]!=fold)
    test=np.flatnonzero(data['folds_'+str(seed)]==fold)
    configs=settings()
    losses=np.full((len(configs),3,9),np.nan)
    weighted=np.full((len(configs),3),np.nan)
    for part,(it,iv) in enumerate(patient_splits(y[train],m[train],3,50000+seed+fold)):
        tr,va=train[it],train[iv]
        xs=StandardScaler().fit(x[tr]);xx=xs.transform(x)
        ps=StandardScaler().fit(phi[tr]);ss=ps.transform(phi)[:,None,:]*np.sqrt(prior)[None,:,:]
        for ci,(family,cc,cp,cs) in enumerate(configs):
            source=ss[:,:,:1] if family=='Pan' else ss
            model=fit_multitask(xx[tr],y[tr],m[tr],source[tr],cc,cp,cs)
            if not model['success'] and model['gradient']>1e-3:
                raise RuntimeError(('optimizer',seed,fold,part,configs[ci],model['gradient']))
            pp=predict(model,xx[va],source[va])
            losses[ci,part]=loss_by_drug(y[va],m[va],pp)
            pclip=np.clip(pp,1e-12,1-1e-12)
            point=np.logaddexp(0,np.log(pclip)-np.log1p(-pclip))-np.where(m[va],y[va],0.)*(np.log(pclip)-np.log1p(-pclip))
            ow=observation_weights(m[va])
            weighted[ci,part]=float(np.sum(ow*point)/ow.sum())
        print('inner_done',seed,fold,part,round(time.time()-start,1),flush=True)
    choices={};parameters=[]
    for family in FAMILIES:
        eligible=[i for i,c in enumerate(configs) if c[0] in ('Clinical',family)]
        for rule in ('global','drugwise'):
            objective=weighted.mean(axis=1)[:,None].repeat(9,axis=1) if rule=='global' else losses.mean(axis=1)
            choice=np.asarray(eligible)[objective[eligible].argmin(axis=0)]
            choices[family+'_'+rule]=choice
            for d,drug in enumerate(data['drug_names'].astype(str)):
                family_chosen,cc,cp,cs=configs[choice[d]]
                parameters.append({'method':family+'_'+rule,'seed':seed,'fold':fold,
                    'drug':drug,'family_selected':family_chosen,'common_C':cc,'private_C':cp,
                    'source_C':cs,'inner_logloss':float(objective[choice[d],d])})
    xs=StandardScaler().fit(x[train]);xx=xs.transform(x)
    ps=StandardScaler().fit(phi[train]);ss=ps.transform(phi)[:,None,:]*np.sqrt(prior)[None,:,:]
    models={};predictions={};diagnostics=[]
    for ci in sorted(set(np.concatenate(list(choices.values())))):
        family,cc,cp,cs=configs[ci]
        source=ss[:,:,:1] if family=='Pan' else ss
        model=fit_multitask(xx[train],y[train],m[train],source[train],cc,cp,cs)
        models[int(ci)]=model
        predictions[ci]=predict(model,xx[test],source[test])
        diagnostics.append({'seed':seed,'fold':fold,'config_index':int(ci),'family':family,
            'common_C':cc,'private_C':cp,'source_C':cs,
            **{key:model[key] for key in ['success','iterations','gradient','objective']}})
    artifact={'models':models,'choices':choices,'configs':configs,'gene_scaler':xs,
              'profile_scaler':ps,'source_prior':prior,'seed':seed,'fold':fold,
              'train_ids':data['patient_ids'][train],'test_ids':data['patient_ids'][test],
              'drug_names':data['drug_names']}
    target=Path(out)/'models';target.mkdir(exist_ok=True,parents=True)
    with (target/f's{seed}_f{fold}.pkl').open('wb') as stream:pickle.dump(artifact,stream)
    np.savez_compressed(target/f's{seed}_f{fold}_inner.npz',losses=losses,weighted=weighted)
    rows,raw=[],[]
    for method,choice in choices.items():
        for d,drug in enumerate(data['drug_names'].astype(str)):
            good=m[test,d];pp=predictions[choice[d]][:,d]
            yy=y[test,d][good]
            rows.append({'method':method,'seed':seed,'fold':fold,'drug':drug,
                'AUC':roc_auc_score(yy,pp[good]),'AUPR':average_precision_score(yy,pp[good])})
            for i in np.flatnonzero(good):raw.append({'method':method,'seed':seed,'fold':fold,
                'patient_id':str(data['patient_ids'][test[i]]),'drug':drug,'label':int(y[test[i],d]),'score':float(pp[i])})
    return rows,raw,parameters,diagnostics


def main(seeds,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(out/'source_prior.npz',weights=source_prior_weights())
    (out/'config.json').write_text(json.dumps({'COMMON':COMMON,'PRIVATE':PRIVATE,'SOURCE':SOURCE,
        'source_zero_included':True,'inner_folds':3,'outer_folds':3,'seeds':seeds,
        'global_choice':'patient-exposure/task normalized held-out logloss',
        'drugwise_choice':'per-drug held-out logloss, no outer-label choice',
        'target_expression_scope_source_views':'all patients, transductive, no response labels',
        'status':'development CV; cohort has been used for method development'},indent=2))
    rows,raw,parameters,diag=[],[],[],[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        todo={pool.submit(task,s,f,str(out/'source_prior.npz'),str(out)):(s,f) for s in seeds for f in range(3)}
        for future in as_completed(todo):
            a,b,c,d=future.result();rows+=a;raw+=b;parameters+=c;diag+=d
            for name,records in [('per_fold',rows),('per_patient',raw),('parameters',parameters),('optimizer',diag)]:
                pd.DataFrame(records).to_csv(out/(name+'.csv'),index=False)
            print('complete',todo[future],flush=True)
    ss=pd.DataFrame(rows).groupby(['method','seed','drug'])[['AUC','AUPR']].mean().reset_index()
    ss.to_csv(out/'per_seed.csv',index=False)
    print(ss.groupby(['drug','method']).AUC.mean().unstack().round(4).to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=[2024,2025,2026])
    parser.add_argument('--out',default=str(OUTPUT/'nested'))
    args=parser.parse_args();main(args.seeds,args.out)
