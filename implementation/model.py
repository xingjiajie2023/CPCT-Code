"""Patient-grouped clinical multi-task models with source response deviations."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,pickle,time
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score
from paths import ROOT,PROFILE_DIR,OUTPUT
from prior import source_prior_weights


def observation_weights(mask):
    w=mask/np.maximum(mask.sum(axis=1,keepdims=True),1)
    w/=np.maximum(w.sum(axis=0,keepdims=True),1e-12)
    w*=len(mask)/mask.shape[1]
    return w


def fit_multitask(x,y,mask,source,c_common=.001,c_private=0.,c_source=0.,start=None):
    n,p=x.shape;d=y.shape[1];k=source.shape[-1]
    yy=np.where(mask,y,0.)
    w=observation_weights(mask)
    xx=x*np.sqrt(c_common)
    xp=x*np.sqrt(c_private)
    ss=source*np.sqrt(c_source)
    size=d+p+(p*d if c_private else 0)+(k*d if c_source else 0)
    initial=np.zeros(size)
    rates=np.sum(w*yy,axis=0)/np.maximum(w.sum(axis=0),1e-12)
    initial[:d]=logit(np.clip(rates,1e-5,1-1e-5))
    def decode(t):
        b=t[:d];shared=t[d:d+p];offset=d+p
        private=t[offset:offset+p*d].reshape(p,d) if c_private else None
        if c_private:offset+=p*d
        gamma=t[offset:].reshape(d,k) if c_source else None
        return b,shared,private,gamma

    def value_grad(t):
        b,shared,private,gamma=decode(t)
        f=b[None,:]+(xx@shared)[:,None]
        if c_private:f=f+xp@private
        if c_source:f=f+np.einsum('ndk,dk->nd',ss,gamma)
        ll=np.sum(w*(np.logaddexp(0,f)-yy*f))+.5*(t[d:]@t[d:])
        r=w*(expit(f)-yy)
        parts=[r.sum(axis=0),xx.T@r.sum(axis=1)+shared]
        if c_private:parts.append((xp.T@r+private).reshape(-1))
        if c_source:parts.append((np.einsum('ndk,nd->dk',ss,r)+gamma).reshape(-1))
        return float(ll),np.concatenate(parts)

    result=minimize(value_grad,initial,method='L-BFGS-B',jac=True,
                    options={'maxiter':400,'ftol':1e-10,'gtol':1e-5,'maxls':30})
    b,shared,private,gamma=decode(result.x)
    return {'b':b,'shared':shared*np.sqrt(c_common),
            'private':None if private is None else private*np.sqrt(c_private),
            'source':None if gamma is None else gamma*np.sqrt(c_source),
            'success':bool(result.success),'iterations':result.nit,
            'gradient':float(np.max(np.abs(result.jac))),'objective':float(result.fun)}


def predict(model,x,source):
    f=model['b'][None,:]+(x@model['shared'])[:,None]
    if model['private'] is not None:f=f+x@model['private']
    if model['source'] is not None:f=f+np.einsum('ndk,dk->nd',source,model['source'])
    return expit(f)


def task(seed,fold,weight_file,out):
    a=np.load(ROOT/'dcit/cache/patients.npz')
    p=np.load(PROFILE_DIR/'profiles.npz')
    w=np.load(weight_file)['weights']
    x,y,m=a['X'].astype(float),a['Y'].astype(float),a['mask'].astype(bool)
    train,test=np.flatnonzero(a['folds_'+str(seed)]!=fold),np.flatnonzero(a['folds_'+str(seed)]==fold)
    xs=StandardScaler().fit(x[train]);xx=xs.transform(x)
    phi=np.column_stack([p['target_pan'],p['target_private']])
    scale=StandardScaler().fit(phi[train]);phi=scale.transform(phi)
    source=phi[:,None,:]*np.sqrt(w)[None,:,:]
    models={'Clinical_shared':(0.,0.),'Clinical_MTL':(.001,0.),
            'Source_MTL':(0.,1.),'Hybrid_MTL':(.0001,1.),'Pan_MTL':(0.,1.),
            'HybridPan_MTL':(.0001,1.),'Matched_MTL':(.0001,1.)}
    rows,raw,diag=[],[],[]
    fitted={}
    for method,(cp,cs) in models.items():
        if method in ('Pan_MTL','HybridPan_MTL'):input_source=source[:,:,:1]
        elif method=='Matched_MTL':
            input_source=np.stack([source[:,d,[0,d+1]] for d in range(9)],axis=1)
        else:input_source=source
        fit=fit_multitask(xx[train],y[train],m[train],input_source[train],
                          c_common=.001,c_private=cp,c_source=cs)
        pred=predict(fit,xx[test],input_source[test])
        fitted[method]=fit
        diag.append({'method':method,'seed':seed,'fold':fold,
            **{k:fit[k] for k in ['success','iterations','gradient','objective']}})
        for j,drug in enumerate(a['drug_names'].astype(str)):
            valid=m[test,j]
            rows.append({'method':method,'seed':seed,'fold':fold,'drug':drug,
                'AUC':roc_auc_score(y[test,j][valid],pred[valid,j]),
                'AUPR':average_precision_score(y[test,j][valid],pred[valid,j])})
            for i in np.flatnonzero(valid):raw.append({'method':method,'seed':seed,'fold':fold,
                'patient_id':a['patient_ids'][test[i]],'drug':drug,'label':int(y[test[i],j]),'score':pred[i,j]})
    destination=Path(out)/'models';destination.mkdir(exist_ok=True,parents=True)
    with (destination/f's{seed}_f{fold}.pkl').open('wb') as stream:
        pickle.dump({'models':fitted,'gene_scaler':xs,'source_scaler':scale,
                     'source_prior':w,'train_ids':a['patient_ids'][train],
                     'test_ids':a['patient_ids'][test],'drug_names':a['drug_names']},stream)
    return rows,raw,diag


def main(seeds,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'source_prior.npz',weights=source_prior_weights())
    (out/'config.json').write_text(json.dumps({'common_C':.001,'private_C':.001,
        'hybrid_private_C':.0001,'source_C':1.,'source_prior':'source-quality conditioned response graph',
        'loss_weight':'inverse exposure count, task normalization, total equals patient count',
        'seeds':seeds,'status':'fixed-setting development probe'},indent=2))
    rows,raw,diags=[],[],[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks={pool.submit(task,s,f,str(out/'source_prior.npz'),str(out)):(s,f) for s in seeds for f in range(3)}
        for t in as_completed(tasks):
            aa,bb,cc=t.result();rows+=aa;raw+=bb;diags+=cc
            pd.DataFrame(rows).to_csv(out/'per_fold.csv',index=False)
            pd.DataFrame(raw).to_csv(out/'per_patient.csv',index=False)
            pd.DataFrame(diags).to_csv(out/'optimizer.csv',index=False)
            print('completed',tasks[t],flush=True)
    ss=pd.DataFrame(rows).groupby(['method','seed','drug'])[['AUC','AUPR']].mean().reset_index()
    ss.to_csv(out/'per_seed.csv',index=False)
    print(ss.groupby(['drug','method']).AUC.mean().unstack().round(4).to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=[2024])
    parser.add_argument('--out',default=str(OUTPUT/'fixed'))
    a=parser.parse_args();main(a.seeds,a.out)
