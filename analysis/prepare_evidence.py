"""Derive manuscript evidence from recorded predictions and selected models."""
from pathlib import Path
import argparse,json,pickle,hashlib
import numpy as np,pandas as pd
from scipy.special import logit,expit

DRUGS=['5-Fluorouracil','Cisplatin','Cyclophosphamide','Docetaxel','Doxorubicin','Etoposide','Gemcitabine','Paclitaxel','Temozolomide']
SHORT=['5-FU','CIS','CTX','DOC','DOX','ETO','GEM','PAC','TMZ']
BASELINES=['AITL','TRANSACT-Clin','TUGDA-Clin','CODE-AE-ADV-Clin','TransDRP-Clin','TCRP','TMEDRP-Clin','TrAdaBoost']


def main(args):
    root=Path(args.cpct_root);rawroot=Path(args.data_root);out=Path(args.out)
    out.mkdir(exist_ok=True,parents=True)
    data=np.load(rawroot/'dcit/cache/patients.npz')
    source=np.load(rawroot/'benchmark_dcit/cache/source_raw_v2.npz')
    profile=np.load(root/'outputs/source/profiles.npz')
    rows=[]
    for d,drug in enumerate(DRUGS):
        m=data['mask'][:,d].astype(bool);positive=int(np.sum(data['Y'][m,d]))
        rows.append({'drug':drug,'short':SHORT[d],'source_observed':int(source['mask'][:,d].sum()),
            'source_train':int(np.sum(source['mask'][:,d]&source['train_mask'])),
            'patients':int(m.sum()),'positive':positive,'negative':int(m.sum())-positive,
            'positive_rate':positive/m.sum()})
    pd.DataFrame(rows).to_csv(out/'cohort.csv',index=False)
    frames=[]
    for metric in ['AUC','AUPR']:
        t=pd.read_csv(root/f'outputs/tables/all_methods_{metric}.csv')
        t=t.set_index('drug').loc[DRUGS]
        t=t[BASELINES+['Clinical_drugwise','Contrast_drugwise']]
        t.rename(columns={'Clinical_drugwise':'Clinical-MTL','Contrast_drugwise':'CPCT'}).to_csv(out/f'main_{metric}.csv')
    seed=pd.read_csv(root/'outputs/nested/per_seed.csv')
    fold=pd.read_csv(root/'outputs/nested/per_fold.csv')
    params=pd.read_csv(root/'outputs/nested/parameters.csv')
    for name,t in [('main_per_seed',seed),('main_per_fold',fold),('selection',params)]:t.to_csv(out/(name+'.csv'),index=False)
    source_validation=pd.read_csv(root/'outputs/source/source_validation.csv')
    source_validation.to_csv(out/'source_validation.csv',index=False)
    ablation=pd.read_csv(Path(args.ablation_root)/'per_seed.csv')
    ablation.to_csv(out/'ablation_per_seed.csv',index=False)
    pd.read_csv(Path(args.ablation_root)/'per_fold.csv').to_csv(out/'ablation_per_fold.csv',index=False)
    prediction=pd.read_csv(root/'outputs/nested/per_patient.csv')
    brier=prediction.assign(Brier=(prediction.score-prediction.label)**2).groupby(['method','seed','fold','drug']).Brier.mean().reset_index()
    brier.to_csv(out/'brier_per_fold.csv',index=False)
    # Conditional minimum validation profiles; no outer performance is used.
    cfg=[]
    for cc in [.0001,.001,.01]:
        for cp in [0.,.0001,.001]:
            cfg.append(('Clinical',cc,cp,0.))
            for family in ['Pan','Contrast']:
                for cs in [.1,1.,10.]:cfg.append((family,cc,cp,cs))
    eligible=[i for i,c in enumerate(cfg) if c[0] in ['Clinical','Contrast']]
    profiles=[]
    for path in sorted((root/'outputs/nested/models').glob('*_inner.npz')):
        split_seed=int(path.stem.split('_')[0][1:]);f=int(path.stem.split('_')[1][1:])
        loss=np.load(path)['losses'].mean(axis=1)
        for label,index,values in [('common_C',1,[.0001,.001,.01]),('private_C',2,[0.,.0001,.001]),('source_C',3,[0.,.1,1.,10.])]:
            for d,drug in enumerate(DRUGS):
                for v in values:
                    ids=[i for i in eligible if cfg[i][index]==v]
                    profiles.append({'seed':split_seed,'fold':f,'drug':drug,'parameter':label,
                                     'value':v,'inner_logloss':float(np.min(loss[ids,d]))})
    pd.DataFrame(profiles).to_csv(out/'parameter_profiles.csv',index=False)
    # Source terms of the same fitted CPCT model, not a refitted clinical model.
    with (root/'outputs/nested/models/s2024_f0.pkl').open('rb') as stream:model=pickle.load(stream)
    phi=np.column_stack([profile['target_pan'],profile['target_private']])
    ss=model['profile_scaler'].transform(phi)[:,None,:]*np.sqrt(model['source_prior'])[None,:,:]
    idmap={str(v):i for i,v in enumerate(data['patient_ids'])}
    chosen=[];distribution=[]
    for drug in ['Cyclophosphamide','Gemcitabine']:
        d=DRUGS.index(drug);ci=int(model['choices']['Contrast_drugwise'][d]);fitted=model['models'][ci]
        rows=prediction[(prediction.method=='Contrast_drugwise')&(prediction.seed==2024)&(prediction.fold==0)&(prediction.drug==drug)].copy()
        ix=np.array([idmap[v] for v in rows.patient_id.astype(str)])
        effect=np.zeros(len(rows)) if fitted['source'] is None else ss[ix,d]@fitted['source'][d]
        rows['source_term']=effect;rows['clinical_term']=logit(np.clip(rows.score,1e-10,1-1e-10))-effect
        for number,(_,row) in enumerate(rows.sort_values('patient_id').iterrows()):
            distribution.append({'drug':drug,'case_index':number,'label':int(row.label),
                'source_term':row.source_term,'clinical_term':row.clinical_term,'score':row.score})
        for label in [0,1]:
            subset=rows[rows.label==label].copy()
            subset['amplitude']=np.abs(subset.source_term)
            subset=subset.sort_values(['amplitude','patient_id'])
            row=subset.iloc[(len(subset)-1)//2]
            chosen.append({'drug':drug,'case':('C' if drug.startswith('Cyclo') else 'G')+str(label),
                'label':int(label),'clinical_term':float(row.clinical_term),'source_term':float(row.source_term),
                'score':float(row.score),'source_zero_score':float(expit(row.clinical_term)),
                'seed':2024,'fold':0,'selection_rule':'lower median absolute source term within recorded class'})
    pd.DataFrame(chosen).to_csv(out/'selected_cases.csv',index=False)
    pd.DataFrame(distribution).to_csv(out/'case_source_distribution.csv',index=False)
    # Small tables have a fixed drug order; no cross-drug averages are written.
    tex=Path(args.tex_out);tex.mkdir(exist_ok=True,parents=True)
    cohort=pd.read_csv(out/'cohort.csv')
    lines=[]
    for _,r in cohort.iterrows():lines.append(f"{r['short']} & {r.source_observed} & {r.source_train} & {r.patients} & {r.positive} & {r.negative} & {r.positive_rate:.3f} \\\\")
    (tex/'cohort_rows.tex').write_text('\n'.join(lines)+'\n')
    labels={'Clinical-MTL':'Clinical-MTL','CPCT':r'\textbf{CPCT}','CODE-AE-ADV-Clin':'CODE-AE-ADV-C','TRANSACT-Clin':'TRANSACT-C','TUGDA-Clin':'TUGDA-C','TransDRP-Clin':'TransDRP-C','TMEDRP-Clin':'TMEDRP-C'}
    for metric in ['AUC','AUPR']:
        t=pd.read_csv(out/f'main_{metric}.csv').set_index('drug')
        lines=[]
        for method in t.columns:
            numbers=[]
            for drug in DRUGS:
                value=t.loc[drug,method];text=f'{value:.4f}'
                if value==t.loc[drug].max():text=r'\textbf{'+text+'}'
                numbers.append(text)
            lines.append(labels.get(method,method)+' & '+' & '.join(numbers)+r' \\')
        (tex/f'main_{metric}_rows.tex').write_text('\n'.join(lines)+'\n')
    macros=[]
    for name,filename in [('cohortrows','cohort_rows.tex'),('aucrows','main_AUC_rows.tex'),('auprrows','main_AUPR_rows.tex')]:
        macros.append('\\newcommand{\\'+name+'}{%\n'+(tex/filename).read_text().rstrip()+'\n}\n')
    (tex/'data_macros.tex').write_text('\n'.join(macros))
    metadata={'source_cells':len(source['X']),'source_train':int(source['train_mask'].sum()),
        'source_validation':int((~source['train_mask']).sum()),'patients':len(data['X']),
        'genes':data['X'].shape[1],'observed_records':int(data['mask'].sum()),
        'abstract_budget_words':200,'main_seed_count':3,'outer_folds':3,'inner_folds':3,
        'new_ablation_names':['NoShared','TotalViews','NoRelations'],
        'case_selection':'two prespecified drugs; seed 2024/fold 0; class-specific median |source term|',
        'parameter_plot_scope':'conditional-minimum inner validation loss, not outer-test sensitivity'}
    (out/'evidence_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Evidence prepared:',metadata)
    print('Cases:\n'+pd.DataFrame(chosen).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpct-root',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--ablation-root',required=True);p.add_argument('--out',required=True);p.add_argument('--tex-out',required=True)
    main(p.parse_args())
