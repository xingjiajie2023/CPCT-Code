"""Publication figures from recorded evidence; no synthetic observations."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Patch
from matplotlib.colors import LinearSegmentedColormap
from pub_style import apply_publication_style,save_pub_set

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source_data';OUT=ROOT/'figures'
DRUGS=['5-Fluorouracil','Cisplatin','Cyclophosphamide','Docetaxel','Doxorubicin','Etoposide','Gemcitabine','Paclitaxel','Temozolomide']
SHORT=['5-FU','CIS','CTX','DOC','DOX','ETO','GEM','PAC','TMZ']
TEAL='#4D8F92';BLUE='#668EB1';GOLD='#C8A16C';GREY='#687581';LIGHT='#E8EDF0'
GREEN='#6E9C88';RED='#C47A79';INK='#29343A'
DIV=LinearSegmentedColormap.from_list('effect',[RED,'#FAFAFA',GREEN])
SEQ=LinearSegmentedColormap.from_list('validation',['#FAFBFC','#C6DADD','#5F9197'])


def style():
    apply_publication_style(font_size=7,axes_linewidth=.7)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,
        'legend.fontsize':6,'axes.titlesize':7,'savefig.facecolor':'white',
        'font.sans-serif':['DejaVu Sans','Liberation Sans'],'axes.unicode_minus':False})


def panel(ax,letter):
    ax.text(-.12,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=8,
            va='top',ha='left',color=INK,clip_on=False)


def export(fig,name):
    return save_pub_set(fig,OUT,name,formats=('svg','pdf','tiff','png'),dpi=600,bbox_inches=None)


def overview():
    fig=plt.figure(figsize=(122/25.4,66/25.4));ax=fig.add_axes([.015,.02,.97,.96]);ax.set_axis_off()
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.015',
                     facecolor=color,edgecolor='#819095',linewidth=.65))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=6.6,color=INK,linespacing=1.3)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.8,color='#66767C'))
    box(.02,.75,.20,.17,'Cell lines\nGene expression\nDrug sensitivity','#EDF1F4')
    box(.28,.75,.40,.17,'Common sensitivity + residual response\nRemove common response trend\nFit drug-specific source predictors','#E6F0F0')
    box(.75,.75,.22,.17,'Frozen source views\nCommon factor $u$\n'+r'Residuals $q_1,\ldots,q_9$','#E3ECEF')
    arrow((.23,.835),(.27,.835));arrow((.69,.835),(.74,.835))
    box(.02,.38,.20,.20,'Patients (training)\nExpression\nObserved responses','#F2EEE7')
    ax.add_patch(FancyBboxPatch((.28,.32),.69,.31,boxstyle='round,pad=.008,rounding_size=.012',
                 facecolor='#FBFCFC',edgecolor='#97A5A7',linewidth=.7))
    ax.text(.625,.605,'Joint common-private clinical predictor',ha='center',fontsize=7,fontweight='bold',color=INK)
    box(.30,.41,.19,.13,'Shared clinical\n'+r'$x^\top\beta$','#E9EEF2')
    box(.53,.41,.19,.13,'Drug-private\n'+r'$x^\top\delta_d$','#F1EADF')
    box(.76,.41,.19,.13,'Source response\n'+r'$\phi_d^\top\gamma_d$','#DDECEE')
    ax.text(.625,.351,'Weighted masked likelihood + separate regularization',ha='center',fontsize=6.4,color=INK)
    arrow((.23,.48),(.29,.48));arrow((.86,.74),(.86,.55))
    box(.28,.08,.40,.13,'Inner grouped validation\nDrug-wise model selection\nZero transfer permitted','#F1F3F4')
    box(.75,.08,.22,.13,'Refit selected models\nOuter-fold prediction\nfor all nine drugs','#E2ECE8')
    arrow((.625,.31),(.48,.22));arrow((.69,.145),(.74,.145))
    ax.text(.02,.985,'Common-private response contrast transfer',fontsize=8,fontweight='bold',color=INK)
    export(fig,'fig1_framework')


def robustness():
    folds=pd.read_csv(DATA/'main_per_fold.csv')
    brier=pd.read_csv(DATA/'brier_per_fold.csv')
    fig,axes=plt.subplots(1,2,figsize=(122/25.4,73/25.4),sharey=True)
    fig.subplots_adjust(left=.15,right=.98,bottom=.18,top=.87,wspace=.28)
    markers={2024:'o',2025:'s',2026:'^'}
    left=folds[folds.method=='Contrast_drugwise'].merge(
        folds[folds.method=='Clinical_drugwise'],on=['seed','fold','drug'],suffixes=('_full','_clinical'))
    right=brier[brier.method=='Contrast_drugwise'].merge(
        brier[brier.method=='Clinical_drugwise'],on=['seed','fold','drug'],suffixes=('_full','_clinical'))
    for j,drug in enumerate(DRUGS):
        a=left[left.drug==drug].sort_values(['seed','fold'])
        for k,(_,r) in enumerate(a.iterrows()):
            axes[0].scatter(r.AUC_full-r.AUC_clinical,j+(k-4)*.055,s=12,
                marker=markers[int(r.seed)],facecolor=TEAL,edgecolor='white',lw=.25,alpha=.75)
        b=right[right.drug==drug].groupby('seed')[['Brier_full','Brier_clinical']].mean()
        for k,(seed,r) in enumerate(b.iterrows()):
            axes[1].scatter(r.Brier_clinical-r.Brier_full,j+(k-1)*.15,s=16,
                marker=markers[int(seed)],facecolor=BLUE,edgecolor='white',lw=.3)
    for ax in axes:
        ax.axvline(0,color='#9DA7AA',lw=.7,ls='--',zorder=0)
        ax.set_ylim(8.5,-.6);ax.set_yticks(range(9),SHORT);ax.grid(axis='y',color='#F0F2F3',lw=.5)
    axes[0].set_xlabel(r'$\Delta$AUROC: CPCT - clinical')
    axes[1].set_xlabel('Brier improvement: clinical - CPCT')
    axes[0].set_title('Paired outer-fold discrimination',pad=9)
    axes[1].set_title('Out-of-fold probability accuracy',pad=9)
    axes[0].set_xlim(min(-.08,left.eval('AUC_full-AUC_clinical').min()-.01),
                     max(.10,left.eval('AUC_full-AUC_clinical').max()+.01))
    panel(axes[0],'a');panel(axes[1],'b')
    handles=[plt.Line2D([0],[0],marker=markers[s],color='none',markerfacecolor=GREY,
                        markeredgecolor='none',markersize=4,label=str(s)) for s in markers]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.005),ncol=3,frameon=False)
    export(fig,'fig2_robustness')


def ablations():
    main=pd.read_csv(DATA/'main_per_seed.csv');ab=pd.read_csv(DATA/'ablation_per_seed.csv')
    full=main[main.method=='Contrast_drugwise'].set_index(['seed','drug'])
    matrix=[]
    for name in ['NoShared','TotalViews','NoRelations']:
        other=ab[ab.method==name].set_index(['seed','drug'])
        delta=(full.AUC-other.AUC).groupby('drug').mean().reindex(DRUGS)
        matrix.append(delta.to_numpy())
    matrix=np.asarray(matrix)
    fig=plt.figure(figsize=(122/25.4,104/25.4));gs=fig.add_gridspec(2,2,height_ratios=[.72,1.35])
    fig.subplots_adjust(left=.16,right=.91,bottom=.11,top=.91,hspace=.72,wspace=.35)
    ax=fig.add_subplot(gs[0,:]);lim=max(abs(matrix.min()),abs(matrix.max()))
    im=ax.imshow(matrix,aspect='auto',cmap=DIV,vmin=-lim,vmax=lim)
    ax.set_xticks(range(9),SHORT);ax.set_yticks(range(3),['No shared\nclinical term','Total source\nviews','No conditional\nrelations'])
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(9):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=5.4)
    ax.set_title('Full CPCT - controlled ablation (AUROC)',pad=10)
    for spine in ax.spines.values():spine.set_visible(False)
    panel(ax,'a')
    cax=fig.add_axes([.946,.70,.012,.16]);fig.colorbar(im,cax=cax);cax.tick_params(labelsize=5)
    bx=fig.add_subplot(gs[1,0]);cx=fig.add_subplot(gs[1,1],sharey=bx)
    pan=main[main.method=='Pan_drugwise'].set_index(['seed','drug'])
    delta=(full.AUC-pan.AUC).unstack('drug').reindex(columns=DRUGS)
    for j in range(9):
        for k,value in enumerate(delta.iloc[:,j]):bx.scatter(value,j+(k-1)*.14,s=14,
                color=TEAL if value>=0 else RED,edgecolor='white',lw=.2)
    bx.axvline(0,color='#9CA5A8',lw=.7,ls='--')
    bx.set_yticks(range(9),SHORT);bx.set_ylim(8.5,-.6)
    bx.set_xlabel(r'$\Delta$AUROC: full - common source')
    bx.set_title('Added residual-response views',pad=9)
    selection=pd.read_csv(DATA/'selection.csv');selection=selection[selection.method=='Contrast_drugwise']
    active=selection.assign(active=selection.source_C>0).groupby('drug').active.sum().reindex(DRUGS).to_numpy()
    cx.barh(range(9),9-active,left=active,color=LIGHT,height=.66)
    cx.barh(range(9),active,color=TEAL,height=.66)
    for j,n in enumerate(active):cx.text(min(n+.18,8.0),j,str(int(n))+'/9',va='center',fontsize=6)
    cx.set_xticks([0,3,6,9]);cx.set_xlim(0,9.6);cx.set_xlabel('Source-enabled models (of 9)')
    cx.tick_params(axis='y',labelleft=False);cx.set_title('Training-selected source use',pad=9)
    panel(bx,'b');panel(cx,'c')
    export(fig,'fig3_components')


def parameters():
    t=pd.read_csv(DATA/'parameter_profiles.csv')
    fig,axes=plt.subplots(1,3,figsize=(122/25.4,76/25.4))
    fig.subplots_adjust(left=.13,right=.84,bottom=.20,top=.86,wspace=.19)
    matrices=[]
    for name in ['common_C','private_C','source_C']:
        a=t[t.parameter==name].groupby(['drug','value']).inner_logloss.mean().unstack().reindex(DRUGS)
        matrices.append(a.subtract(a.min(axis=1),axis=0))
    vmax=max(float(m.to_numpy().max()) for m in matrices)
    for i,(ax,m,title) in enumerate(zip(axes,matrices,['Shared clinical $C_\\beta$','Private clinical $C_\\delta$','Source $C_\\gamma$'])):
        im=ax.imshow(m.to_numpy(),aspect='auto',cmap=SEQ,vmin=0,vmax=vmax)
        labels=[('0' if v==0 else ('0.0001' if v==.0001 else str(v).rstrip('0').rstrip('.'))) for v in m.columns]
        ax.set_xticks(range(len(labels)),labels,rotation=55,ha='right')
        ax.set_yticks(range(9),SHORT if i==0 else ['']*9);ax.tick_params(length=0)
        for j in range(9):
            for k in range(m.shape[1]):ax.text(k,j,f'{m.iloc[j,k]:.3f}',ha='center',va='center',fontsize=5.4,
                color='white' if m.iloc[j,k]>vmax*.65 else INK)
        ax.set_title(title,pad=12)
        for spine in ax.spines.values():spine.set_visible(False)
        panel(ax,'abc'[i])
    cax=fig.add_axes([.875,.25,.014,.55]);cb=fig.colorbar(im,cax=cax)
    cb.ax.tick_params(labelsize=5.5);cb.set_label('Excess inner-validation log loss',fontsize=6,labelpad=4)
    export(fig,'fig4_parameters')


def cases():
    dist=pd.read_csv(DATA/'case_source_distribution.csv');selected=pd.read_csv(DATA/'selected_cases.csv')
    fig,axes=plt.subplots(2,2,figsize=(122/25.4,90/25.4))
    fig.subplots_adjust(left=.14,right=.98,bottom=.12,top=.91,hspace=.75,wspace=.36)
    for k,(drug,title) in enumerate([('Cyclophosphamide','Cyclophosphamide'),('Gemcitabine','Gemcitabine')]):
        ax=axes[0,k];t=dist[dist.drug==drug]
        vals=[t[t.label==label].source_term.to_numpy() for label in [0,1]]
        ax.boxplot(vals,positions=[0,1],widths=.52,patch_artist=True,showfliers=False,
            boxprops={'facecolor':'#DCE9E9','edgecolor':TEAL,'linewidth':.7},
            medianprops={'color':INK,'linewidth':.9},whiskerprops={'color':GREY,'linewidth':.6},
            capprops={'color':GREY,'linewidth':.6})
        for label,v in enumerate(vals):
            offset=np.linspace(-.15,.15,len(v)) if len(v)>1 else [0]
            ax.scatter(np.full(len(v),label)+offset,v,s=7,color=GREY,alpha=.45,lw=0)
        ax.axhline(0,color='#9CA5A8',ls='--',lw=.7)
        ax.set_xticks([0,1],[f'Nonresponse\n$n={len(vals[0])}$',f'Response\n$n={len(vals[1])}$'])
        ax.set_ylabel('Source logit contribution');ax.set_title(title,pad=10)
        panel(ax,'ab'[k])
        ax=axes[1,k];u=selected[selected.drug==drug].sort_values('label')
        x=np.arange(2);base=u.clinical_term.to_numpy();source=u.source_term.to_numpy();total=base+source
        ax.bar(x,base,width=.50,color='#CBD5DE',edgecolor='white',label='Clinical term')
        ax.bar(x,source,bottom=base,width=.50,color=TEAL,edgecolor='white',label='Source term')
        ax.scatter(x,total,color=INK,s=18,marker='D',zorder=4)
        for j,(_,row) in enumerate(u.iterrows()):
            y=max(0,float(total[j]),float(base[j]))
            ax.annotate(f'$p={row.score:.3f}$',(j,y),xytext=(0,6),textcoords='offset points',
                        ha='center',va='bottom',fontsize=6)
        ax.set_xticks(x,[f'{v}\n$y={int(label)}$' for v,label in zip(u['case'],u.label)])
        ax.set_ylabel('Joint-model logit');ax.set_title('Median-amplitude held-out cases',pad=9)
        span=max(max(abs(base)),max(abs(total)),.1)
        ax.set_ylim(min(0,min(base),min(total))-.18*span,max(0,max(base),max(total))+.38*span)
        panel(ax,'cd'[k])
    fig.legend(handles=[Patch(facecolor='#CBD5DE',label='Clinical term of joint model'),Patch(facecolor=TEAL,label='Source term')],
               loc='lower center',bbox_to_anchor=(.55,.005),ncol=2,frameon=False)
    export(fig,'fig5_cases')


if __name__=='__main__':
    style();OUT.mkdir(exist_ok=True,parents=True)
    for fn in [overview,robustness,ablations,parameters,cases]:fn();print('rendered',fn.__name__,flush=True)
