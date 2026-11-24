# -*- coding: utf-8 -*-
"""Redraw Figure 1 (framework) and Figure 2 (a-e experiment panels).

Uses only the existing source_data evidence and the published numeric values.
New independent filenames; original figures are not overwritten.  Output PDF
(embedded fonts) + editable SVG, publication style from the manuscript.

Figure 2 is redrawn with box plots (fold-level distributions) and violin plots
(patient-level score distributions) instead of bars/heatmaps.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "source_data"
OUT = ROOT / "figures"
DRUGS = ['5-Fluorouracil','Cisplatin','Cyclophosphamide','Docetaxel','Doxorubicin',
         'Etoposide','Gemcitabine','Paclitaxel','Temozolomide']
SHORT = ['5-FU','CIS','CTX','DOC','DOX','ETO','GEM','PAC','TMZ']
TEAL='#3E7C80'; BLUE='#5B7FA3'; GREY='#5C6873'; LIGHT='#EEF2F4'; INK='#25313A'
GREEN='#4E8C6A'; RED='#B05A5A'; GOLD='#B69260'
LINE='#7C888E'


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Liberation Sans'],
        'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.linewidth':.7,
        'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False,
        'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'axes.titlesize':7.8,'savefig.facecolor':'white','axes.unicode_minus':False,
        'axes.axisbelow':True})


def export(fig, stem):
    for fmt in ('pdf','svg','png'):
        fig.savefig(OUT / f'{stem}.{fmt}', dpi=600, bbox_inches='tight')
    plt.close(fig)


def box(ax,x,y,w,h,text,fc,fs=6.2,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.004,rounding_size=.012',
                 facecolor=fc,edgecolor='#8C9A9F',linewidth=.7))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,
            fontweight='bold' if bold else 'normal',linespacing=1.25)


def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.9,color='#5B6B72'))


def fig1():
    fig=plt.figure(figsize=(122/25.4,92/25.4));ax=fig.add_axes([.01,.01,.98,.98]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.5,.975,'CPCT framework: three computation modules',ha='center',fontsize=9,fontweight='bold',color=INK)
    for (x0,w,fc,ec) in [(.012,.30,'#F4F8F8',TEAL),(.345,.31,'#F7F6F2',GOLD),(.688,.30,'#F3F6F8',BLUE)]:
        ax.add_patch(FancyBboxPatch((x0,.56),w,.35,boxstyle='round,pad=.005,rounding_size=.012',
                     facecolor=fc,edgecolor=ec,linewidth=1.0))
    ax.text(.162,.882,'1  Common–Private\nResponse Decomposition',ha='center',fontsize=6.3,fontweight='bold',color=TEAL,linespacing=1.1)
    ax.text(.500,.882,'2  Drug-Relation-Guided\nTransfer Priors',ha='center',fontsize=6.3,fontweight='bold',color='#8A6E42',linespacing=1.1)
    ax.text(.838,.882,'3  Joint Common–Private\nClinical Learning',ha='center',fontsize=6.3,fontweight='bold',color=BLUE,linespacing=1.1)
    box(ax,.024,.742,.276,.095,'Frozen source $-\\mathrm{Z}_d$\ncommon proxy $u(x)$','#E4EDF0',5.9)
    box(ax,.024,.632,.276,.098,'Remove trend $h(u)$\nresiduals $\\epsilon_d,\\epsilon_e$','#DCEAEC',5.9)
    box(ax,.024,.578,.276,.045,'view + ridge selected on source val','#EEF2F4',5.2)
    arrow(ax,(.162,.742),(.162,.733)); arrow(ax,(.162,.632),(.162,.625))
    box(ax,.357,.742,.284,.095,r'$\rho_{de}=\mathrm{Corr}(\epsilon_d,\epsilon_e)^2$','#EFE9DE',6.2)
    box(ax,.357,.640,.284,.09,'reliability $\\eta_e$ from held-out $R_e^2$','#EFE9DE',5.7)
    box(ax,.357,.578,.284,.052,r'$v_{de}=(0.1+0.9\rho_{de})\eta_e$','#E9DFCC',6.2)
    arrow(ax,(.500,.742),(.500,.734)); arrow(ax,(.500,.640),(.500,.632))
    box(ax,.700,.742,.276,.095,'shared $x^\\top\\beta$; private $x^\\top\\delta_d$','#E8EEF3',5.7)
    box(ax,.700,.636,.276,.095,'transfer $\\phi_d^\\top\\gamma_d$\n$\\phi_d=\\mathrm{diag}(\\sqrt{v_d})\\mathrm{Std}(u,q_{1..9})$','#E4EEF0',5.6)
    box(ax,.700,.578,.276,.048,'inner drug-wise selection; zero OK','#EEF2F4',5.2)
    arrow(ax,(.838,.742),(.838,.734)); arrow(ax,(.838,.636),(.838,.626))
    arrow(ax,(.312,.688),(.343,.688)); arrow(ax,(.656,.612),(.687,.700))
    ax.text(.5,.515,r'$\epsilon_d,\epsilon_e\;\rightarrow\;\rho_{de}\;\rightarrow\;v_{de}\;\rightarrow\;\phi_d(x)$',
            ha='center',fontsize=8,fontweight='bold',color=INK)
    box(ax,.012,.355,.30,.115,'Training patients only:\nexpression scalers, inner folds,\nmodel fitting and selection','#F1F4F6',5.8)
    box(ax,.345,.355,.31,.115,'Held-out outer patients:\nno fitting, no adaptation,\nlabels isolated','#F4F1EC',5.8)
    box(ax,.688,.355,.30,.115,'Per drug: HARD selection\nAUROC / AUPR / Brier\npaired patient bootstrap','#EEF4F1',5.8)
    arrow(ax,(.312,.412),(.343,.412)); arrow(ax,(.656,.412),(.687,.412))
    ax.text(.5,.255,'No graph neural network or message passing is used; drug relations enter only as shrinkage scales.',
            ha='center',fontsize=6.4,color=GREY,style='italic')
    ax.text(.5,.195,'Source views are label-free w.r.t. clinical outcomes; source ridge fitting uses source cells only.',
            ha='center',fontsize=6.2,color=GREY)
    export(fig,'fig1_framework_revised')


def panel_letter(ax,s):
    ax.text(-.14,1.12,s,transform=ax.transAxes,fontsize=9,fontweight='bold',color=INK,clip_on=False)


def style_box(bp, colors):
    for patch,c in zip(bp['boxes'],colors):
        patch.set_facecolor(c);patch.set_edgecolor(LINE);patch.set_linewidth(.7);patch.set_alpha(.9)
    for k in ('whiskers','caps'):
        for l in bp[k]: l.set_color(LINE);l.set_linewidth(.7)
    for m in bp['medians']: m.set_color(INK);m.set_linewidth(1.0)
    for f in bp.get('fliers',[]):
        f.set_marker('o');f.set_markersize(2);f.set_markerfacecolor(LINE);f.set_markeredgecolor('none');f.set_alpha(.6)


def style_violin(vp, color):
    for b in vp['bodies']:
        b.set_facecolor(color);b.set_edgecolor(LINE);b.set_linewidth(.5);b.set_alpha(.72)
    for k in ('cmeans','cmins','cmaxes','cbars'):
        if k in vp: vp[k].set_color(LINE);vp[k].set_linewidth(.6)
    if 'cmedians' in vp: vp['cmedians'].set_color(INK);vp['cmedians'].set_linewidth(1.0)


def fig2():
    cohort=pd.read_csv(DATA/'cohort.csv')
    mainfold=pd.read_csv(DATA/'main_per_fold.csv')
    abl=pd.read_csv(DATA/'ablation_per_fold.csv')
    params=pd.read_csv(DATA/'parameter_profiles.csv')
    pat=pd.read_csv(DATA/'inductive_per_patient.csv')

    fig=plt.figure(figsize=(140/25.4,162/25.4))
    gs=fig.add_gridspec(3,2,height_ratios=[1,1,1.08],hspace=.62,wspace=.44)
    fig.subplots_adjust(left=.13,right=.97,bottom=.07,top=.95)

    # (a) label distribution -- counts, so a grouped horizontal bar is the honest encoding
    ax=fig.add_subplot(gs[0,0])
    y=np.arange(9);h=.38
    ax.barh(y-h/2,cohort.positive,h,color=GREEN,edgecolor='none',label='Responder (CR+PR)')
    ax.barh(y+h/2,cohort.negative,h,color=RED,edgecolor='none',label='Non-responder',alpha=.85)
    ax.set_yticks(y,SHORT);ax.invert_yaxis();ax.set_xlabel('patients')
    ax.set_title('Real clinical label distribution',loc='left')
    ax.grid(axis='x',color=LIGHT,lw=.6);ax.set_axisbelow(True)
    panel_letter(ax,'a');ax.legend(loc='center right',fontsize=5.6)

    # (b) AUROC distribution across held-out folds: CPCT vs Clinical (paired box plots)
    ax=fig.add_subplot(gs[0,1])
    cp=mainfold[mainfold.method=='Contrast_drugwise']
    cl=mainfold[mainfold.method=='Clinical_drugwise']
    dc=[cp.loc[cp.drug==d,'AUC'].to_numpy() for d in DRUGS]
    dl=[cl.loc[cl.drug==d,'AUC'].to_numpy() for d in DRUGS]
    pos=np.arange(9);w=.36
    b1=ax.boxplot(dc,positions=pos-w/2,widths=w,patch_artist=True,showfliers=True,manage_ticks=False)
    b2=ax.boxplot(dl,positions=pos+w/2,widths=w,patch_artist=True,showfliers=True,manage_ticks=False)
    style_box(b1,[TEAL]*9);style_box(b2,['#B7C0C5']*9)
    ax.set_xticks(pos,SHORT,rotation=45,ha='right');ax.set_ylabel('AUROC (held-out folds)')
    ax.set_title('Ranking robustness across 9 folds',loc='left')
    ax.grid(axis='y',color=LIGHT,lw=.6);ax.set_axisbelow(True)
    panel_letter(ax,'b')
    ax.legend([Patch(facecolor=TEAL),Patch(facecolor='#B7C0C5')],['CPCT','Clinical-MTL'],
              loc='lower right',fontsize=5.6)

    # (c) component ablation: distribution of AUROC drop when a component is removed
    ax=fig.add_subplot(gs[1,0])
    full=mainfold[mainfold.method=='Contrast_drugwise'].set_index(['seed','fold','drug']).AUC
    names=['NoShared','TotalViews','NoRelations']
    labels=['No shared\nclinical','Total source\nviews','No conditional\nrelations']
    dd=[]
    for nm in names:
        s=abl[abl.method==nm].set_index(['seed','fold','drug']).AUC
        dd.append((full-s).dropna().to_numpy())
    bc=ax.boxplot(dd,positions=[1,2,3],widths=.55,patch_artist=True,showfliers=True)
    style_box(bc,[ '#9FBCC0','#6F9AA0',TEAL])
    ax.axhline(0,color=GREY,lw=.7,ls='--')
    ax.set_xticks([1,2,3],labels);ax.set_ylabel(r'$\Delta$AUC (Full $-$ ablated)')
    ax.set_title('Component ablation (81 fold-drug points)',loc='left')
    ax.grid(axis='y',color=LIGHT,lw=.6);ax.set_axisbelow(True)
    panel_letter(ax,'c')

    # (d) inner-validation sensitivity: log-loss distribution per C_beta value
    ax=fig.add_subplot(gs[1,1])
    sel=params[params.parameter=='common_C']
    vals=sorted(sel.value.unique())
    dp=[sel.loc[sel.value==v,'inner_logloss'].to_numpy() for v in vals]
    bd=ax.boxplot(dp,positions=range(len(vals)),widths=.55,patch_artist=True,showfliers=True)
    style_box(bd,['#CBDAE5','#96B3CB',BLUE])
    ax.set_xticks(range(len(vals)),[f'{v:g}' for v in vals])
    ax.set_xlabel('$C_\\beta$ (shared clinical)');ax.set_ylabel('inner log-loss')
    ax.set_title('Inner-validation sensitivity',loc='left')
    ax.grid(axis='y',color=LIGHT,lw=.6);ax.set_axisbelow(True)
    panel_letter(ax,'d')

    # (e) patient-level score distribution by true response: violin plots per drug
    ax=fig.add_subplot(gs[2,:])
    sub=pat[pat.method=='CPCT_inductive']
    sub=sub.groupby(['patient_id','drug','label'],as_index=False).score.mean()
    vp=[sub[(sub.drug==d)&(sub.label==1)].score.to_numpy() for d in DRUGS]
    vn=[sub[(sub.drug==d)&(sub.label==0)].score.to_numpy() for d in DRUGS]
    xp=np.arange(9);wv=.34
    v1=ax.violinplot(vp,positions=xp-wv/2,widths=wv,showmedians=True,showextrema=True,points=60)
    v2=ax.violinplot(vn,positions=xp+wv/2,widths=wv,showmedians=True,showextrema=True,points=60)
    style_violin(v1,GREEN);style_violin(v2,'#C0C7CB')
    ax.set_xticks(xp,SHORT);ax.set_ylabel('predicted response score (CPCT inductive)')
    ax.set_title('Patient score distribution by true response',loc='left')
    ax.grid(axis='y',color=LIGHT,lw=.6);ax.set_axisbelow(True)
    panel_letter(ax,'e')
    ax.legend([Patch(facecolor=GREEN,alpha=.72),Patch(facecolor='#C0C7CB',alpha=.72)],
              ['Responder (CR+PR)','Non-responder'],loc='lower right',
              bbox_to_anchor=(1.0,1.03),ncol=2,fontsize=5.8,columnspacing=1.4)
    export(fig,'fig2_experiments_ace')


if __name__=='__main__':
    style(); OUT.mkdir(exist_ok=True)
    fig1(); fig2()
    print('FIGURES DONE')
