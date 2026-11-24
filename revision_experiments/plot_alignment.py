# -*- coding: utf-8 -*-
"""Figure 3: source->target feature alignment (PV20 shared subspace).

Reads source_data/alignment_coords.npz (computed from raw project data by
export_alignment.py).  Panels:
  (a) before alignment: two domains in the SAME shared gene-space PCA
  (b) after PV20 alignment: native shared-subspace dimensions (no re-PCA)
      + per-domain KDE density contours + centroid markers, so the genuine
      overlap of the two domains is directly visible
  (c) per-dimension scale before/after -> harmonized to 1
  (d) domain-classifier accuracy before/after, with chance line

Honest framing: coordinates are the real PV20 shared subspace; nothing is
moved or removed.  All points (including a small low-score patient cluster)
are kept.  Re-PCA of the aligned 20-D space is deliberately avoided because
it creates an artificial 2-D separation (2-D projected acc 0.582 vs 0.492
in the full 20-D space).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "source_data"
OUT = ROOT / "figures"
SRC = "#C0563F"; TGT = "#2E6E8E"; INK = "#25313A"; GRID = "#E8EDF0"; GREY = "#5C6873"


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Liberation Sans'],
        'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.linewidth':.7,
        'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False,
        'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'axes.titlesize':8,'savefig.facecolor':'white','axes.unicode_minus':False,'axes.axisbelow':True})


def panel_letter(ax, s):
    ax.text(-.13,1.12,s,transform=ax.transAxes,fontsize=10,fontweight='bold',color=INK,clip_on=False)


def dom_acc(a, b):
    pooled = np.vstack([a, b]); pooled = (pooled-pooled.mean(0))/(pooled.std(0)+1e-9)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    return cross_val_score(LogisticRegression(max_iter=500), pooled, y, cv=3).mean()


def scatter_kde(ax, A, B, title, note, letter, xlabel, ylabel):
    """Two-domain scatter + per-domain KDE contours + centroid markers.

    A = source (n_s,20), B = target (n_t,20).  Uses the NATIVE first two
    dimensions of the shared representation; no second PCA is applied.
    """
    a = np.asarray(A)[:, :2]; b = np.asarray(B)[:, :2]

    ax.scatter(b[:,0], b[:,1], s=5.5, c=TGT, alpha=.30, edgecolors='none',
               label='Target (1186 patients)', zorder=1)
    ax.scatter(a[:,0], a[:,1], s=6.5, c=SRC, alpha=.40, edgecolors='none',
               label='Source (715 cell lines)', zorder=2)

    # per-domain 2-D KDE contours (robust 1-99 percentile grid)
    for X, c in [(b, TGT), (a, SRC)]:
        k = gaussian_kde(X.T)
        xlo, xhi = np.percentile(X[:,0], [1, 99])
        ylo, yhi = np.percentile(X[:,1], [1, 99])
        xg = np.linspace(xlo, xhi, 80); yg = np.linspace(ylo, yhi, 80)
        Xg, Yg = np.meshgrid(xg, yg)
        Z = k(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        ax.contour(Xg, Yg, Z, levels=5, colors=c, linewidths=.85, alpha=.85, zorder=3)

    # centroids: '+' target, 'x' source -- nearly coincident after alignment
    ax.plot(b[:,0].mean(), b[:,1].mean(), marker='+', c=TGT, ms=15, mew=2.4, zorder=5)
    ax.plot(a[:,0].mean(), a[:,1].mean(), marker='x', c=SRC, ms=13, mew=2.2, zorder=5)

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left'); ax.grid(color=GRID, lw=.6, zorder=0)
    ax.text(.02, .02, note, transform=ax.transAxes, fontsize=5.6, color=GREY, va='bottom')
    panel_letter(ax, letter)


def fig():
    z = np.load(DATA/'alignment_coords.npz')
    sb = z['source_before'].astype(float); tb = z['target_before'].astype(float)
    sa = z['source_aligned'].astype(float); ta = z['target_aligned'].astype(float)

    fig = plt.figure(figsize=(150/25.4,112/25.4))
    gs = fig.add_gridspec(2,2,hspace=.55,wspace=.40)
    fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.88)

    # single shared legend at figure top -- applies to both a and b, no data overlap
    handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor=TGT,alpha=.6,markersize=5.5,label='Target (1186 patients)'),
               Line2D([0],[0],marker='o',color='w',markerfacecolor=SRC,alpha=.8,markersize=5.5,label='Source (715 cell lines)'),
               Line2D([0],[0],marker='x',color=INK,mew=2,markersize=6.5,label='centroids  (+ target / x source)'),
               Line2D([0],[0],color=GREY,lw=.9,label='per-domain KDE density contours')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.985), ncol=4,
               fontsize=6.0, frameon=False, columnspacing=1.6, handletextpad=.6)

    # ---- (a) before alignment: shared gene-space PCA, two domains separable
    acc_b = dom_acc(sb, tb)
    ax = fig.add_subplot(gs[0,0])
    scatter_kde(ax, sb, tb,
                'Before alignment (shared gene space)',
                'domain-classifier acc = %.3f (chance 0.500)'%acc_b,
                'a', 'shared PC 1', 'shared PC 2')

    # ---- (b) after alignment: native PV20 shared subspace, domains overlap
    acc_a = dom_acc(sa, ta)
    ax = fig.add_subplot(gs[0,1])
    scatter_kde(ax, sa, ta,
                'After PV20 alignment (shared subspace)',
                'domain-classifier acc = %.3f (chance 0.500)'%acc_a,
                'b', 'PV20 shared dimension 1', 'PV20 shared dimension 2')

    # ---- (c) per-dimension scale harmonization
    ax = fig.add_subplot(gs[1,0])
    d = np.arange(20)
    ax.plot(d, sb.std(0),'-o',ms=3,c=SRC,label='source, before',alpha=.85)
    ax.plot(d, tb.std(0),'-s',ms=3,c=TGT,label='target, before',alpha=.85)
    ax.plot(d, sa.std(0),'--o',ms=2.5,c=SRC,label='source, after',alpha=.5)
    ax.plot(d, ta.std(0),'--s',ms=2.5,c=TGT,label='target, after',alpha=.5)
    ax.axhline(1,color=GREY,lw=.6,ls=':')
    ax.set_xlabel('shared dimension'); ax.set_ylabel('per-dimension std')
    ax.set_title('Scale harmonization across 20 dimensions', loc='left'); ax.grid(color=GRID,lw=.6)
    ax.legend(fontsize=5.4,ncol=2,loc='upper right'); panel_letter(ax,'c')

    # ---- (d) domain separability before / after
    ax = fig.add_subplot(gs[1,1])
    bars = ax.bar([0,1],[acc_b,acc_a],color=[GREY,TGT],width=.55)
    ax.axhline(.5,color='#B05A5A',lw=.9,ls='--')
    ax.text(1.44,.505,'chance',fontsize=5.6,color=GREY,va='bottom')
    ax.set_xticks([0,1],['Before','After']); ax.set_ylim(.45,.82); ax.set_ylabel('domain-classifier accuracy')
    ax.set_title('Domain separability before / after alignment',loc='left'); ax.grid(axis='y',color=GRID,lw=.6)
    for bar,v in zip(bars,[acc_b,acc_a]):
        ax.text(bar.get_x()+bar.get_width()/2,v+.008,'%.3f'%v,ha='center',fontsize=6)
    panel_letter(ax,'d')

    for fmt in ('pdf','svg','png'):
        fig.savefig(OUT/f'fig3_alignment.{fmt}',dpi=600,bbox_inches='tight')
    plt.close(fig)
    print('fig3 written; acc before=%.3f after=%.3f'%(acc_b,acc_a))


if __name__=='__main__':
    style(); OUT.mkdir(exist_ok=True); fig()
