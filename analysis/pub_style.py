import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def apply_publication_style(font_size=7, axes_linewidth=0.8, use_tex=False):
    """Apply Nature-style rcParams. Call once before creating any figures.

    Defaults target a 183 mm (double-column) final figure: 7 pt type, 0.8 pt axes.
    Pass a larger size only when the canvas itself is larger (slide, poster).
    """
    # ── MANDATORY: editable SVG text ──────────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['svg.fonttype'] = 'none'
    # ── Layout & style ────────────────────────────────────────────────────────
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = axes_linewidth
    plt.rcParams['legend.frameon'] = False
    if use_tex:
        plt.rcParams['text.usetex'] = True

def save_pub_set(fig, out_dir, stem, formats=('svg', 'pdf', 'tiff'), dpi=600,
                 bbox_inches='tight', close=True):
    """
    Export one figure as a submission-grade file set.

    Parameters
    ----------
    fig      : matplotlib Figure
    out_dir : str | Path — target directory (created if missing)
    stem    : str          — file stem, e.g. 'figure-01'
    formats : tuple        — subset of ('svg', 'pdf', 'tiff', 'png')
    dpi     : int          — raster dpi (600 for line art and dense panels)
    bbox_inches : str|None — 'tight' keeps the frameless stat strip and panel
                             labels; use None only if you manage margins yourself.

    Returns
    -------
    list[str] — saved file paths, in order
    """
    import os
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = str(out / f'{stem}.{fmt}')
        kw = {}
        if fmt in ('svg', 'pdf', 'tiff', 'png') and bbox_inches:
            kw['bbox_inches'] = bbox_inches
        fig.savefig(p, dpi=dpi, **kw)
        saved.append(p)
    if close:
        plt.close(fig)
    return saved
