# CPCT: Response Contrast Decomposition and Drug-Relation Guided Transfer for Clinical Drug Response Prediction

This repository contains the experimental code for **CPCT** (Common–Private
Response Contrast Transfer), a method that transfers response-contrast knowledge
from preclinical cell lines (source domain) to clinical patients (target domain)
for drug-response prediction. CPCT combines three components:

1. **Common–Private Response Decomposition** — pan-response and drug-specific
   response contrasts learned from source cell lines, with a shared
   principal-vector (PV20) subspace used for source→target scale harmonization.
2. **Drug-Relation-Guided Transfer Priors** — a transfer prior over source views
   built from source-view reliability and drug–drug response relations.
3. **Joint Common–Private Clinical Learning** — a multi-task clinical model that
   uses the source-derived views as a guided prior, with patient-grouped nested
   selection and a hard clinical readout.

The code implements the source-view construction, the nested patient
cross-validation training and selection, the component ablations, and the
figure/table generation used in the paper.

> **Note on data.** This repository intentionally contains **code only** — no
> trained models, no prediction results, no processed data arrays, and no
> figures. The underlying clinical and preclinical datasets are subject to their
> original access conditions and are **not** redistributed here. You must supply
> the input arrays described in [Data preparation](#4-data-preparation) to run
> the pipeline.

---

## 1. Repository structure

```
.
├── README.md
├── requirements.txt                  # Python dependencies (copy at repository root)
├── implementation/                   # Core CPCT algorithm (reproduces the main results)
│   ├── paths.py                      # Data / output root configuration (environment variables)
│   ├── splits.py                     # Patient-level K-fold splits
│   ├── source.py                     # Source-only pan/private response contrasts + PV20 view
│   ├── prior.py                      # Source-view reliability & drug-relation transfer prior
│   ├── model.py                      # Fixed-setting probe / single configuration fit
│   ├── run.py                        # Nested patient cross-validation (main experiment)
│   └── requirements.txt              # Same dependency list as the root requirements.txt
├── analysis/                         # Figures & evidence tables for the paper
│   ├── pub_style.py                  # Publication plotting style and export helpers
│   ├── run_ablations.py              # NoShared / TotalViews / NoRelations ablations
│   ├── prepare_evidence.py           # Derive paper tables/CSV from archived model outputs
│   └── make_figures.py               # Recreate framework / robustness / ablation figures
└── revision_experiments/             # Additional robustness experiments (see section 6)
    ├── rev_engine.py                 # Shared patient-level evaluation engine
    ├── rev_source.py                 # Fold-specific (strict inductive) source views
    ├── verify_source_rebuild.py      # Source-view reconstruction checks
    ├── validate_engine.py            # From-scratch engine consistency check
    ├── check_pv20.py                 # PV20 projection reproducibility check
    ├── exp_inductive.py              # Strict inductive generalization
    ├── exp_crossfit.py               # Cross-fitted source reliability
    ├── exp_prior.py                  # Drug-relation prior sensitivity
    ├── exp_bootstrap.py              # Patient-level paired bootstrap
    ├── exp_audit.py                  # Clinical-label provenance audit
    ├── export_alignment.py           # Export source→target alignment coordinates
    ├── plot_alignment.py             # Source→target alignment scatter figure
    ├── exp_figures.py                # Recreate revised framework / experiment figures
    └── make_report.py                # Assemble the experiment report from result files
```

---

## 2. Environment

Tested with **Python 3.12**. Core dependencies (see `requirements.txt`):

```
numpy==2.3.5
scipy==1.17.0
scikit-learn==1.8.0
pandas==2.2.3
matplotlib==3.10.6
```

Setup:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`matplotlib` is only required for the `analysis/` and `revision_experiments/`
plotting scripts; the core training pipeline needs only numpy / scipy /
scikit-learn / pandas. Some exercises use `concurrent.futures` from the standard
library. No GPU and no deep-learning framework are required to run this code —
the upstream representations consumed as input must be prepared separately
(section 4).

---

## 3. Configuration (environment variables)

Paths are configured entirely through environment variables, so no source file
needs to be edited.

| Variable | Meaning | Default |
|---|---|---|
| `DCIT_DATA_ROOT` | Root directory containing the input data arrays (section 4) | `<repo_parent>/real_data/DCIT_REAL_DATA_v1` |
| `CPCT_OUTPUT_ROOT` | Root directory for all outputs (source views, models, tables) | `<implementation_dir>/outputs` |

Example:

```bash
export DCIT_DATA_ROOT=/path/to/DCIT_REAL_DATA_v1
export CPCT_OUTPUT_ROOT=/path/to/cpct_outputs
```

---

## 4. Data preparation

The code expects the following files **relative to `DCIT_DATA_ROOT`**. These are
inputs only; they are not included in this repository.

| Relative path | Content |
|---|---|
| `dcit/cache/patients.npz` | Patient expression matrix `X` (1186 patients × 1306 genes), response labels `Y` and masks `mask` for 9 drugs, `train` split mask, `patient_ids` |
| `benchmark_dcit/cache/source_raw_v2.npz` | Source cell-line raw data: `cell_ids`, `train_mask`, drug masks / responses |
| `dcit/cache/source_latents.npz` | Source cell-line latent representations |
| `representations/transdrp.npz` | Upstream native scores and id arrays: `src_ids`, `tgt_ids`, `source_train_mask`, `src_native_score`, `tgt_native_score` (used only as a fixed, frozen proxy — no patient fitting) |

Key arrays are aligned by id in the code (`patient_ids`, `cell_ids`), not by row
position; keep these arrays consistent when preparing your own data. The nine
drugs are: 5-Fluorouracil, Cisplatin, Cyclophosphamide, Docetaxel, Doxorubicin,
Etoposide, Gemcitabine, Paclitaxel, Temozolomide.

The upstream `transdrp.npz` representation is a product of the earlier DCIT /
TransDRP pipeline. If you only want to reproduce the CPCT contribution on your
own data, you can substitute any fixed source/target score matrix with the same
array keys and shapes.

---

## 5. Running the core pipeline

Run the scripts from the `implementation/` directory (they import sibling
modules such as `paths` and `splits` directly).

### Step 1 — Build source views (pan/private contrasts + PV20 shared subspace)

```bash
cd implementation
python source.py
```

Writes to `$CPCT_OUTPUT_ROOT/source/`: `preprocessing.pkl`, `prepared.pkl`,
`profiles.npz`, `source_heads.pkl`, `source_validation.csv`, `source_prior.npz`.

### Step 2 — Main nested cross-validation experiment

```bash
python run.py --seeds 2024 2025 2026
```

Runs nested patient cross-validation (patient-grouped outer folds and inner
configuration selection) and writes per-fold / per-seed results, selected
configurations and fitted models to `$CPCT_OUTPUT_ROOT/nested/` by default
(override with `--out`). Each patient's drug records stay in the same fold.

### Step 3 — Fixed-setting probe (optional)

```bash
python model.py --seeds 2024
```

Single-setting probe that fits one configuration per fold; useful for debugging
and for checking source-view effects without the full nested selection.

---

## 6. Ablations and paper figures

These scripts consume the archived outputs of the core pipeline. They take
explicit roots on the command line rather than relying on environment defaults.

```bash
# Component ablations: NoShared / TotalViews / NoRelations
python analysis/run_ablations.py \
    --data-root /path/to/DCIT_REAL_DATA_v1 \
    --cpct-root /path/to/cpct_outputs \
    --out /path/to/ablation_outputs

# Derive paper tables / evidence CSVs from archived model outputs
python analysis/prepare_evidence.py \
    --cpct-root /path/to/cpct_outputs \
    --data-root /path/to/DCIT_REAL_DATA_v1 \
    --ablation-root /path/to/ablation_outputs \
    --out /path/to/source_data \
    --tex-out /path/to/tables

# Recreate the framework / robustness / ablation / parameter figures
# (reads <repo>/source_data and writes <repo>/figures)
python analysis/make_figures.py
```

`make_figures.py` resolves its input/output relative to the repository root
(`<repo>/source_data` → `<repo>/figures`), so place the output of
`prepare_evidence.py --out` there before running it.

### Revision / robustness experiments

`revision_experiments/` contains the additional analyses from the manuscript
revision. They require the archived per-patient predictions and source-view
artifacts of a completed main run; several of them reference relative archive
directories (e.g. `../../cpct_select/...`) and expect those artifacts to be
present. Run them after the core pipeline, from that directory. Typical order:

```bash
cd revision_experiments
python validate_engine.py          # consistency check: from-scratch engine vs archived predictions
python check_pv20.py               # PV20 projection reproducibility check
python exp_inductive.py            # strict inductive generalization
python exp_crossfit.py             # cross-fitted source reliability
python exp_prior.py                # drug-relation prior sensitivity
python exp_bootstrap.py            # patient-level paired bootstrap
python exp_audit.py                # clinical-label provenance audit
python export_alignment.py         # export alignment coordinates
python plot_alignment.py           # source→target alignment figure
python exp_figures.py              # revised framework / experiment figures
python make_report.py              # assemble the experiment report
```

These scripts write their outputs into a local `source_data/` directory and
`figures/` next to them. They are self-contained given the archived artifacts;
no conclusions are generated automatically — `make_report.py` only assembles
computed values and explicitly flags any missing or unverifiable fields.

---

## 7. Outputs

Running the pipeline produces (under `$CPCT_OUTPUT_ROOT` and the analysis
output directories):

- source-view artifacts (`profiles.npz`, `source_prior.npz`, `preprocessing.pkl`, `prepared.pkl`);
- per-fold / per-seed prediction tables and selected configurations;
- fitted model checkpoints;
- component-ablation results;
- derived evidence tables (AUROC / AUPR / Brier, cohort, selection, parameters, cases);
- publication figures.

No such outputs are committed to this repository.

---

## 8. Reproducibility notes

- Random seeds for the outer patient partitions are 2024 / 2025 / 2026 (three
  seeds × three folds = nine outer partitions). A patient's drug records are
  always assigned to the same fold.
- Inner selection uses 36 candidate configurations (common regularization
  ∈ {1e-4, 1e-3, 1e-2} × private ∈ {0, 1e-4, 1e-3} × source ∈ {0, 0.1, 1, 10});
  the source-free Clinical-MTL control uses the nine `source = 0` candidates.
- The reported readout is HARD. Model selection uses unweighted inner
  validation log loss with patient-balanced training weights.
- Reproducibility checks included in `revision_experiments/` verify that the
  source views reconstruct exactly from the prepared inputs (max |diff| = 0)
  and that the from-scratch clinical engine matches archived predictions to
  numerical precision (max |diff| ≈ 1e-16).

For questions, please open an issue in this repository.
