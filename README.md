# CPCT: Response Contrast Decomposition and Drug-Relation Guided Transfer for Clinical Drug Response Prediction

This repository contains the **core running code** for CPCT (Common–Private
Response Contrast Transfer): the method that transfers response-contrast
knowledge from preclinical cell lines (source domain) to clinical patients
(target domain) for drug-response prediction.

CPCT combines three components:

1. **Common–Private Response Decomposition** — pan-response and drug-specific
   response contrasts learned from source cell lines, with a shared
   principal-vector (PV20) subspace for source→target scale harmonization.
2. **Drug-Relation-Guided Transfer Priors** — a transfer prior over source views
   built from source-view reliability and drug–drug response relations.
3. **Joint Common–Private Clinical Learning** — a multi-task clinical model that
   uses the source-derived views as a guided prior, with patient-grouped nested
   selection and a hard clinical readout.

> **Code only.** No trained models, prediction results, processed data arrays,
> or figures are distributed here. The underlying clinical and preclinical
> datasets are subject to their original access conditions and must be supplied
> by the user (see [Data preparation](#4-data-preparation)).

---

## 1. Files

```
.
├── README.md
├── requirements.txt          # Python dependencies
└── implementation/
    ├── paths.py              # Data / output root configuration (environment variables)
    ├── splits.py             # Patient-level K-fold splits
    ├── source.py             # Source-only pan/private response contrasts + PV20 shared view
    ├── prior.py              # Source-view reliability & drug-relation transfer prior
    ├── model.py              # Fixed-setting probe / single-configuration fit
    ├── run.py                # Nested patient cross-validation (main experiment)
    └── requirements.txt      # Same dependency list (local copy)
```

---

## 2. Environment

Tested with **Python 3.12**. Dependencies (see `requirements.txt`):

```
numpy==2.3.5
scipy==1.17.0
scikit-learn==1.8.0
pandas==2.2.3
```

Setup:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

No GPU and no deep-learning framework are required. The upstream source/target
representations consumed as input must be prepared separately (section 4).

---

## 3. Configuration (environment variables)

Paths are configured entirely through environment variables — no source file
needs to be edited.

| Variable | Meaning | Default |
|---|---|---|
| `DCIT_DATA_ROOT` | Root directory containing the input data arrays (section 4) | `<repo_parent>/real_data/DCIT_REAL_DATA_v1` |
| `CPCT_OUTPUT_ROOT` | Root directory for all outputs (source views, models, result tables) | `<implementation_dir>/outputs` |

Example:

```bash
export DCIT_DATA_ROOT=/path/to/DCIT_REAL_DATA_v1
export CPCT_OUTPUT_ROOT=/path/to/cpct_outputs
```

---

## 4. Data preparation

The code expects the following files **relative to `DCIT_DATA_ROOT`**. These are
inputs only and are not included in this repository.

| Relative path | Content |
|---|---|
| `dcit/cache/patients.npz` | Patient expression matrix `X` (1186 patients × 1306 genes), response labels `Y` and masks `mask` for 9 drugs, `train` split mask, `patient_ids` |
| `benchmark_dcit/cache/source_raw_v2.npz` | Source cell-line raw data: `cell_ids`, `train_mask`, drug masks / responses |
| `dcit/cache/source_latents.npz` | Source cell-line latent representations |
| `representations/transdrp.npz` | Upstream native scores and id arrays: `src_ids`, `tgt_ids`, `source_train_mask`, `src_native_score`, `tgt_native_score` (used as a fixed, frozen proxy — no patient-data fitting) |

Arrays are aligned by id (`patient_ids`, `cell_ids`), not by row position; keep
them consistent when preparing your own data. The nine drugs are:
5-Fluorouracil, Cisplatin, Cyclophosphamide, Docetaxel, Doxorubicin, Etoposide,
Gemcitabine, Paclitaxel, Temozolomide.

The upstream `transdrp.npz` representation is a product of an earlier pipeline.
To reproduce only the CPCT contribution on your own data, you may substitute any
fixed source/target score matrix with the same array keys and shapes.

---

## 5. Running the code

Run the scripts from the `implementation/` directory — they import sibling
modules (`paths`, `splits`, etc.) directly.

### Step 1 — Build source views (pan/private contrasts + PV20 shared subspace)

```bash
cd implementation
export DCIT_DATA_ROOT=/path/to/DCIT_REAL_DATA_v1
export CPCT_OUTPUT_ROOT=/path/to/cpct_outputs
python source.py
```

Writes to `$CPCT_OUTPUT_ROOT/source/`: `preprocessing.pkl`, `prepared.pkl`,
`profiles.npz`, `source_heads.pkl`, `source_validation.csv`, `source_prior.npz`.

### Step 2 — Main nested cross-validation experiment

```bash
python run.py --seeds 2024 2025 2026
```

Runs nested patient cross-validation (patient-grouped outer folds and inner
configuration selection). Writes per-fold / per-seed results, selected
configurations and fitted models to `$CPCT_OUTPUT_ROOT/nested/` by default
(override with `--out`). Each patient's drug records stay in the same fold.

### Step 3 — Fixed-setting probe (optional)

```bash
python model.py --seeds 2024
```

Fits one configuration per fold; useful for debugging and for checking
source-view effects without the full nested selection. Defaults to
`$CPCT_OUTPUT_ROOT/fixed` (override with `--out`).

### Command-line options

| Script | Options |
|---|---|
| `source.py` | none (uses environment variables) |
| `run.py` | `--seeds` (default `2024 2025 2026`), `--out` (default `$CPCT_OUTPUT_ROOT/nested`) |
| `model.py` | `--seeds` (default `2024`), `--out` (default `$CPCT_OUTPUT_ROOT/fixed`) |

---

## 6. Outputs

Running the pipeline produces, under `$CPCT_OUTPUT_ROOT`:

- source-view artifacts (`profiles.npz`, `source_prior.npz`, `preprocessing.pkl`, `prepared.pkl`);
- per-fold / per-seed prediction tables and selected configurations;
- fitted model checkpoints.

All outputs are generated locally at run time; none are committed here.

---

## 7. Reproducibility notes

- Outer patient partitions use seeds 2024 / 2025 / 2026 (three seeds × three
  folds = nine outer partitions). A patient's drug records are always assigned
  to the same fold.
- Inner selection uses 36 candidate configurations (common regularization
  ∈ {1e-4, 1e-3, 1e-2} × private ∈ {0, 1e-4, 1e-3} × source ∈ {0, 0.1, 1, 10});
  the source-free Clinical-MTL control uses the nine `source = 0` candidates.
- The reported readout is HARD. Model selection uses unweighted inner
  validation log loss with patient-balanced training weights.

For questions, please open an issue in this repository.
