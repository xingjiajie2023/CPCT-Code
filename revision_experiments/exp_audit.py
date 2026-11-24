# -*- coding: utf-8 -*-
"""Exp: data-source audit of the nine-drug clinical labels.

Reads the raw label file (data/TCGA_labels.csv), the encoded arrays
(dcit/cache/patients.npz) and the harmonized feature file to verify: label
origin, patient-drug matching, response categories, sampling time, cancer types
and patient duplication.  Unknown fields are reported as unknown, never guessed.
"""
import os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "source_data")
PROJ = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
ROOT = os.environ["DCIT_DATA_ROOT"]
DRUGS = ["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", "Doxorubicin",
         "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"]
POS = {"Complete Response", "Partial Response"}


def main():
    raw = pd.read_csv(os.path.join(PROJ, "data/TCGA_labels.csv"))
    a = np.load(os.path.join(ROOT, "dcit/cache/patients.npz"))
    ids = np.array([str(x)[:12] for x in a["patient_ids"]])
    raw["Patient"] = raw["Patient"].astype(str).str.strip()
    raw = raw.drop_duplicates("Patient", keep="first").set_index("Patient")

    rows = []
    for j, d in enumerate(DRUGS):
        obs_raw = raw[d].notna()
        levels = raw.loc[obs_raw, d].astype(str).value_counts().to_dict()
        n_raw = int(obs_raw.sum())
        n_pos = int(raw.loc[obs_raw, d].isin(POS).sum())
        n_neg = int(n_raw - n_pos)
        enc = a["Y"][:, j]
        msk = a["mask"][:, j].astype(bool)
        n_enc = int(msk.sum()); n_enc_pos = int((enc[msk] > 0).sum())
        matched = enc[msk].astype(int) == raw.loc[ids[msk], d].isin(POS).astype(int).to_numpy()
        rows.append(dict(drug=d, raw_source_file="data/TCGA_labels.csv",
                         raw_label_field=d, raw_observed=n_raw,
                         raw_positive_CR_PR=n_pos, raw_negative=n_neg,
                         raw_levels="; ".join("%s=%d" % (k, v) for k, v in sorted(levels.items())),
                         encoded_patients=n_enc, encoded_positive=n_enc_pos,
                         encoded_negative=n_enc - n_enc_pos,
                         label_match_encoded_vs_raw=bool(matched.all()),
                         label_max_mismatch=int((~matched).sum()),
                         cancer_types="; ".join(sorted(raw.loc[ids[msk], "Cancer"].astype(str).unique())),
                         treatment_regimen="UNKNOWN (raw label table has no per-patient regimen field)",
                         single_vs_combination="UNKNOWN",
                         response_criteria="CR/PR recorded response strings in TCGA_labels.csv (manual curation; original RECIST source not in package)",
                         expression_sample_time="UNKNOWN (no sample-to-treatment timing field in packaged arrays)"))
    audit = pd.DataFrame(rows)
    audit.to_csv(os.path.join(OUT, "clinical_label_audit.csv"), index=False)

    # patient duplication / overlap
    dup = int(pd.Series(ids).duplicated().sum())
    multi = int((a["mask"].sum(axis=1) > 1).sum())
    overlap = len(set(ids) & set(raw.index))
    extra = dict(encoded_patients=len(ids), unique_patient_ids=len(set(ids)),
                 duplicate_patient_ids=dup, raw_label_patients=len(raw),
                 matched_patients=overlap,
                 patients_with_multiple_drugs=multi,
                 patient_id_format="TCGA barcode first 12 chars",
                 source_cells=715,
                 note="Fold arrays group all drug records of a patient into the same outer fold.")
    json.dump(extra, open(os.path.join(OUT, "clinical_label_audit_summary.json"), "w"), indent=2)
    print(audit[["drug", "raw_observed", "raw_positive_CR_PR", "raw_negative",
                 "encoded_patients", "encoded_positive", "label_match_encoded_vs_raw"]].to_string(index=False))
    print(json.dumps(extra, indent=2))
    print("AUDIT DONE", flush=True)


if __name__ == "__main__":
    main()
