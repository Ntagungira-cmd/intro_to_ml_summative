"""Project-wide configuration: paths, seeds and column definitions."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FIG_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
TUNER_DIR = PROJECT_ROOT / "tuner_logs"

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
RAW_DATA_FILE = "diabetic_data.csv"
IDS_MAPPING_FILE = "IDS_mapping.csv"

SEED = 42

# --------------------------------------------------------------------------
# Column definitions (names as they appear in diabetic_data.csv)
# --------------------------------------------------------------------------
TARGET_RAW = "readmitted"
TARGET = "readmit_30"          # binary target: 1 if "<30"
TARGET_3CLASS = "readmit_3"    # 0 = NO, 1 = >30, 2 = <30
CLASS_NAMES_3 = ["NO", ">30", "<30"]

ID_COLS = ["encounter_id", "patient_nbr"]

# Discharge dispositions that mean the patient died or went to hospice
# (11 Expired, 13/14 Hospice, 19/20/21 Expired ... hospice).
EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# The 23 medication columns in the dataset.
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]
PRIOR_VISIT_COLS = ["number_outpatient", "number_emergency", "number_inpatient"]

RAW_NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
ENGINEERED_NUMERIC_COLS = [
    "age_mid", "total_prior_visits", "n_med_changes", "n_diabetes_drugs",
]

# Categoricals shared by both feature sets
BASE_CATEGORICAL_COLS = [
    "race", "gender", "payer_code", "medical_specialty",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
]

# Columns whose rare levels are collapsed into "Other" (threshold is tunable)
RARE_COLLAPSE_COLS = [
    "medical_specialty", "admission_source", "discharge_disposition",
]

# High-cardinality categoricals that get learned embeddings in the DL model
EMBEDDING_COLS = [
    "diag_1_group", "diag_2_group", "diag_3_group", "medical_specialty",
    "admission_source", "discharge_disposition", "admission_type", "payer_code",
]

# Columns used for the subgroup error analysis / fairness check
SUBGROUP_COLS = ["age", "race", "gender", "admission_type", "prior_inpatient_bin"]


def set_global_seed(seed: int = SEED, deterministic_tf: bool = True) -> None:
    """Fix Python, NumPy and TensorFlow seeds.

    TensorFlow is imported lazily so that this module can be used in a
    scikit-learn-only session.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)  # python + numpy + tf
        if deterministic_tf:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception as exc:  # pragma: no cover - depends on TF build
                print(f"Op determinism not enabled: {exc}")
    except ImportError:
        pass


def ensure_dirs() -> None:
    for d in (RAW_DIR, FIG_DIR, RESULTS_DIR, MODELS_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)
