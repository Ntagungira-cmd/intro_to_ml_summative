"""Feature engineering.

Everything in ``add_engineered_features`` is a *row-wise* transformation (it
looks at one encounter at a time), so it cannot leak information between the
train / validation / test splits and can safely run before splitting.

Anything that has to *learn* from data (e.g. which categories are rare) is
implemented as a scikit-learn transformer so that it is fitted on the training
data only, inside a Pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

# --------------------------------------------------------------------------
# ICD-9 grouping (Strack et al., 2014, Table 2)
# --------------------------------------------------------------------------
ICD9_GROUP_ORDER = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]


def icd9_to_group(code) -> str:
    """Map one ICD-9 code (string) to the clinical group used by Strack et al."""
    if code is None or (isinstance(code, float) and np.isnan(code)) or code is pd.NA:
        return "Missing"
    code = str(code).strip()
    if code in ("", "?", "nan"):
        return "Missing"
    if code[0] in ("V", "E"):          # supplementary / external causes
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if np.floor(value) == 250:
        return "Diabetes"
    if 390 <= value <= 459 or np.floor(value) == 785:
        return "Circulatory"
    if 460 <= value <= 519 or np.floor(value) == 786:
        return "Respiratory"
    if 520 <= value <= 579 or np.floor(value) == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or np.floor(value) == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


# --------------------------------------------------------------------------
# Readable labels for the three *_id columns
# --------------------------------------------------------------------------
_UNKNOWN_LABELS = {"null", "not mapped", "not available", "unknown/invalid", "nan", ""}

_ID_TO_NAME = {
    "admission_type_id": "admission_type",
    "discharge_disposition_id": "discharge_disposition",
    "admission_source_id": "admission_source",
}


def _clean_label(label) -> str:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return "Unknown"
    label = str(label).strip()
    return "Unknown" if label.lower() in _UNKNOWN_LABELS else label


def add_id_labels(df: pd.DataFrame, id_mappings: dict[str, dict[int, str]]) -> pd.DataFrame:
    """Add ``admission_type``, ``discharge_disposition`` and ``admission_source``
    columns with readable labels from IDS_mapping.csv.

    The codes "NULL", "Not Mapped", "Not Available" and "Unknown/Invalid" all
    mean the value was not recorded, so they are merged into one "Unknown" level.
    """
    df = df.copy()
    for id_col, name in _ID_TO_NAME.items():
        mapping = {k: _clean_label(v) for k, v in id_mappings[id_col].items()}
        df[name] = df[id_col].map(mapping).fillna("Unknown").astype(object)
    return df


# --------------------------------------------------------------------------
# Row-wise engineered features
# --------------------------------------------------------------------------
def age_bracket_to_midpoint(age: pd.Series) -> pd.Series:
    """"[70-80)" -> 75.0"""
    bounds = age.astype(str).str.extract(r"\[(\d+)-(\d+)\)").astype(float)
    return (bounds[0] + bounds[1]) / 2


def add_engineered_features(df: pd.DataFrame, id_mappings: dict | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with all engineered columns added.

    New columns
    -----------
    diag_{1,2,3}_group   ICD-9 clinical group
    total_prior_visits   outpatient + emergency + inpatient visits in the prior year
    n_med_changes        number of drugs whose dose went Up or Down
    n_diabetes_drugs     number of drugs prescribed (value != "No")
    age_mid              midpoint of the 10-year age bracket
    admission_type / discharge_disposition / admission_source  readable labels
    prior_inpatient_bin  0 / 1 / 2 / 3+ (only used for the subgroup analysis)
    """
    df = df.copy()
    for col in config.DIAG_COLS:
        df[f"{col}_group"] = df[col].map(icd9_to_group).astype(object)

    df["total_prior_visits"] = df[config.PRIOR_VISIT_COLS].sum(axis=1)

    drugs = df[config.DRUG_COLS]
    df["n_med_changes"] = drugs.isin(["Up", "Down"]).sum(axis=1)
    df["n_diabetes_drugs"] = (drugs != "No").sum(axis=1)

    df["age_mid"] = age_bracket_to_midpoint(df["age"])

    if id_mappings is not None:
        df = add_id_labels(df, id_mappings)

    df["prior_inpatient_bin"] = pd.cut(
        df["number_inpatient"], bins=[-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"]
    ).astype(str)
    return df


def near_constant_columns(df: pd.DataFrame, cols, threshold: float = 0.999) -> list[str]:
    """Columns whose most frequent value covers at least ``threshold`` of rows.

    Call this on the *training* split only.
    """
    out = []
    for c in cols:
        top_share = df[c].value_counts(normalize=True, dropna=False).iloc[0]
        if top_share >= threshold:
            out.append(c)
    return out


def get_feature_columns(feature_set: str, drug_cols: list[str]) -> dict[str, list[str]]:
    """Column lists for the two feature sets used in the ablation.

    ``engineered``: raw numerics + engineered numerics, ICD-9 groups, readable
    ID labels with rare-level collapsing, age as an ordinal midpoint.

    ``baseline``: the original columns only - raw ICD-9 codes, raw ID codes and
    the age bracket as a plain categorical, no aggregates.
    """
    if feature_set == "engineered":
        numeric = config.RAW_NUMERIC_COLS + config.ENGINEERED_NUMERIC_COLS
        categorical = (
            config.BASE_CATEGORICAL_COLS + list(drug_cols)
            + [f"{c}_group" for c in config.DIAG_COLS]
            + ["admission_type", "discharge_disposition", "admission_source"]
        )
        rare = list(config.RARE_COLLAPSE_COLS)
    elif feature_set == "baseline":
        numeric = list(config.RAW_NUMERIC_COLS)
        categorical = (
            config.BASE_CATEGORICAL_COLS + list(drug_cols) + ["age"]
            + config.DIAG_COLS
            + ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
        )
        rare = []
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return {"numeric": numeric, "categorical": categorical, "rare": rare}


def as_model_frame(df: pd.DataFrame, cols: dict[str, list[str]]) -> pd.DataFrame:
    """Select model columns, cast categoricals to plain Python strings
    (so pandas 2.x and 3.x behave the same) and numerics to float."""
    X = df[cols["numeric"] + cols["categorical"]].copy()
    for c in cols["categorical"]:
        X[c] = X[c].astype(str).where(X[c].notna(), "Missing").astype(object)
    for c in cols["numeric"]:
        X[c] = X[c].astype(float)
    return X


# --------------------------------------------------------------------------
# Learned transformer: rare-category collapsing
# --------------------------------------------------------------------------
class RareCategoryCollapser(TransformerMixin, BaseEstimator):
    """Replace categories seen in fewer than ``min_freq`` (fraction) of the
    training rows with ``other_label``. Unseen categories at transform time are
    also mapped to ``other_label``.

    ``min_freq`` is a hyperparameter and can be tuned through a Pipeline,
    e.g. ``prep__rare__collapse__min_freq``.
    """

    def __init__(self, min_freq: float = 0.01, other_label: str = "Other"):
        self.min_freq = min_freq
        self.other_label = other_label

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.keep_ = {}
        for c in X.columns:
            freq = X[c].astype(str).value_counts(normalize=True)
            self.keep_[c] = set(freq[freq >= self.min_freq].index)
        return self

    def transform(self, X):
        is_frame = isinstance(X, pd.DataFrame)
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for c in X.columns:
            s = X[c].astype(str)
            X[c] = s.where(s.isin(self.keep_[c]), self.other_label).astype(object)
        return X if is_frame else X.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)
