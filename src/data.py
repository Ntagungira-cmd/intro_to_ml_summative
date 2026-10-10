"""Loading, cleaning and splitting the Diabetes 130-US Hospitals dataset."""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import config


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def download_uci_zip(raw_dir: Path = config.RAW_DIR, url: str = config.UCI_ZIP_URL,
                     retries: int = 5, wait: float = 5.0) -> None:
    """Download the UCI zip and extract the two CSVs into ``raw_dir``.

    The UCI server occasionally answers with an HTML error page (e.g. 502), so
    the response is checked to be a real zip and the request is retried.
    """
    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            if not zipfile.is_zipfile(io.BytesIO(resp.content)):
                raise ValueError("response is not a zip file (server error page?)")
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in (config.RAW_DATA_FILE, config.IDS_MAPPING_FILE):
                    member = next(m for m in zf.namelist() if m.endswith(name))
                    (raw_dir / name).write_bytes(zf.read(member))
            print(f"Downloaded and extracted to {raw_dir}")
            return
        except Exception as exc:  # network errors, bad zip, ...
            last_error = exc
            print(f"Download attempt {attempt}/{retries} failed: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def extract_zip(zip_path: Path, raw_dir: Path = config.RAW_DIR) -> None:
    """Extract the two CSVs from a local copy of the zip (e.g. on Google Drive)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in (config.RAW_DATA_FILE, config.IDS_MAPPING_FILE):
            member = next(m for m in zf.namelist() if m.endswith(name))
            (raw_dir / name).write_bytes(zf.read(member))
    print(f"Extracted {zip_path} to {raw_dir}")


def load_raw(source: str = "auto", raw_dir: Path = config.RAW_DIR,
             zip_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load ``diabetic_data.csv``.

    source = "local"    read from ``raw_dir`` (fails if the file is missing)
    source = "download" always download from UCI first
    source = "zip"      extract ``zip_path`` (e.g. a Google Drive copy) first
    source = "auto"     use the local file if present, otherwise download
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / config.RAW_DATA_FILE
    if source == "zip":
        extract_zip(zip_path, raw_dir)
    elif source == "download" or (source == "auto" and not csv_path.exists()):
        download_uci_zip(raw_dir)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Put diabetic_data.csv and IDS_mapping.csv in "
            f"{raw_dir} or set DATA_SOURCE='download'."
        )
    # keep_default_na=False: the literal string "None" in A1Cresult and
    # max_glu_serum means "test not measured" and must not become NaN.
    # ICD-9 codes are kept as text ("250.83", "V45").
    df = pd.read_csv(csv_path, dtype={c: str for c in config.DIAG_COLS},
                     keep_default_na=False, na_values=[], low_memory=False)
    return df, raw_dir / config.IDS_MAPPING_FILE


def load_id_mappings(path: Path) -> dict[str, dict[int, str]]:
    """Parse IDS_mapping.csv, which stacks three small tables separated by
    blank lines (admission_type_id, discharge_disposition_id, admission_source_id)."""
    mappings: dict[str, dict[int, str]] = {}
    current = None
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for row in pd.read_csv(io.StringIO(text), header=None, dtype=str,
                           keep_default_na=False).itertuples(index=False):
        key, desc = (row[0] or "").strip(), (row[1] or "").strip()
        if key == "" and desc == "":
            current = None
            continue
        if key.endswith("_id"):
            current = key
            mappings[current] = {}
            continue
        if current is not None and key.isdigit():
            mappings[current][int(key)] = desc
    return mappings


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def replace_question_marks(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace("?", np.nan)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of missing values per column (only columns with any missing)."""
    rep = pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
    })
    return rep[rep.n_missing > 0].sort_values("pct_missing", ascending=False)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.TARGET] = (df[config.TARGET_RAW] == "<30").astype(int)
    df[config.TARGET_3CLASS] = df[config.TARGET_RAW].map({"NO": 0, ">30": 1, "<30": 2})
    return df


def clean(df: pd.DataFrame, log: list | None = None) -> pd.DataFrame:
    """Apply the cleaning decisions documented in the notebook.

    1. drop ``weight`` (~97 % missing)
    2. explicit "Missing" level for payer_code, medical_specialty and race;
       "Not measured" level for A1Cresult and max_glu_serum
    3. drop the few rows with gender "Unknown/Invalid"
    4. drop encounters ending in death or hospice
    5. add binary and 3-class targets
    ``log`` (optional list) receives (step, rows_remaining) tuples.
    """
    log = log if log is not None else []
    log.append(("raw", len(df)))
    df = df.drop(columns=["weight"])
    for c in ["payer_code", "medical_specialty", "race"]:
        df[c] = df[c].fillna("Missing")
    for c in ["A1Cresult", "max_glu_serum"]:
        df[c] = df[c].replace({"None": "Not measured"}).fillna("Not measured")
    df = df[df["gender"] != "Unknown/Invalid"]
    log.append(("drop gender Unknown/Invalid", len(df)))
    df = df[~df["discharge_disposition_id"].isin(config.EXCLUDED_DISCHARGE_IDS)]
    log.append(("drop expired / hospice discharges", len(df)))
    df = add_targets(df)
    return df.reset_index(drop=True)


def first_encounter_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Option (a): keep each patient's first encounter.

    The dataset has no admission date; ``encounter_id`` increases over time,
    so the smallest encounter_id per patient is taken as the first encounter
    (the same approach as Strack et al., 2014).
    """
    return (df.sort_values("encounter_id")
              .drop_duplicates("patient_nbr", keep="first")
              .reset_index(drop=True))


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def stratified_split(y, val_size=0.15, test_size=0.15, seed=config.SEED):
    """Return (train_idx, val_idx, test_idx), stratified on ``y``."""
    idx = np.arange(len(y))
    y = np.asarray(y)
    trval, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(trval, test_size=rel_val, stratify=y[trval], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


def grouped_stratified_split(y, groups, val_size=0.15, test_size=0.15, seed=config.SEED):
    """Patient-grouped, approximately stratified train/val/test split.

    Uses StratifiedGroupKFold: one fold (about ``test_size``) becomes the test
    set, then the remainder is split again for validation. No patient appears
    in more than one split.
    """
    idx = np.arange(len(y))
    y, groups = np.asarray(y), np.asarray(groups)
    n_test = int(round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    trval, test = next(sgkf.split(idx, y, groups))
    rel_val = val_size / (1 - test_size)
    sgkf2 = StratifiedGroupKFold(n_splits=int(round(1 / rel_val)), shuffle=True, random_state=seed)
    tr_rel, val_rel = next(sgkf2.split(trval, y[trval], groups[trval]))
    return np.sort(trval[tr_rel]), np.sort(trval[val_rel]), np.sort(test)


def split_summary(y, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    y = np.asarray(y)
    rows = [{"split": k, "n": len(v), "n_positive": int(y[v].sum()),
             "pct_positive": round(100 * y[v].mean(), 2)} for k, v in splits.items()]
    return pd.DataFrame(rows)
