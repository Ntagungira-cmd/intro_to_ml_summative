# Predicting 30-Day Hospital Readmission of Diabetic Patients

Machine Learning Project: traditional machine learning (scikit-learn, LightGBM) compared with deep learning (TensorFlow/Keras) for predicting whether a diabetic inpatient is readmitted within 30 days of discharge.

| | |
|---|---|
| **Main task** | Binary classification: `readmitted == "<30"` (positive) vs. `">30"` / `"NO"` |
| **Secondary experiment** | 3-class classification (`<30` / `>30` / `NO`) |
| **Main metric** | PR-AUC (average precision), with ROC-AUC, F1, recall, precision, balanced accuracy, Brier score |
| **Notebook** | [`readmission_prediction.ipynb`](readmission_prediction.ipynb) |

**Links**
- GitHub repository: `<ADD LINK>`
- Dataset zip on Google Drive: `<ADD LINK>`
- Video presentation: `<ADD LINK>`
- Open in Colab: `https://colab.research.google.com/github/<your-github-username>/<your-repo-name>/blob/main/readmission_prediction.ipynb`

---

## Dataset

**Diabetes 130-US Hospitals for Years 1999–2008**, UCI Machine Learning Repository, ID 296:
<https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>

101,766 inpatient encounters of diabetic patients at 130 US hospitals, 50 columns (demographics, admission/discharge details, diagnoses, lab tests, 23 diabetes medications, readmission outcome).

> Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014, Article 781670. https://doi.org/10.1155/2014/781670

The data is **not** stored in this repository. Get it in one of three ways:

1. **Automatic (default):** the notebook downloads the zip from UCI when `data/raw/diabetic_data.csv` is missing (`DATA_SOURCE = "auto"`).
2. **Manual:** download the zip from the UCI page above (or from the Google Drive link) and put `diabetic_data.csv` and `IDS_mapping.csv` in `data/raw/`.
3. **Google Drive in Colab:** set `DATA_SOURCE = "zip"`, `MOUNT_GOOGLE_DRIVE = True` and `ZIP_PATH = "/content/drive/MyDrive/.../diabetes.zip"`.

---

## How to run

### Google Colab
1. Push this repository to GitHub and set `REPO_URL` in the first code cell of the notebook.
2. Open the notebook in Colab (`File → Open notebook → GitHub`) and pick a GPU runtime (`Runtime → Change runtime type → T4 GPU`).
3. `Runtime → Run all`. The first cell clones the repository and installs the pinned packages Colab does not have; Colab's own TensorFlow is kept.

### Local
```bash
git clone <your-repo-url> && cd <repo>
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: LightGBM needs OpenMP ->  brew install libomp
jupyter lab readmission_prediction.ipynb
```

### Configuration (Section 1.2 of the notebook)
| Setting | Meaning |
|---|---|
| `FAST_MODE = True` | 12,000-row subsample, tiny search budgets, 4 epochs: checks the whole notebook in a couple of minutes |
| `FAST_MODE = False` | full run used for the report |
| `DATA_SOURCE` | `"auto"`, `"local"`, `"download"` or `"zip"` |
| `N_ITER`, `DL_EPOCHS`, `TUNER_MAX_EPOCHS`, `N_BOOT` | search / training / bootstrap budgets |

**Reproducibility.** One seed (42) is used for Python, NumPy and TensorFlow (`keras.utils.set_random_seed`), with TensorFlow op determinism on, and every splitter, model and search gets `random_state=42`. Package versions are pinned in `requirements.txt` and written to `results/environment_versions.csv` on every run. Results can still differ slightly between CPU and GPU, or between library versions.

---

## Method overview

1. **Cleaning:** `?` → NaN; drop `weight` (97 % missing); explicit `Missing` level for `payer_code`, `medical_specialty`, `race`; `Not measured` level for `A1Cresult` / `max_glu_serum`; remove encounters ending in death or hospice (discharge IDs 11, 13, 14, 19, 20, 21).
2. **Patient leakage:** main pipeline keeps each patient's first encounter (option a, as in Strack et al.); keeping all encounters with patient-grouped CV (option b), and a naive split, are compared in a separate experiment.
3. **Split:** stratified 70 / 15 / 15 train / validation / test. The test set is used once, at the end.
4. **Features:** ICD-9 diagnosis groups, total prior visits, number of medication changes, number of diabetes drugs, age midpoint, readable admission/discharge labels, rare-category collapsing (threshold tuned), plus an ablation without these features.
5. **Imbalance:** none vs. class weights vs. random undersampling vs. SMOTE-NC (inside imblearn Pipelines); decision thresholds chosen on the validation set (max-F1 and recall ≥ 0.70).
6. **Traditional ML:** Dummy, Logistic Regression (L1/L2), Random Forest, HistGradientBoosting, LightGBM, calibrated Linear SVM; `RandomizedSearchCV` scored on PR-AUC.
7. **Deep learning:** `tf.data` pipelines; (1) Sequential baseline MLP, (2) Functional deep MLP with BatchNorm/Dropout/L2, (3) Functional embedding model, KerasTuner Hyperband tuning, (4) class weights vs. focal loss.
8. **Evaluation:** test-set table with 95 % bootstrap CIs, ROC/PR curves, confusion matrices, learning curves, calibration/Brier, permutation importance, SHAP, subgroup error analysis (fairness check), 3-class experiment.

---

## Results (test set, fill in after the final run)

Values from `results/test_comparison_report_table.csv`, each model at its validation max-F1 threshold.

| Model | Precision | Recall | F1 | ROC-AUC [95 % CI] | PR-AUC [95 % CI] | Brier |
|---|---|---|---|---|---|---|
| Dummy (prior) | | | | | | |
| Logistic Regression | | | | | | |
| Random Forest | | | | | | |
| HistGradientBoosting | | | | | | |
| LightGBM | | | | | | |
| Linear SVM (calibrated) | | | | | | |
| Best DL model: `<name>` | | | | | | |

3-class experiment (macro-F1, test): `<value>`

---

## Repository structure

```
.
├── readmission_prediction.ipynb   main notebook (runs locally and in Colab)
├── src/                           reusable modules imported by the notebook
│   ├── config.py                  paths, seeds, column lists
│   ├── data.py                    download / load / clean / split
│   ├── features.py                feature engineering, RareCategoryCollapser
│   ├── pipelines.py               sklearn + imblearn preprocessing pipelines, DL preprocessor
│   ├── ml_models.py               model zoo, search spaces, tuning helpers
│   ├── dl_models.py               tf.data pipeline, Keras models, KerasTuner HyperModel
│   ├── evaluation.py              thresholds, metrics, bootstrap CIs, subgroup analysis
│   └── plotting.py                figure helpers (300-dpi PNG)
├── figures/                       all figures (fig01_... to fig28_...)
├── results/                       all tables as CSV
├── models/
│   ├── best_traditional_model.joblib   full fitted pipeline (preprocessing + model)
│   ├── best_dl_model.keras             native Keras format
│   ├── best_dl_model_savedmodel/       TensorFlow SavedModel export (git-ignored)
│   ├── dl_dense_preprocessor.joblib    fitted preprocessors for the networks
│   ├── dl_embedding_preprocessor.joblib
│   ├── feature_columns.json            columns the models expect
│   └── thresholds.json                 validation-chosen decision thresholds
├── data/raw/                      dataset goes here (git-ignored)
├── requirements.txt
└── .gitignore
```

If `best_traditional_model.joblib` is over GitHub's 100 MB limit (possible when a Random Forest wins), upload it to Google Drive and link it here instead.

## Inference

The last section of the notebook loads the saved models from `models/` and scores a few raw rows of `diabetic_data.csv`:

```python
import joblib, json
from src import data, features
model = joblib.load("models/best_traditional_model.joblib")
cols = json.load(open("models/feature_columns.json"))
thr = json.load(open("models/thresholds.json"))["best_traditional_model"]["f1_threshold"]

raw, ids_path = data.load_raw("local")
df = features.add_engineered_features(data.clean(data.replace_question_marks(raw.head(10))),
                                      data.load_id_mappings(ids_path))
proba = model.predict_proba(features.as_model_frame(df, cols))[:, 1]
flag = proba >= thr
```
