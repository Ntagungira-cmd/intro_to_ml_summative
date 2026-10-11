"""Preprocessing pipelines.

Every model uses the same three-stage structure, built with imbalanced-learn's
Pipeline so that resampling only ever happens on the training part of each
fold:

    prep     ColumnTransformer -> pandas DataFrame
             numeric:   median impute + StandardScaler
             rare cols: "Missing" impute + RareCategoryCollapser (tunable)
             other cat: "Missing" impute
    sampler  "passthrough" | RandomUnderSampler | SMOTENC
             (SMOTE-NC needs the categoricals *before* one-hot encoding,
              which is why encoding is a separate, later step)
    encode   OneHotEncoder on the categoricals, numerics passed through
    clf      the estimator
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config
from .features import RareCategoryCollapser

IMBALANCE_STRATEGIES = ["none", "class_weight", "undersample", "smote_nc"]


def build_prep(cols: dict, rare_min_freq: float = 0.01) -> ColumnTransformer:
    rare = [c for c in cols["rare"] if c in cols["categorical"]]
    other_cat = [c for c in cols["categorical"] if c not in rare]
    transformers = [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), cols["numeric"]),
        ("cat", SimpleImputer(strategy="constant", fill_value="Missing"), other_cat),
    ]
    if rare:
        transformers.insert(1, ("rare", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("collapse", RareCategoryCollapser(min_freq=rare_min_freq)),
        ]), rare))
    ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def build_encoder(cols: dict, min_frequency=None) -> ColumnTransformer:
    """One-hot encode categoricals. ``min_frequency`` is only used for the
    'baseline' ablation set, where raw ICD-9 codes have hundreds of levels."""
    handle = "infrequent_if_exist" if min_frequency else "ignore"
    ohe = OneHotEncoder(handle_unknown=handle, min_frequency=min_frequency,
                        sparse_output=False, dtype="float32")
    return ColumnTransformer([("onehot", ohe, cols["categorical"])],
                             remainder="passthrough", verbose_feature_names_out=False)


def build_sampler(strategy: str, cols: dict, seed: int = config.SEED):
    if strategy in ("none", "class_weight"):
        return "passthrough"
    if strategy == "undersample":
        return RandomUnderSampler(random_state=seed)
    if strategy == "smote_nc":
        return SMOTENC(categorical_features=list(cols["categorical"]), random_state=seed)
    raise ValueError(strategy)


def build_model_pipeline(estimator, cols: dict, strategy: str = "none",
                         rare_min_freq: float = 0.01, onehot_min_frequency=None,
                         seed: int = config.SEED) -> ImbPipeline:
    return ImbPipeline([
        ("prep", build_prep(cols, rare_min_freq)),
        ("sampler", build_sampler(strategy, cols, seed)),
        ("encode", build_encoder(cols, onehot_min_frequency)),
        ("clf", estimator),
    ])


def build_dense_preprocessor(cols: dict, rare_min_freq: float = 0.01) -> Pipeline:
    """prep + encode without a model: produces the dense matrix fed to the MLPs."""
    return Pipeline([("prep", build_prep(cols, rare_min_freq)),
                     ("encode", build_encoder(cols))])


class EmbeddingPreprocessor:
    """Preprocessing for the Keras embedding model.

    ``transform`` returns a dict of arrays that matches the model's named
    inputs: ``numeric`` (scaled numerics + one-hot low-cardinality
    categoricals) and one int32 index array per embedding column, where
    index 0 is reserved for unknown categories.
    """

    def __init__(self, cols: dict, embedding_cols=None, rare_min_freq: float = 0.01):
        self.cols = cols
        self.embedding_cols = [c for c in (embedding_cols or config.EMBEDDING_COLS)
                               if c in cols["categorical"]]
        self.rare_min_freq = rare_min_freq

    def fit(self, X, y=None):
        dense_cols = {
            "numeric": self.cols["numeric"],
            "categorical": [c for c in self.cols["categorical"] if c not in self.embedding_cols],
            "rare": [],
        }
        self.dense_ = build_dense_preprocessor(dense_cols).fit(X)
        emb_rare = [c for c in self.embedding_cols if c in self.cols["rare"]]
        steps = [("impute", SimpleImputer(strategy="constant", fill_value="Missing"))]
        self.collapse_ = ColumnTransformer(
            [("rare", RareCategoryCollapser(self.rare_min_freq), emb_rare)],
            remainder="passthrough", verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        self.ordinal_ = Pipeline(steps + [
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        Xe = self.collapse_.fit_transform(X[self.embedding_cols])[self.embedding_cols]
        self.ordinal_.fit(Xe)
        cats = self.ordinal_.named_steps["ord"].categories_
        self.cardinalities_ = {c: len(k) + 1 for c, k in zip(self.embedding_cols, cats)}
        self.n_numeric_ = self.dense_.transform(X.iloc[:2]).shape[1]
        return self

    def transform(self, X) -> dict:
        out = {"numeric": self.dense_.transform(X).astype("float32")}
        Xe = self.collapse_.transform(X[self.embedding_cols])[self.embedding_cols]
        codes = self.ordinal_.transform(Xe).astype("int32") + 1  # unknown (-1) -> 0
        for i, c in enumerate(self.embedding_cols):
            out[c] = codes[:, i]
        return out

    def fit_transform(self, X, y=None) -> dict:
        return self.fit(X).transform(X)
