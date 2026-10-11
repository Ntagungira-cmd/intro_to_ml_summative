"""Traditional ML models, search spaces and tuning helpers."""
from __future__ import annotations

import time
import warnings

import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.svm import LinearSVC

from . import config
from .pipelines import build_model_pipeline

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}

# scikit-learn >= 1.8 deprecates LogisticRegression(penalty=...) in favour of
# l1_ratio (0 = L2, 1 = L1). Support both so the code runs on older Colab images.
_LR_USES_L1_RATIO = "deprecated" in str(LogisticRegression().get_params().get("penalty"))


def _lgbm():
    from lightgbm import LGBMClassifier  # imported lazily (needs OpenMP on macOS)
    return LGBMClassifier


def make_estimator(name: str, strategy: str = "none", seed: int = config.SEED,
                   n_classes: int = 2):
    """Untuned estimator with sensible defaults. ``strategy='class_weight'``
    switches on balanced class weights where the estimator supports it."""
    cw = "balanced" if strategy == "class_weight" else None
    if name == "Dummy":
        return DummyClassifier(strategy="prior")
    if name == "LogReg":
        kw = {"l1_ratio": 0.0} if _LR_USES_L1_RATIO else {"penalty": "l2"}
        return LogisticRegression(solver="liblinear", C=1.0, max_iter=2000,
                                  class_weight=cw, random_state=seed, **kw)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      max_samples=0.7, class_weight=cw,
                                      n_jobs=1, random_state=seed)
    if name == "HistGB":
        return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300,
                                              class_weight=cw, random_state=seed)
    if name == "LightGBM":
        return _lgbm()(n_estimators=400, learning_rate=0.03, num_leaves=31,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                       class_weight=cw, n_jobs=1, random_state=seed, verbose=-1)
    if name == "LinearSVM":
        return CalibratedClassifierCV(
            LinearSVC(C=0.1, class_weight=cw, max_iter=5000, random_state=seed),
            method="sigmoid", cv=3)
    raise ValueError(name)


def search_space(name: str) -> dict:
    """Hyperparameter distributions (prefixed for the pipeline). The rare-
    category threshold is tuned for every model as part of feature engineering."""
    rare = {"prep__rare__collapse__min_freq": [0.005, 0.01, 0.02, 0.05]}
    if name == "LogReg":
        reg = ({"clf__l1_ratio": [0.0, 1.0]} if _LR_USES_L1_RATIO
               else {"clf__penalty": ["l1", "l2"]})
        return {**rare, **reg, "clf__C": loguniform(1e-3, 10)}
    if name == "RandomForest":
        return {**rare,
                "clf__n_estimators": [200, 300],
                "clf__max_depth": [None, 12, 20],
                "clf__min_samples_leaf": [1, 5, 20, 50],
                "clf__max_features": ["sqrt", 0.2, 0.4],
                "clf__max_samples": [0.5, 0.8]}
    if name == "HistGB":
        return {**rare,
                "clf__learning_rate": loguniform(0.02, 0.2),
                "clf__max_iter": [200, 400],
                "clf__max_leaf_nodes": [15, 31, 63],
                "clf__min_samples_leaf": [20, 50, 100],
                "clf__l2_regularization": loguniform(1e-3, 10)}
    if name == "LightGBM":
        return {**rare,
                "clf__n_estimators": [200, 400, 800],
                "clf__learning_rate": loguniform(0.01, 0.1),
                "clf__num_leaves": [15, 31, 63],
                "clf__min_child_samples": [20, 50, 100],
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.4, 0.6),
                "clf__reg_lambda": loguniform(1e-3, 10)}
    if name == "LinearSVM":
        return {**rare, "clf__estimator__C": loguniform(1e-4, 1)}
    return {}


def compare_imbalance_strategies(model_names, X, y, cols, cv, strategies,
                                 n_jobs=-1, seed=config.SEED) -> pd.DataFrame:
    """Cross-validate each (model, strategy) pair with default hyperparameters."""
    rows = []
    for name in model_names:
        for strategy in strategies:
            pipe = build_model_pipeline(make_estimator(name, strategy, seed), cols, strategy, seed=seed)
            t0 = time.time()
            res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
            rows.append({
                "model": name, "strategy": strategy,
                "cv_pr_auc_mean": res["test_pr_auc"].mean(), "cv_pr_auc_std": res["test_pr_auc"].std(),
                "cv_roc_auc_mean": res["test_roc_auc"].mean(), "cv_roc_auc_std": res["test_roc_auc"].std(),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"{name:12s} {strategy:13s} PR-AUC={rows[-1]['cv_pr_auc_mean']:.4f} "
                  f"ROC-AUC={rows[-1]['cv_roc_auc_mean']:.4f} ({rows[-1]['seconds']}s)")
    return pd.DataFrame(rows)


def tune_model(name, X, y, cols, cv, strategy, n_iter, groups=None,
               n_jobs=-1, seed=config.SEED) -> tuple:
    """RandomizedSearchCV on PR-AUC (refit), with ROC-AUC logged as well.

    Returns (fitted search or pipeline, results DataFrame)."""
    pipe = build_model_pipeline(make_estimator(name, strategy, seed), cols, strategy, seed=seed)
    space = search_space(name)
    t0 = time.time()
    if not space:  # Dummy: nothing to tune
        res = cross_validate(pipe, X, y, cv=cv, groups=groups, scoring=SCORING, n_jobs=n_jobs)
        pipe.fit(X, y)
        df = pd.DataFrame([{"model": name, "params": "{}",
                            "mean_test_pr_auc": res["test_pr_auc"].mean(),
                            "std_test_pr_auc": res["test_pr_auc"].std(),
                            "mean_test_roc_auc": res["test_roc_auc"].mean(),
                            "std_test_roc_auc": res["test_roc_auc"].std(),
                            "mean_fit_time": res["fit_time"].mean(), "rank_test_pr_auc": 1}])
        print(f"{name}: CV PR-AUC={df.mean_test_pr_auc[0]:.4f}")
        return pipe, df
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring=SCORING, refit="pr_auc",
                                cv=cv, n_jobs=n_jobs, random_state=seed, error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        search.fit(X, y, groups=groups)
    cvr = pd.DataFrame(search.cv_results_)
    df = pd.DataFrame({
        "model": name,
        "params": [str({k: getattr(v, "item", lambda: v)() for k, v in p.items()})
                   for p in cvr["params"]],
        "mean_test_pr_auc": cvr["mean_test_pr_auc"], "std_test_pr_auc": cvr["std_test_pr_auc"],
        "mean_test_roc_auc": cvr["mean_test_roc_auc"], "std_test_roc_auc": cvr["std_test_roc_auc"],
        "mean_fit_time": cvr["mean_fit_time"], "rank_test_pr_auc": cvr["rank_test_pr_auc"],
    })
    print(f"{name}: best CV PR-AUC={search.best_score_:.4f} "
          f"({len(df)} configs, {time.time() - t0:.0f}s)\n  best params: {search.best_params_}")
    return search, df


def best_params_without_prefix(params: dict) -> dict:
    return {k.replace("clf__", ""): v for k, v in params.items() if k.startswith("clf__")}
