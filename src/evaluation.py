"""Thresholds, metrics, bootstrap confidence intervals and error analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from . import config


def choose_thresholds(y_true, proba, target_recall: float = 0.70) -> dict:
    """Pick decision thresholds on the *validation* set.

    f1_threshold      maximises F1
    recall_threshold  highest threshold whose recall is still >= target_recall
    """
    precision, recall, thr = precision_recall_curve(y_true, proba)
    p, r = precision[:-1], recall[:-1]          # last point has no threshold
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    i = int(np.nanargmax(f1))
    ok = np.where(r >= target_recall)[0]
    j = int(ok[-1]) if len(ok) else 0           # thresholds are increasing
    return {
        "f1_threshold": float(thr[i]), "val_f1_at_f1_threshold": float(f1[i]),
        "val_precision_at_f1_threshold": float(p[i]), "val_recall_at_f1_threshold": float(r[i]),
        "target_recall": target_recall, "recall_threshold": float(thr[j]),
        "val_precision_at_recall_threshold": float(p[j]), "val_recall_at_recall_threshold": float(r[j]),
    }


def binary_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def bootstrap_ci(y_true, proba, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = config.SEED) -> dict:
    """Percentile bootstrap 95 % CIs for ROC-AUC and PR-AUC (resampling test rows)."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    n = len(y_true)
    roc, pr = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        roc.append(roc_auc_score(y_true[idx], proba[idx]))
        pr.append(average_precision_score(y_true[idx], proba[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"roc_auc_ci_low": np.percentile(roc, lo), "roc_auc_ci_high": np.percentile(roc, hi),
            "pr_auc_ci_low": np.percentile(pr, lo), "pr_auc_ci_high": np.percentile(pr, hi)}


def comparison_table(y_true, probas: dict, thresholds: dict, n_boot: int = 1000,
                     seed: int = config.SEED) -> pd.DataFrame:
    """One row per model: metrics at its tuned (validation max-F1) threshold + CIs."""
    rows = []
    for name, p in probas.items():
        row = {"model": name, **binary_metrics(y_true, p, thresholds[name])}
        row.update(bootstrap_ci(y_true, p, n_boot=n_boot, seed=seed))
        rows.append(row)
    cols = ["model", "threshold", "accuracy", "balanced_accuracy", "precision", "recall", "f1",
            "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "pr_auc", "pr_auc_ci_low",
            "pr_auc_ci_high", "brier"]
    return pd.DataFrame(rows)[cols].sort_values("pr_auc", ascending=False).reset_index(drop=True)


def format_ci_table(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Compact version for the report: 'ROC-AUC [low, high]' strings."""
    out = df[["model", "threshold", "accuracy", "balanced_accuracy", "precision",
              "recall", "f1"]].round(digits).copy()
    f = f"{{:.{digits}f}}"
    for m in ("roc_auc", "pr_auc"):
        out[m + " (95% CI)"] = [f"{f.format(v)} [{f.format(l)}, {f.format(h)}]"
                                for v, l, h in zip(df[m], df[m + "_ci_low"], df[m + "_ci_high"])]
    out["brier"] = df["brier"].round(digits)
    return out


def subgroup_error_analysis(meta: pd.DataFrame, y_true, pred, group_cols,
                            min_positives: int = 30, recall_gap: float = 0.10) -> pd.DataFrame:
    """Confusion counts, recall, precision, FNR and FPR per subgroup.

    A subgroup is flagged when its recall is more than ``recall_gap`` below the
    overall recall and it has at least ``min_positives`` positive cases (so
    the estimate is not pure noise)."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    overall_recall = recall_score(y_true, pred)
    rows = []
    for col in group_cols:
        for level, idx in meta.groupby(col, observed=True).indices.items():
            yt, yp = y_true[idx], pred[idx]
            tp = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
            rec = tp / (tp + fn) if tp + fn else np.nan
            rows.append({
                "attribute": col, "group": str(level), "n": len(idx), "n_positive": tp + fn,
                "prevalence": (tp + fn) / len(idx), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                "recall": rec, "precision": tp / (tp + fp) if tp + fp else np.nan,
                "fnr": fn / (tp + fn) if tp + fn else np.nan,
                "fpr": fp / (fp + tn) if fp + tn else np.nan,
                "share_of_all_FN": fn, "share_of_all_FP": fp,
            })
    df = pd.DataFrame(rows)
    df["share_of_all_FN"] = df["share_of_all_FN"] / max(1, int(((y_true == 1) & (pred == 0)).sum()))
    df["share_of_all_FP"] = df["share_of_all_FP"] / max(1, int(((y_true == 0) & (pred == 1)).sum()))
    df["recall_gap_vs_overall"] = df["recall"] - overall_recall
    df["flag_low_recall"] = (df["recall_gap_vs_overall"] < -recall_gap) & (df["n_positive"] >= min_positives)
    df.attrs["overall_recall"] = overall_recall
    return df
