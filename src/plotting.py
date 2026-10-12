"""Plotting helpers. Every figure is saved as a 300-dpi PNG in figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve)

from . import config

# Fixed categorical order (validated colour-blind-safe on adjacent pairs).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8985"
SEQ_CMAP = "Blues"


def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=PALETTE), "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#e6e5e1", "lines.linewidth": 2,
        "legend.frameon": False,
    })


def save(fig, name: str, fig_dir: Path = config.FIG_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"saved {path.relative_to(config.PROJECT_ROOT)}")
    return path


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
def class_distribution(series: pd.Series, title: str, ax=None):
    ax = ax or plt.gca()
    counts = series.value_counts()
    ax.bar(counts.index.astype(str), counts.values, color=PALETTE[0], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({100 * v / counts.sum():.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(title); ax.set_ylabel("Encounters"); ax.margins(y=0.18)
    return ax


def missingness_bar(rep: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    rep = rep.sort_values("pct_missing")
    ax.barh(rep.index, rep["pct_missing"], color=PALETTE[0], height=0.6)
    for i, v in enumerate(rep["pct_missing"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% missing ('?' in the raw file)"); ax.set_xlim(0, 105)
    ax.set_title("Missing values per column")
    return ax


def rate_by_category(df: pd.DataFrame, col: str, target: str, ax=None, top: int = 12,
                     min_count: int = 30):
    """Readmission rate per level (levels with fewer than ``min_count`` rows hidden)."""
    ax = ax or plt.gca()
    g = df.groupby(col, observed=True)[target].agg(["mean", "size"])
    g = g[g["size"] >= min_count].sort_values("size", ascending=False).head(top).sort_values("mean")
    labels = [l if len(l) <= 38 else l[:36] + "…" for l in g.index.astype(str)]
    ax.barh(labels, 100 * g["mean"], color=PALETTE[0], height=0.6)
    ax.axvline(100 * df[target].mean(), color=NEUTRAL, ls="--", lw=1.2, label="overall")
    ax.set_xlabel("% readmitted <30 days"); ax.set_title(col); ax.legend(loc="lower right")
    return ax


# --------------------------------------------------------------------------
# Curves on one plot
# --------------------------------------------------------------------------
def _style_for(i, name, dl_names=()):
    return {"color": PALETTE[i % len(PALETTE)], "ls": "--" if name in dl_names else "-"}


def roc_overlay(y_true, probas: dict, title: str, dl_names=(), ax=None):
    ax = ax or plt.gca()
    for i, (name, p) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, p):.3f})", **_style_for(i, name, dl_names))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls=":", lw=1.2, label="chance (0.500)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title); ax.legend(title="ROC-AUC", loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    return ax


def pr_overlay(y_true, probas: dict, title: str, dl_names=(), ax=None):
    ax = ax or plt.gca()
    for i, (name, p) in enumerate(probas.items()):
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=f"{name} ({average_precision_score(y_true, p):.3f})",
                **_style_for(i, name, dl_names))
    base = np.mean(y_true)
    ax.axhline(base, color=NEUTRAL, ls=":", lw=1.2, label=f"prevalence ({base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(title="PR-AUC (AP)", loc="upper right", fontsize=8)
    return ax


def confusion_pair(y_true, pred, title: str, labels=("Not <30", "<30")):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, norm, sub in zip(axes, [None, "true"], ["counts", "normalised by true class"]):
        cm = confusion_matrix(y_true, pred, normalize=norm)
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, cmap=SEQ_CMAP, colorbar=False, values_format=".2f" if norm else "d")
        ax.set_title(f"{title}\n({sub})"); ax.grid(False)
    fig.tight_layout()
    return fig


def threshold_curve(y_true, proba, thresholds: dict, title: str, ax=None):
    """Precision, recall and F1 against the decision threshold (validation set)."""
    ax = ax or plt.gca()
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    ax.plot(thr, prec[:-1], label="precision", color=PALETTE[0])
    ax.plot(thr, rec[:-1], label="recall", color=PALETTE[1])
    ax.plot(thr, f1, label="F1", color=PALETTE[2])
    ax.axvline(thresholds["f1_threshold"], color=NEUTRAL, ls="--", lw=1.2)
    ax.text(thresholds["f1_threshold"], 0.97, " max-F1", fontsize=8, va="top")
    ax.axvline(thresholds["recall_threshold"], color=NEUTRAL, ls=":", lw=1.2)
    ax.text(thresholds["recall_threshold"], 0.88, f" recall={thresholds['target_recall']:.2f}",
            fontsize=8, va="top")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.set_title(title); ax.legend(loc="center right")
    return ax


def calibration_overlay(y_true, probas: dict, title: str, n_bins: int = 10, dl_names=(), ax=None):
    ax = ax or plt.gca()
    for i, (name, p) in enumerate(probas.items()):
        frac, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=4, label=name, **_style_for(i, name, dl_names))
    lim = max(0.5, max(np.quantile(p, 0.99) for p in probas.values()))
    ax.plot([0, lim], [0, lim], color=NEUTRAL, ls=":", lw=1.2, label="perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction <30")
    ax.set_title(title); ax.legend(fontsize=8)
    return ax


def learning_curve_plot(train_sizes, train_scores, val_scores, title: str, ax=None, metric="PR-AUC"):
    ax = ax or plt.gca()
    for scores, label, color in [(train_scores, "training", PALETTE[0]),
                                 (val_scores, "cross-validation", PALETTE[1])]:
        m, s = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(train_sizes, m, marker="o", ms=5, label=label, color=color)
        ax.fill_between(train_sizes, m - s, m + s, alpha=0.15, color=color, lw=0)
    ax.set_xlabel("Training examples"); ax.set_ylabel(metric); ax.set_title(title); ax.legend()
    return ax


def history_plot(hist: pd.DataFrame, title: str):
    """Training vs validation loss and PR-AUC / ROC-AUC for one Keras run."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    epochs = np.arange(1, len(hist) + 1)
    for ax, key, lab in zip(axes, ["loss", "pr_auc", "roc_auc"], ["Loss", "PR-AUC", "ROC-AUC"]):
        ax.plot(epochs, hist[key], label="training", color=PALETTE[0])
        ax.plot(epochs, hist[f"val_{key}"], label="validation", color=PALETTE[1])
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab); ax.set_title(lab)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    best = int(hist["val_pr_auc"].values.argmax()) + 1
    for ax in axes:
        ax.axvline(best, color=NEUTRAL, ls=":", lw=1.2)
    axes[0].legend()
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig


def importance_bar(df: pd.DataFrame, title: str, top: int = 20, ax=None):
    ax = ax or plt.gca()
    d = df.head(top).iloc[::-1]
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color=PALETTE[0],
            height=0.6, error_kw={"lw": 1, "ecolor": NEUTRAL})
    ax.set_xlabel("Drop in PR-AUC when shuffled"); ax.set_title(title)
    return ax


def subgroup_recall_plot(err: pd.DataFrame, title: str):
    attrs = err["attribute"].unique()
    fig, axes = plt.subplots(1, len(attrs), figsize=(3.2 * len(attrs), 4), sharex=True)
    overall = err.attrs.get("overall_recall", np.nan)
    for ax, a in zip(np.atleast_1d(axes), attrs):
        d = err[(err.attribute == a) & (err.n_positive > 0)].sort_values("group")
        colors = [PALETTE[7] if f else PALETTE[0] for f in d["flag_low_recall"]]
        ax.barh(d["group"], d["recall"], color=colors, height=0.6)
        ax.axvline(overall, color=NEUTRAL, ls="--", lw=1.2)
        ax.set_title(a); ax.set_xlim(0, 1); ax.set_xlabel("Recall")
        for i, (r, n) in enumerate(zip(d["recall"], d["n_positive"])):
            ax.text(min(r, 0.85) + 0.02, i, f"n+={n}", va="center", fontsize=7)
    fig.suptitle(f"{title}  (dashed = overall recall {overall:.2f}; red = flagged; "
                 f"n+ = positives in group; groups without positives omitted)", y=1.02)
    fig.tight_layout()
    return fig
