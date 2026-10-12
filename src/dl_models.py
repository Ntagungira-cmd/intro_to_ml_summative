"""Deep learning: tf.data pipelines, Keras architectures and training helpers."""
from __future__ import annotations

import time

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from . import config

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------
# tf.data input pipeline
# --------------------------------------------------------------------------
def make_dataset(X, y=None, batch_size: int = 512, shuffle: bool = False,
                 seed: int = config.SEED) -> tf.data.Dataset:
    """from_tensor_slices -> cache -> (shuffle) -> batch -> prefetch.

    ``X`` is a 2-D array (MLPs) or a dict of arrays keyed by input name
    (embedding model). Caching happens before shuffling so the data is
    materialised once while every epoch still sees a new order.
    """
    if y is not None:
        y = np.asarray(y, dtype="float32").reshape(-1, 1)
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.cache()
    if shuffle:
        n = len(y) if y is not None else len(next(iter(X.values())) if isinstance(X, dict) else X)
        ds = ds.shuffle(buffer_size=n, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


# --------------------------------------------------------------------------
# Metrics, losses, callbacks
# --------------------------------------------------------------------------
def make_metrics():
    return [
        keras.metrics.AUC(curve="PR", name="pr_auc"),
        keras.metrics.AUC(curve="ROC", name="roc_auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]


def make_loss(kind: str = "bce", alpha: float = 0.75, gamma: float = 2.0):
    if kind == "bce":
        return keras.losses.BinaryCrossentropy()
    if kind == "focal":
        # alpha weights the positive class (apply_class_balancing=True)
        return keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,
                                                    alpha=alpha, gamma=gamma)
    raise ValueError(kind)


def compile_model(model, learning_rate=1e-3, loss="bce", **loss_kw):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=make_loss(loss, **loss_kw), metrics=make_metrics())
    return model


def make_callbacks(run_name: str, patience: int = 6, checkpoint: bool = True):
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5,
                                          patience=max(2, patience // 3), min_lr=1e-5, verbose=0),
    ]
    if checkpoint:
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cbs.append(keras.callbacks.ModelCheckpoint(
            str(config.CHECKPOINT_DIR / f"{run_name}.keras"), monitor="val_pr_auc",
            mode="max", save_best_only=True, verbose=0))
    return cbs


def balanced_class_weight(y) -> dict:
    y = np.asarray(y)
    n, pos = len(y), y.sum()
    return {0: float(n / (2 * (n - pos))), 1: float(n / (2 * pos))}


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------
def build_baseline_mlp(n_features: int, units: int = 64) -> keras.Model:
    """Experiment 1 - Sequential API: one hidden layer."""
    return keras.Sequential([
        keras.Input(shape=(n_features,), name="features"),
        layers.Dense(units, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="exp1_baseline_mlp")


def _dense_block(x, units, dropout, l2, name):
    x = layers.Dense(units, kernel_regularizer=regularizers.l2(l2), name=f"{name}_dense")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return layers.Dropout(dropout, name=f"{name}_drop")(x)


def build_deep_mlp(n_features: int, units=(256, 128, 64), dropout: float = 0.3,
                   l2: float = 1e-4, name: str = "exp2_deep_mlp") -> keras.Model:
    """Experiment 2 - Functional API: Dense -> BatchNorm -> ReLU -> Dropout blocks."""
    inp = keras.Input(shape=(n_features,), name="features")
    x = inp
    for i, u in enumerate(units):
        x = _dense_block(x, u, dropout, l2, f"block{i + 1}")
    out = layers.Dense(1, activation="sigmoid", name="readmit_prob")(x)
    return keras.Model(inp, out, name=name)


def build_embedding_model(cardinalities: dict, n_numeric: int, emb_size: int = 8,
                          units=(128, 64), dropout: float = 0.3, l2: float = 1e-4,
                          name: str = "exp3_embedding_model") -> keras.Model:
    """Experiment 3 - Functional API with one learned embedding per
    high-cardinality categorical, concatenated with the numeric/one-hot block."""
    inputs, parts = {}, []
    for col, card in cardinalities.items():
        inp = keras.Input(shape=(), dtype="int32", name=col)
        dim = int(min(emb_size, max(2, (card + 1) // 2)))
        emb = layers.Embedding(card, dim, name=f"emb_{col}")(inp)
        inputs[col] = inp
        parts.append(emb)
    num_in = keras.Input(shape=(n_numeric,), name="numeric")
    inputs["numeric"] = num_in
    x = layers.Concatenate(name="concat")(parts + [num_in])
    for i, u in enumerate(units):
        x = _dense_block(x, u, dropout, l2, f"block{i + 1}")
    out = layers.Dense(1, activation="sigmoid", name="readmit_prob")(x)
    return keras.Model(inputs, out, name=name)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(model, train_ds, val_ds, run_name: str, epochs: int = 40,
          class_weight=None, patience: int = 6, verbose: int = 0):
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weight, callbacks=make_callbacks(run_name, patience),
                        shuffle=False, verbose=verbose)  # shuffling is done in tf.data
    hist = pd.DataFrame(history.history)
    hist.index.name = "epoch"
    best = hist["val_pr_auc"].idxmax()
    print(f"{run_name}: {len(hist)} epochs, best val PR-AUC={hist['val_pr_auc'].max():.4f} "
          f"(epoch {best + 1}), {time.time() - t0:.0f}s")
    return hist


def predict_proba(model, X, batch_size: int = 2048) -> np.ndarray:
    return model.predict(make_dataset(X, batch_size=batch_size), verbose=0).ravel()


# --------------------------------------------------------------------------
# KerasTuner
# --------------------------------------------------------------------------
class EmbeddingHyperModel(kt.HyperModel):
    """Tunes learning rate, batch size, depth, width, dropout, L2 and
    embedding size of the embedding model. Batch size is a fit-time
    hyperparameter, so ``fit`` is overridden to build the tf.data pipeline."""

    def __init__(self, cardinalities, n_numeric, class_weight=None, seed=config.SEED):
        super().__init__()
        self.cardinalities, self.n_numeric = cardinalities, n_numeric
        self.class_weight, self.seed = class_weight, seed

    def build(self, hp):
        n_layers = hp.Int("n_layers", 1, 3)
        first = hp.Choice("units", [64, 128, 256])
        units = tuple(max(16, first // (2 ** i)) for i in range(n_layers))
        model = build_embedding_model(
            self.cardinalities, self.n_numeric,
            emb_size=hp.Choice("emb_size", [4, 8, 16]), units=units,
            dropout=hp.Float("dropout", 0.0, 0.5, step=0.1),
            l2=hp.Choice("l2", [0.0, 1e-5, 1e-4, 1e-3]), name="tuned_embedding_model")
        return compile_model(model, hp.Float("learning_rate", 1e-4, 3e-3, sampling="log"))

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        bs = hp.Choice("batch_size", [256, 512, 1024])
        train_ds = make_dataset(x, y, batch_size=bs, shuffle=True, seed=self.seed)
        val_ds = make_dataset(*validation_data, batch_size=2048)
        return model.fit(train_ds, validation_data=val_ds, class_weight=self.class_weight,
                         shuffle=False, **kwargs)


def hp_summary(tuner, n: int = 10) -> pd.DataFrame:
    rows = []
    for t in tuner.oracle.get_best_trials(n):
        rows.append({"trial_id": t.trial_id, "val_pr_auc": t.score, **t.hyperparameters.values})
    return pd.DataFrame(rows)
