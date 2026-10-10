"""Reusable code for the diabetic 30-day readmission project.

Modules
-------
config      paths, seeds, column lists and run-time settings
data        downloading / loading the UCI files and cleaning the raw table
features    row-wise feature engineering and the rare-category transformer
pipelines   scikit-learn / imbalanced-learn preprocessing pipelines
ml_models   traditional model zoo, search spaces and tuning helpers
dl_models   tf.data pipelines, Keras architectures and training helpers
evaluation  thresholds, metrics, bootstrap CIs and subgroup error analysis
plotting    figure helpers that save 300-dpi PNGs to figures/
"""
