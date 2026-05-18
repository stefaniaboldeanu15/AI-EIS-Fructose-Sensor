from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .data_utils import get_eis_derived_columns, numeric_features, drop_bad_feature_columns, ensure_dir
from .plot_utils import save_bar_importance, confusion_matrix_plot, pca_scatter


def run_model_3(df, outdir, target_col="sensor_group", random_state=42):
    outdir = Path(outdir)
    ensure_dir(outdir)

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found.")

    feature_cols = get_eis_derived_columns(df)
    X = numeric_features(df, feature_cols)
    X, missing = drop_bad_feature_columns(X)
    y = df[target_col].astype(str)

    # Keep classes with at least 2 samples so stratified split can work
    counts = y.value_counts()
    keep_classes = counts[counts >= 2].index
    mask = y.isin(keep_classes)
    X = X.loc[mask]
    y = y.loc[mask]

    if y.nunique() < 2:
        raise ValueError("Need at least two sensor groups with >=2 samples each.")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, class_weight="balanced", min_samples_leaf=1))
    ])

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=stratify)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    confusion_matrix_plot(cm, labels, outdir / "confusion_matrix_sensor_group.png", "Model 3: EIS fingerprinting of sensor groups")

    report_dict = classification_report(y_test, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_excel(outdir / "classification_report.xlsx")

    metrics = {
        "model": "Model 3 - EIS features to sensor group",
        "target": target_col,
        "n_samples": int(len(X)),
        "n_classes": int(y.nunique()),
        "n_features_used": int(X.shape[1]),
        "accuracy_test": float(accuracy_score(y_test, pred)),
        "balanced_accuracy_test": float(balanced_accuracy_score(y_test, pred)),
    }

    pd.DataFrame([metrics]).to_excel(outdir / "metrics.xlsx", index=False)

    model = pipeline.named_steps["model"]
    imp = save_bar_importance(model.feature_importances_, X.columns, outdir / "feature_importance_top30_sensor_group.png", "Model 3 feature importance", top_n=30)
    imp.to_excel(outdir / "feature_importance.xlsx", index=False)

    # PCA by sensor group
    pca_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=random_state))
    ])
    pcs = pca_pipeline.fit_transform(X)
    pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], target_col: y.values})
    pca_scatter(pca_df, target_col, outdir / "pca_by_sensor_group.png", "PCA of EIS features by sensor group")
    missing.to_excel(outdir / "missing_fraction_all_eis_features.xlsx")
    return metrics
