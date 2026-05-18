from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .data_utils import get_eis_derived_columns, numeric_features, drop_bad_feature_columns, ensure_dir
from .plot_utils import save_bar_importance, predicted_vs_actual, residual_plot, pca_scatter


def run_model_1(df, outdir, target_col="concentration_mM", random_state=42):
    outdir = Path(outdir)
    ensure_dir(outdir)

    feature_cols = get_eis_derived_columns(df)
    X = numeric_features(df, feature_cols)
    X, missing = drop_bad_feature_columns(X)
    y = pd.to_numeric(df[target_col], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < 20:
        raise ValueError("Model 1 needs at least ~20 valid samples.")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, min_samples_leaf=2))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "model": "Model 1 - EIS features to concentration",
        "target": target_col,
        "n_samples": int(len(X)),
        "n_features_used": int(X.shape[1]),
        "r2_test": float(r2_score(y_test, pred)),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae_test": float(mean_absolute_error(y_test, pred)),
    }

    pd.DataFrame({"actual": y_test, "predicted": pred, "residual": y_test - pred}).to_excel(outdir / "predictions.xlsx", index=False)
    predicted_vs_actual(y_test, pred, outdir / "predicted_vs_actual.png", "Model 1: EIS-based fructose concentration prediction")
    residual_plot(y_test, pred, outdir / "residuals.png", "Model 1 residuals")

    model = pipeline.named_steps["model"]
    imp = save_bar_importance(model.feature_importances_, X.columns, outdir / "feature_importance_top30.png", "Model 1 feature importance", top_n=30)
    imp.to_excel(outdir / "feature_importance.xlsx", index=False)

    # PCA visualization
    pca_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=random_state))
    ])
    pcs = pca_pipeline.fit_transform(X)
    pca_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "concentration_mM": y.values})
    pca_scatter(pca_df, "concentration_mM", outdir / "pca_by_concentration.png", "PCA of EIS features by concentration")

    pd.Series(metrics).to_json(outdir / "metrics.json", indent=2)
    pd.DataFrame([metrics]).to_excel(outdir / "metrics.xlsx", index=False)
    missing.to_excel(outdir / "missing_fraction_all_eis_features.xlsx")
    return metrics
