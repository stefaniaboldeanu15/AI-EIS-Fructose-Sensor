from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_utils import get_fabrication_columns, numeric_features, ensure_dir
from .plot_utils import save_bar_importance, predicted_vs_actual, residual_plot


def run_model_2(df, outdir, include_q_cqds=False, random_state=42):
    outdir = Path(outdir)
    ensure_dir(outdir)

    feature_cols = get_fabrication_columns(df, include_q_cqds=include_q_cqds)
    targets = [c for c in ["Rs", "Rct", "CPE_T", "CPE_P", "W"] if c in df.columns]

    X_all = numeric_features(df, feature_cols)
    results = []

    for target in targets:
        target_dir = outdir / f"target_{target}"
        ensure_dir(target_dir)
        y = pd.to_numeric(df[target], errors="coerce")
        valid = y.notna()
        X = X_all.loc[valid]
        yy = y.loc[valid]

        if len(X) < 20:
            results.append({"target": target, "status": "skipped_too_few_samples", "n_samples": int(len(X))})
            continue

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, min_samples_leaf=2))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.25, random_state=random_state)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        metrics = {
            "model": "Model 2 - Fabrication parameters to electrochemical behavior",
            "target": target,
            "n_samples": int(len(X)),
            "n_features_used": int(X.shape[1]),
            "r2_test": float(r2_score(y_test, pred)),
            "rmse_test": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae_test": float(mean_absolute_error(y_test, pred)),
        }
        pd.DataFrame({"actual": y_test, "predicted": pred, "residual": y_test - pred}).to_excel(target_dir / "predictions.xlsx", index=False)
        predicted_vs_actual(y_test, pred, target_dir / "predicted_vs_actual.png", f"Fabrication → {target}")
        residual_plot(y_test, pred, target_dir / "residuals.png", f"Fabrication → {target} residuals")

        model = pipeline.named_steps["model"]
        imp = save_bar_importance(model.feature_importances_, X.columns, target_dir / "fabrication_feature_importance.png", f"Fabrication features for {target}", top_n=20)
        imp.to_excel(target_dir / "feature_importance.xlsx", index=False)
        pd.DataFrame([metrics]).to_excel(target_dir / "metrics.xlsx", index=False)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_excel(outdir / "model_2_summary_metrics.xlsx", index=False)
    return results_df
