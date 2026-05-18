from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .data_utils import ensure_dir


def run_lod_analysis(df, outdir, signal_col="Rct", group_col="sensor_group", concentration_col="concentration_mM"):
    outdir = Path(outdir)
    ensure_dir(outdir)
    rows = []

    if signal_col not in df.columns:
        raise ValueError(f"LOD signal column {signal_col} not found.")

    for group, g in df.groupby(group_col):
        conc = pd.to_numeric(g[concentration_col], errors="coerce")
        signal = pd.to_numeric(g[signal_col], errors="coerce")
        valid = conc.notna() & signal.notna()
        gg = g.loc[valid].copy()
        conc = conc.loc[valid]
        signal = signal.loc[valid]

        blank_signal = signal[conc == 0]
        nonblank = conc > 0
        n_blank = int(blank_signal.shape[0])
        n_nonblank = int(nonblank.sum())
        n_total = int(valid.sum())

        status = "ok"
        slope = np.nan
        intercept = np.nan
        r2 = np.nan
        sigma_blank = np.nan
        lod = np.nan

        if n_blank < 3:
            status = "warning_less_than_3_blank_replicates"
        if n_nonblank < 3:
            status = "warning_less_than_3_nonblank_points"

        if n_blank >= 2 and n_nonblank >= 2:
            X = conc.values.reshape(-1, 1)
            y = signal.values
            model = LinearRegression().fit(X, y)
            pred = model.predict(X)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
            r2 = float(r2_score(y, pred))
            sigma_blank = float(np.std(blank_signal.values, ddof=1)) if n_blank >= 2 else np.nan
            if slope != 0 and not np.isnan(sigma_blank):
                lod = float(3.3 * sigma_blank / abs(slope))

        rows.append({
            "sensor_group": group,
            "signal_used": signal_col,
            "n_total": n_total,
            "n_blank": n_blank,
            "n_nonblank": n_nonblank,
            "slope": slope,
            "intercept": intercept,
            "r2_calibration": r2,
            "sigma_blank": sigma_blank,
            "LOD_mM": lod,
            "LOD_uM": lod * 1000 if pd.notna(lod) else np.nan,
            "status": status,
        })

    result = pd.DataFrame(rows).sort_values(["status", "LOD_mM"], na_position="last")
    result.to_excel(outdir / f"lod_by_sensor_group_using_{signal_col}.xlsx", index=False)
    return result
