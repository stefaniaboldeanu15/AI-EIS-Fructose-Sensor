import re
from pathlib import Path
import numpy as np
import pandas as pd

EIS_RE = re.compile(r"^(Zreal|Zimag|Zmod|phase)_([0-9]+(?:\.[0-9]+)?)Hz$")

FITTED_FEATURES = ["Rs", "Rct", "CPE_T", "CPE_P", "W"]

FABRICATION_FEATURES_DEFAULT = [
    "polypyrrole", "LiClO4", "TS", "CV", "CV_cycles", "Chrono", "Chrono_s",
    "CQDs", "Nafion", "Chitosan"
]

OPTIONAL_FABRICATION_FEATURES = ["Q_CQDs (ml)"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_dataset(path, sheet=0):
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def concentration_to_mM(df):
    """Create concentration_mM from concentration + concentration_unit.
    If unit is missing, assumes values are already in mM.
    """
    df = df.copy()
    if "concentration" not in df.columns:
        raise ValueError("Dataset must contain a 'concentration' column.")
    if "concentration_unit" not in df.columns:
        df["concentration_mM"] = pd.to_numeric(df["concentration"], errors="coerce")
        return df

    unit = df["concentration_unit"].astype(str).str.lower().str.replace("µ", "u", regex=False).str.strip()
    conc = pd.to_numeric(df["concentration"], errors="coerce")

    df["concentration_mM"] = conc
    df.loc[unit.isin(["um", "µm", "micromolar", "micromol/l", "umol/l"]), "concentration_mM"] = conc / 1000.0
    df.loc[unit.isin(["mm", "millimolar", "mmol/l"]), "concentration_mM"] = conc
    return df


def get_eis_columns(df):
    cols = []
    for c in df.columns:
        if EIS_RE.match(c):
            cols.append(c)
    # Sort by frequency descending, then by metric order
    order = {"Zreal": 0, "Zimag": 1, "Zmod": 2, "phase": 3}
    def key(c):
        m = EIS_RE.match(c)
        metric, freq = m.group(1), float(m.group(2))
        return (-freq, order[metric])
    return sorted(cols, key=key)


def get_fitted_columns(df):
    return [c for c in FITTED_FEATURES if c in df.columns]


def get_eis_derived_columns(df):
    return get_eis_columns(df) + get_fitted_columns(df)


def get_fabrication_columns(df, include_q_cqds=False):
    cols = [c for c in FABRICATION_FEATURES_DEFAULT if c in df.columns]
    if include_q_cqds:
        cols += [c for c in OPTIONAL_FABRICATION_FEATURES if c in df.columns]
    return cols


def create_sensor_group(df):
    """Create a readable sensor group label from fabrication columns."""
    df = df.copy()
    labels = []
    for _, row in df.iterrows():
        parts = []
        if row.get("base_electrode", "") not in [np.nan, None, ""]:
            parts.append(str(row.get("base_electrode")))
        else:
            parts.append("electrode")
        if row.get("polypyrrole", 0) == 1:
            parts.append("PPy")
        if row.get("LiClO4", 0) == 1:
            parts.append("LiClO4")
        if row.get("TS", 0) == 1:
            parts.append("pTS")
        if row.get("CV", 0) == 1:
            cycles = row.get("CV_cycles", "")
            parts.append(f"CV{int(cycles) if pd.notna(cycles) else ''}cy")
        if row.get("Chrono", 0) == 1:
            chrono_s = row.get("Chrono_s", "")
            parts.append(f"Chrono{int(chrono_s) if pd.notna(chrono_s) else ''}s")
        if row.get("CQDs", 0) == 1:
            parts.append("CQDs")
        if row.get("Nafion", 0) == 1:
            parts.append("Nafion")
        if row.get("Chitosan", 0) == 1:
            parts.append("Chitosan")
        labels.append("/".join(parts))
    df["sensor_group"] = labels
    return df


def numeric_features(df, columns):
    X = df[columns].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def drop_bad_feature_columns(X, max_missing_fraction=0.45):
    missing_frac = X.isna().mean()
    keep = missing_frac[missing_frac <= max_missing_fraction].index.tolist()
    return X[keep], missing_frac.sort_values(ascending=False)


def save_dataframe(df, path):
    ensure_dir(Path(path).parent)
    df.to_excel(path, index=False)
