# EIS Fructose ML Package — Three Scientific Models

This package uses one master Excel dataset and creates three separated ML analyses:

## Model 1 — EIS-based fructose concentration prediction
**Question:** Can the measured EIS response predict fructose concentration?

- Input features: `Zreal_*`, `Zimag_*`, `Zmod_*`, `phase_*`, `Rs`, `Rct`, `CPE_T`, `CPE_P`, `W`
- Target: `concentration_mM`
- Results: R², RMSE, MAE, predicted-vs-actual plot, residuals, feature importance

## Model 2 — Fabrication parameters → electrochemical behavior
**Question:** How do fabrication parameters affect electrochemical behavior?

- Input features: `polypyrrole`, `LiClO4`, `TS`, `CV`, `CV_cycles`, `Chrono`, `Chrono_s`, `CQDs`, `Nafion`, `Chitosan`
- Excluded by default: `Q_CQDs (ml)` because it is redundant if always 3 mL when CQDs = 1
- Targets: `Rs`, `Rct`, `CPE_T`, `CPE_P`, `W`
- Results: performance metrics and fabrication feature importance for each electrochemical parameter

## Model 3 — EIS fingerprinting of sensor/coating architecture
**Question:** Can EIS distinguish sensor architectures without giving the model the fabrication recipe?

- Input features: EIS-derived only: `Zreal_*`, `Zimag_*`, `Zmod_*`, `phase_*`, `Rs`, `Rct`, `CPE_T`, `CPE_P`, `W`
- Target: auto-generated `sensor_group`
- Results: accuracy, confusion matrix, PCA by sensor group, feature importance

## LOD analysis
The package also estimates LOD for each sensor group using:

`LOD = 3.3 × sigma_blank / slope`

The default analytical signal is `Rct`. You can change it with `--lod_signal`.

## Setup in VS Code / PowerShell

```powershell
cd "path\to\eis_three_model_package"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py --input "data\master_dataset.xlsx"
```

## Important interpretation note
The three models are intentionally separated to avoid data leakage:

- Do **not** use fabrication recipe columns when proving EIS can classify coatings.
- Do **not** use `sensor_group` metadata when claiming pure EIS-based concentration prediction, unless you explicitly describe the model as sensor-aware.
- Use fabrication features only when the scientific question is fabrication optimization.

## Output folders
All results are exported to:

```text
results/
├── model_1_eis_concentration/
├── model_2_fabrication_behavior/
├── model_3_eis_sensor_discrimination/
├── lod_analysis/
└── processed_dataset.xlsx
```
