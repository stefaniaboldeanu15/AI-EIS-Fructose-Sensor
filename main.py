import argparse
from pathlib import Path
import pandas as pd

from src.data_utils import read_dataset, concentration_to_mM, create_sensor_group, ensure_dir, save_dataframe
from src.model_1_eis_concentration import run_model_1
from src.model_2_fabrication_behavior import run_model_2
from src.model_3_eis_sensor_discrimination import run_model_3
from src.lod_analysis import run_lod_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Three-model ML workflow for EIS fructose dataset")
    parser.add_argument("--input", required=True, help="Path to master_dataset.xlsx")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index. Default: first sheet")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--lod_signal", default="Rct", help="Signal used for LOD calculation. Default: Rct")
    parser.add_argument("--include_q_cqds", action="store_true", help="Include Q_CQDs (ml) in fabrication model. Use only if it varies meaningfully.")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    print("Loading dataset...")
    df = read_dataset(args.input, sheet=args.sheet)
    df = concentration_to_mM(df)
    df = create_sensor_group(df)

    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    save_dataframe(df, outdir / "processed_dataset.xlsx")

    summary = []

    print("Running Model 1: EIS features -> concentration...")
    try:
        m1 = run_model_1(df, outdir / "model_1_eis_concentration", target_col="concentration_mM", random_state=args.random_state)
        summary.append(m1)
        print("Model 1 complete.")
    except Exception as e:
        print(f"Model 1 failed: {e}")
        summary.append({"model": "Model 1", "status": "failed", "error": str(e)})

    print("Running Model 2: fabrication parameters -> electrochemical parameters...")
    try:
        m2 = run_model_2(df, outdir / "model_2_fabrication_behavior", include_q_cqds=args.include_q_cqds, random_state=args.random_state)
        summary.append({"model": "Model 2", "status": "complete", "targets": ", ".join(m2.get("target", pd.Series()).astype(str).tolist())})
        print("Model 2 complete.")
    except Exception as e:
        print(f"Model 2 failed: {e}")
        summary.append({"model": "Model 2", "status": "failed", "error": str(e)})

    print("Running Model 3: EIS features -> sensor group...")
    try:
        m3 = run_model_3(df, outdir / "model_3_eis_sensor_discrimination", target_col="sensor_group", random_state=args.random_state)
        summary.append(m3)
        print("Model 3 complete.")
    except Exception as e:
        print(f"Model 3 failed: {e}")
        summary.append({"model": "Model 3", "status": "failed", "error": str(e)})

    print("Running LOD analysis...")
    try:
        lod = run_lod_analysis(df, outdir / "lod_analysis", signal_col=args.lod_signal, group_col="sensor_group", concentration_col="concentration_mM")
        summary.append({"model": "LOD analysis", "status": "complete", "signal": args.lod_signal, "n_sensor_groups": int(lod.shape[0])})
        print("LOD analysis complete.")
    except Exception as e:
        print(f"LOD analysis failed: {e}")
        summary.append({"model": "LOD analysis", "status": "failed", "error": str(e)})

    pd.DataFrame(summary).to_excel(outdir / "workflow_summary.xlsx", index=False)
    print("All done. Results saved in:", outdir.resolve())


if __name__ == "__main__":
    main()
