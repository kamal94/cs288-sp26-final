"""
run_all_plots.py

Runs plot_answer_change for every CSV in the outputs/ directory.
Output PNGs are saved to plots/ at the project root.

Usage:
    python eval/run_all_plots.py
    python eval/run_all_plots.py --outputs-dir path/to/outputs --plots-dir path/to/plots
"""

import argparse
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

from eval.plot_accuracy_trend import _infer_output_path, load_records, compute_change_percentages, plot


def run_all(outputs_dir: Path, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(outputs_dir.glob("truthfulqa_*.csv"))
    if not csvs:
        print(f"No CSV files found in {outputs_dir}")
        return

    for csv_path in csvs:
        # Reuse the naming logic but redirect to plots_dir
        default_name = Path(_infer_output_path(csv_path)).name
        out_path = plots_dir / default_name

        print(f"\n--- {csv_path.name} -> {out_path} ---")
        try:
            records = load_records(csv_path)
            print(f"Loaded {len(records)} records")
            turns, correct_pcts, incorrect_pcts = compute_change_percentages(records)

            print("Turn | Correct-init change% | Incorrect-init change%")
            for t, c, i in zip(turns, correct_pcts, incorrect_pcts):
                print(f"  {t:2d} |        {c:6.2f}%        |         {i:6.2f}%")

            plot(turns, correct_pcts, incorrect_pcts, str(out_path))
        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default=str(root / "outputs"))
    parser.add_argument("--plots-dir",   default=str(root / "plots"))
    args = parser.parse_args()
    run_all(Path(args.outputs_dir), Path(args.plots_dir))


if __name__ == "__main__":
    main()
