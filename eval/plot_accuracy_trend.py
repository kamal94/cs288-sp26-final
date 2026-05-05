"""
plot_accuracy_trend.py

Generates "Average Accuracy per Turn" graph from an outputs file.
One line, one graph per file.

CSV format (columns): conversation_id, correct_answer, static_type,
  turn_1_answer, turn_2_probe, turn_2_answer, ...

Usage:
    python plot_accuracy_trend.py outputs.csv
    python plot_accuracy_trend.py outputs.csv --out my_graph.png
"""

import csv
import json
import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _infer_output_path(input_path):
    stem = Path(input_path).stem
    stem = re.sub(r'_\d{8}_\d{6}$', '', stem)
    m = re.search(r'static_eval_(.+)$', stem)
    model = m.group(1) if m else stem
    prefix = "truthfulqa_" if stem.lower().startswith("truthfulqa") else ""
    return str(Path(input_path).parent / f"{prefix}{model}_accuracy_trend.png")


def load_records(path):
    if Path(path).suffix.lower() == ".csv":
        return _load_records_csv(path)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("results", "data", "records", "output"):
        if key in data and isinstance(data[key], list):
            return data[key]
    raise ValueError(f"Cannot find a list of records in {path}.")


def _load_records_csv(path):
    records = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer_cols = sorted(
                [k for k in row if re.match(r'^turn_\d+_answer$', k)],
                key=lambda k: int(re.search(r'\d+', k).group()),
            )
            records.append({
                "answers": [row[k] for k in answer_cols],
                "correct_answer": row["correct_answer"],
            })
    return records


def get_answers(record):
    for key in ("answers", "responses", "turns", "model_answers"):
        if key in record and isinstance(record[key], list):
            return [str(a).strip().upper() for a in record[key]]
    raise KeyError(f"No answers list found. Keys: {list(record.keys())}")


def get_correct(record):
    for key in ("correct_answer", "label", "answer", "gold", "ground_truth"):
        if key in record:
            return str(record[key]).strip().upper()
    raise KeyError(f"No correct answer found. Keys: {list(record.keys())}")


def compute_accuracy(records):
    max_turns = max(len(get_answers(r)) for r in records)

    # Turn 0 = initial answer, turns 1..N = after each probe
    labels = ["Initial"] + [f"F{t}" for t in range(1, max_turns)]

    accuracies = []
    for t in range(max_turns):
        eligible = [r for r in records if len(get_answers(r)) > t]
        if not eligible:
            accuracies.append(0.0)
            continue
        correct = sum(
            1 for r in eligible
            if get_answers(r)[t] == get_correct(r)
        )
        accuracies.append(100.0 * correct / len(eligible))

    return labels, accuracies


def plot(labels, accuracies, out_path, title=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(labels, accuracies, color="steelblue", marker="o", linewidth=2)

    ax.set_title(title or "Accuracy Trends per Follow-Up", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Accuracy (%)", fontsize=12)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs_file")
    parser.add_argument("--out", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    out_path = args.out or _infer_output_path(args.outputs_file)
    records = load_records(args.outputs_file)
    print(f"Loaded {len(records)} records from {args.outputs_file}")

    labels, accuracies = compute_accuracy(records)

    print("\nStep | Accuracy%")
    for l, a in zip(labels, accuracies):
        print(f"  {l:>8} | {a:6.2f}%")

    plot(labels, accuracies, out_path, title=args.title)


if __name__ == "__main__":
    main()