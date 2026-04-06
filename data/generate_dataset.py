"""Generate a randomized MCQ dataset from TruthfulQA.

Output format is compatible with eval/static_eval.py:
- question
- correct answer
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


INPUT_COLUMNS = ("Question", "Correct Answers", "Incorrect Answers")
OUTPUT_COLUMNS = ("question", "correct answer")
CHOICE_LABELS = ["A", "B", "C", "D", "E"]
logger = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_semicolon_field(raw: str) -> List[str]:
    """Split a semicolon-delimited field and normalize whitespace."""
    items = []
    for part in (raw or "").split(";"):
        cleaned = part.strip()
        if cleaned:
            items.append(cleaned)
    return items


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = value.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(value.strip())
    return out


def mcq_text(question: str, options: List[str]) -> str:
    lines = [question.strip()]
    for label, option in zip(CHOICE_LABELS, options):
        lines.append(f"{label}) {option}")
    return "\n".join(lines)


def build_dataset_rows(
    input_csv: Path,
    max_questions: Optional[int],
    seed: int,
    log_every: int,
) -> List[Dict[str, str]]:
    started = time.perf_counter()
    rng = random.Random(seed)
    logger.info("Reading source dataset from %s", input_csv)

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing a header row.")
        missing = [col for col in INPUT_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {missing}. "
                f"Expected: {list(INPUT_COLUMNS)}"
            )
        source_rows = list(reader)
    logger.info("Loaded %s source rows", len(source_rows))

    rng.shuffle(source_rows)
    output_rows: List[Dict[str, str]] = []
    skipped_rows = 0

    for idx, row in enumerate(source_rows, start=1):
        question = (row.get("Question") or "").strip()
        if not question:
            skipped_rows += 1
            continue

        correct_candidates = dedupe_keep_order(parse_semicolon_field(row.get("Correct Answers", "")))
        incorrect_candidates = dedupe_keep_order(
            parse_semicolon_field(row.get("Incorrect Answers", ""))
        )
        if not correct_candidates or len(incorrect_candidates) < 4:
            skipped_rows += 1
            continue

        correct_choice = rng.choice(correct_candidates)
        incorrect_choices = rng.sample(incorrect_candidates, 4)

        all_choices = [correct_choice] + incorrect_choices
        rng.shuffle(all_choices)

        correct_letter = CHOICE_LABELS[all_choices.index(correct_choice)]
        prompt = mcq_text(question, all_choices)
        output_rows.append({"question": prompt, "correct answer": correct_letter})
        if len(output_rows) <= 3:
            logger.debug(
                "Sample generated row %s: correct_letter=%s question_prefix=%r",
                len(output_rows),
                correct_letter,
                question[:80],
            )
        if log_every > 0 and len(output_rows) % log_every == 0:
            logger.info(
                "Generated %s rows so far (processed %s source rows)",
                len(output_rows),
                idx,
            )

        if max_questions is not None and len(output_rows) >= max_questions:
            break

    if not output_rows:
        raise ValueError("No valid rows were generated. Check input dataset quality.")
    logger.info(
        "Finished generation: %s rows (skipped=%s) in %.2fs",
        len(output_rows),
        skipped_rows,
        time.perf_counter() - started,
    )
    return output_rows


def write_dataset_csv(output_csv: Path, rows: List[Dict[str, str]]) -> None:
    started = time.perf_counter()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "Wrote dataset CSV to %s (%s rows) in %.2fs",
        output_csv,
        len(rows),
        time.perf_counter() - started,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("data/TruthfulQA.csv"),
        help="Path to TruthfulQA source CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("data/dataset.csv"),
        help="Path for generated dataset CSV.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=250,
        help="Maximum number of rows to generate (default: 250).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling/shuffling.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for dataset generation.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Progress log interval in generated rows (default: 50).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    max_questions = None if args.max_questions <= 0 else args.max_questions

    rows = build_dataset_rows(
        input_csv=args.input_csv,
        max_questions=max_questions,
        seed=args.seed,
        log_every=args.log_every,
    )
    write_dataset_csv(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

