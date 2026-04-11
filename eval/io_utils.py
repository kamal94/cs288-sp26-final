"""I/O helpers for static evaluation."""

from __future__ import annotations

import csv
import json
import random
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_INPUT_COLUMNS = ("question", "correct answer")
UNPARSEABLE = "UNPARSEABLE"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_uuid(length: int = 8) -> str:
    return uuid.uuid4().hex[:length]


def normalize_answer_letter(value: str) -> str:
    text = (value or "").strip().upper()
    if not text:
        raise ValueError("Empty answer letter.")
    letter = text[0]
    if not ("A" <= letter <= "Z"):
        raise ValueError(f"Invalid answer letter: {value!r}")
    return letter


def load_input_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Input CSV missing required columns: {missing}. "
                f"Expected at least {list(REQUIRED_INPUT_COLUMNS)}."
            )

        rows: List[Dict[str, str]] = []
        for idx, row in enumerate(reader, start=2):
            question = (row.get("question") or "").strip()
            correct = normalize_answer_letter(row.get("correct answer") or "")
            if not question:
                raise ValueError(f"Row {idx}: 'question' is empty.")
            rows.append({"question": question, "correct_answer": correct})

    if not rows:
        raise ValueError("Input CSV contains no data rows.")
    return rows


def parse_answer_letter(text: str) -> str:
    """Try to parse a final answer letter from model text."""
    if not text:
        return UNPARSEABLE

    match = re.search(r"answer\s*[:\-]\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        return match.group(1).upper()
    return UNPARSEABLE


def infer_option_letters_from_question(question: str) -> List[str]:
    """Extract option labels (A, B, C...) from question text."""
    candidates = set()
    for pattern in [
        r"\(([A-Z])\)",
        r"\b([A-Z])\s*[\)\.\-:]",  # A) / A. / A- / A:
    ]:
        for match in re.finditer(pattern, question):
            candidates.add(match.group(1))
    return sorted(candidates)


def pick_probe_answer(last_answer: str, question: str) -> str:
    options = infer_option_letters_from_question(question)
    others = [o for o in options if o != last_answer]
    if others:
        return random.choice(others)
    for fallback in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if fallback != last_answer:
            return fallback
    return "A"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def write_rows_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

