# Static 7-Turn Evaluation

This project runs a static multi-turn sycophancy evaluation loop inspired by TRUTH DECAY.

## What it does

- Reads an input CSV with columns:
  - `question`
  - `correct answer`
- Runs exactly 7 turns for each question and each static type:
  - `Feedback`
  - `AreYouSure`
  - `Answer`
  - `Mimicry`
- Writes:
  - one aggregate output CSV
  - one JSON request/response log per conversation ID

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`
- Qwen (OpenAI-compatible): `QWEN_BASE_URL`, optional `QWEN_API_KEY`
- Llama (OpenAI-compatible): `LLAMA_BASE_URL`, optional `LLAMA_API_KEY`

## Usage

```bash
python -m eval.static_eval \
  --input_csv data/questions.csv \
  --output_csv outputs/static_eval.csv \
  --logs_dir outputs/logs \
  --provider openai \
  --model gpt-4o-mini \
  --num_turns 7 \
  --log_level INFO
```

## Build dataset from TruthfulQA

Generate a randomized MCQ dataset with 5 options per question
(1 sampled correct + 4 sampled incorrect), outputting columns
`question` and `correct answer`:

```bash
python data/generate_dataset.py \
  --input_csv data/TruthfulQA.csv \
  --output_csv data/questions.csv \
  --max_questions 250 \
  --seed 42 \
  --log_level INFO \
  --log_every 50
```

Then run evaluation against `data/questions.csv`.

### Provider options

- `--provider openai`
- `--provider anthropic`
- `--provider google`
- `--provider qwen` (requires `QWEN_BASE_URL` or `--qwen_base_url`)
- `--provider llama` (requires `LLAMA_BASE_URL` or `--llama_base_url`)

## Dry run (no model API calls)

Useful to validate schema and output wiring quickly:

```bash
python -m eval.static_eval \
  --input_csv data/questions.csv \
  --output_csv outputs/static_eval.csv \
  --logs_dir outputs/logs \
  --provider openai \
  --model gpt-4o-mini \
  --num_turns 7 \
  --dry_run \
  --log_level DEBUG
```

## Logging controls

Both scripts support runtime-adjustable logging:

- `--log_level` one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `data/generate_dataset.py` also supports `--log_every` for periodic progress logs

## Output format

Aggregate CSV columns:

- `conversation_id`
- `question`
- `correct_answer`
- `static_type`
- `turn_1_answer` ... `turn_7_answer`

Conversation logs:

- Stored in `--logs_dir`
- File name: `<conversation_id>.json`
- Includes full request payloads, full raw model responses, parsed turn answers, and metadata.

