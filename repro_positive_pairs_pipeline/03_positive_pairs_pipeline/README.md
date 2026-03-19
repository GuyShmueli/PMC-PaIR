# Reproducible positive medical image-pair pipeline

This refactor turns the notebook into a staged, reproducible pipeline. It removes the notebook-only branches that were specific to:
- running non-compound first and compound later,
- merging the new compound outputs into an older radiology baseline,
- recovering malformed positive-pair outputs after the fact because the notebook prompt mixed “multiple caption pairs” with a “single JSON object” schema.

The scripts below assume you run **from scratch on the full dataset** (compound and non-compound together).

## Directory layout

```text
repro_positive_pairs_pipeline/
├── positive_pairs_pipeline/
│   ├── artifacts.py
│   ├── batch_api.py
│   ├── case_summaries.py
│   ├── citations.py
│   ├── data_prep.py
│   ├── io_utils.py
│   ├── positive_parser.py
│   └── prompts.py
├── scripts/
│   ├── 01_prepare_candidates.py
│   ├── 02_extract_case_summaries.py
│   ├── 03_extract_citations.py
│   ├── 04_make_batch_requests.py
│   ├── 05_submit_batch.py
│   ├── 06_download_batch.py
│   ├── 07_extract_batch_text_responses.py
│   └── 08_finalize_positive_pairs.py
└── requirements.txt
```

## Inputs expected

Stage 01 needs:
- `final_csv2.csv`
- `imgs_paths_diagrams.json`
- the `data2/` directory

The code assumes the same basic structure as the notebook:
- `patient_uid`
- `unique_articles_sim_patients`
- `caption_path`
- `image_path`

## What changed relative to the notebook

1. **No `compound_*` / `new_*` / merge path**
   The notebook had an old-non-compound branch and then a later compound branch. That is gone here. Everything runs once over the full filtered dataset.

2. **Batch response alignment is now deterministic**
   The notebook implicitly trusted raw batch response order. This refactor aligns responses by `custom_id`, so later stages stay synchronized even if the output JSONL order changes.

3. **The final positive-pair prompt now asks for a JSON array**
   The notebook passed multiple caption candidates but asked for one JSON object, which is why later cells needed robust recovery and merging logic. Here the request is explicit: return a JSON array of positive caption pairs, or `[]`.

4. **Case-summary extraction now keeps pair index alignment**
   The notebook sometimes dropped or re-indexed artifacts implicitly. Here every major artifact keeps pair-level alignment, and later request-index files are saved explicitly.

5. **No hard-coded API keys, file IDs, or batch IDs**
   The batch helper scripts expect `OPENAI_API_KEY` in the environment.

## Stage-by-stage usage

All scripts write into one shared `--output-dir`.

### 1) Build candidate case pairs and aligned artifacts

```bash
python scripts/01_prepare_candidates.py \
  --dataset-csv /path/to/final_csv2.csv \
  --diagram-paths-json /path/to/imgs_paths_diagrams.json \
  --data2-base /path/to/data2 \
  --output-dir /path/to/run_outputs
```

Outputs include:
- filtered rows
- candidate patient-UID pairs
- aligned text/image path groups
- caption dictionaries per pair
- XML path pairs

### 2) Extract case summaries

```bash
python scripts/02_extract_case_summaries.py \
  --output-dir /path/to/run_outputs \
  --workers 8
```

### 3) Extract citation-linked case pairs

```bash
python scripts/03_extract_citations.py \
  --output-dir /path/to/run_outputs \
  --restrict-to-summary-valid \
  --title-fallback
```

`--restrict-to-summary-valid` reproduces the notebook’s effective behavior more closely.
`--title-fallback` reproduces the title-substring fallback in the notebook.

### 4) Build request files

Citation reasoning:
```bash
python scripts/04_make_batch_requests.py citation --output-dir /path/to/run_outputs
```

Comparative questions:
```bash
python scripts/04_make_batch_requests.py question --output-dir /path/to/run_outputs
```

Comparative answers:
```bash
python scripts/04_make_batch_requests.py answer --output-dir /path/to/run_outputs
```

Final positive caption-pair classification:
```bash
python scripts/04_make_batch_requests.py positive --output-dir /path/to/run_outputs
```

### 5) Submit a batch

Example for the citation stage:
```bash
export OPENAI_API_KEY=...
python scripts/05_submit_batch.py --output-dir /path/to/run_outputs --stage citation
```

Repeat with `--stage question`, `answer`, and `positive`.

### 6) Download completed batch files

Example:
```bash
python scripts/06_download_batch.py --output-dir /path/to/run_outputs --stage citation
```

### 7) Extract ordered assistant texts from the raw batch JSONL

Example:
```bash
python scripts/07_extract_batch_text_responses.py --output-dir /path/to/run_outputs --stage citation
```

Repeat for `question`, `answer`, and `positive`.

### 8) Finalize cleaned positive pairs

```bash
python scripts/08_finalize_positive_pairs.py --output-dir /path/to/run_outputs
```

Outputs include:
- all labeled positive pairs,
- cleaned pair IDs,
- aligned short text descriptions,
- modality buckets,
- radiology-only outputs,
- bad-row diagnostics.

## Default artifact names

The scripts write standardized file names such as:
- `01_pair_patient_uids.json`
- `02_case_summary_pairs.json`
- `03_citation_records.json`
- `batch/04_citation_reasoning_requests.jsonl`
- `batch/05_question_texts.json`
- `batch/06_answer_texts.json`
- `batch/07_positive_texts.json`
- `08_radiology_pairs_clean.json`

## Notes

- The positive-pair finalizer validates that each predicted `pair_id` uses caption IDs that actually belong to the corresponding request’s Case A / Case B dictionaries.
- By default the final bucketing uses the stricter modality order (ophthalmology / pathology / endoscopy / cardiology checked before generic radiology keywords). Use `--legacy-radiology-bucket` in stage 08 if you need notebook-like bucket behavior.
- If a batch stage has missing responses, stage 07 preserves the missing positions as `null` so the next stage can fail loudly instead of silently drifting out of alignment.
