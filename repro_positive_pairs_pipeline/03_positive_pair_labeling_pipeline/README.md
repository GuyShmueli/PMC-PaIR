# Pipeline 03 — Positive image-pair labeling and validation

This standalone pipeline consumes one completed
`02_case_pair_selection_pipeline` run. It explains the citation, generates
comparative questions, answers them from the two case descriptions, proposes
positive inter-case caption pairs, and validates/deduplicates the result.

Pipeline 03 has its own configuration, run directory, implementation
fingerprint, and `run_manifest.json`. It never appends to or modifies the
Pipeline 02 run.

## Boundary with Pipeline 02

Pipeline 02 publishes this immutable handoff:

| Artifact | Content |
|---|---|
| `03_eligible_cases.jsonl` | Selected human cases, with case descriptions and caption groups |
| `03_eligible_pairs.jsonl` | Stable pair identities and matched citation evidence |
| `03_handoff_manifest.json` | Producer identity, semantic contract, counts, byte sizes, and SHA-256 hashes |
| `02_species_classifications.jsonl` | One `human` or `animal` label for every screened case |
| `02_pair_species.jsonl` | One `human` or `animal` label for every constructed case pair |
| `02_species_stats.json` | Screen configuration and aggregate case/pair counts |

`initialize` requires a completed Pipeline 02 v6.1.0 manifest and handoff
schema v4. It verifies the handoff contract and all hashes, checks
case/pair/caption alignment, and requires a completed
`02_human_animal_screen` stage whose three canonical artifacts are hash-bound
by the Pipeline 02 manifest. The accepted ruleset is `human_animal_v1`. Every
constructed case pair must have exactly one binary label. A pair is `human`
only when both of its cases are labeled `human`; otherwise it is `animal`.
Only human pairs may appear in the eligible handoff. Missing, malformed, or
internally inconsistent records stop initialization. Optional adjudications
may replace an automatic label, but the final label must still be either
`human` or `animal`. Pipeline 03 snapshots all three species-screen artifacts
alongside the handoff.
Original, potentially non-contiguous
`pair_index` values and canonical case orientation are preserved.

Citation context uses the plural field `citation_paragraphs`, which is always
a JSON array and may contain paragraphs or header-qualified table rows.
Pipeline 03 rejects the singular field `citation_paragraph` as a schema
mismatch.

Pipeline 03 does not read Level-1 CSV, NXML, JPG, or TXT files. The immutable
handoff contains all text, membership, and population-screening data needed by
stages 04–08. Its derived schema-v1 compatibility views intentionally omit the
`population_scope` field; the original schema-v2 case records remain preserved
in `00_eligible_cases.jsonl` and bound by the run manifest.

## Environment and tests

Python 3.11 is the tested runtime.

```bash
conda env create -f environment.yml
conda activate repro-positive-pair-labeling

# or
python -m pip install -r requirements-dev.txt

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

No test contacts OpenAI or NCBI.

## 1. Initialize a labeling run

Copy and review the model/batch configuration:

```bash
cp config.example.toml labeling.toml
```

Then import one Pipeline 02 selection run into a separate, new output directory:

```bash
python scripts/run_pipeline.py initialize \
  --selection-run /path/to/case_pair_selection_run \
  --config labeling.toml \
  --output-dir /path/to/positive_pair_labeling_run
```

`import` is an alias for `initialize`. Initialization is atomic: a failed or
interrupted import does not publish a partial output directory. An empty
selection is valid and can be finalized immediately without API calls.

## 2. Run the four model stages

Stages are dependent and must be completed in order:

```text
04 citation explanation
   → 05 comparative questions
   → 06 case-grounded answers
   → 07 positive caption-pair proposals
   → 08 validation and deduplication
```

For offline or externally managed Batch processing:

```bash
python scripts/run_pipeline.py build \
  --output-dir /path/to/positive_pair_labeling_run --stage citation

python scripts/run_pipeline.py ingest \
  --output-dir /path/to/positive_pair_labeling_run --stage citation \
  --response-file /path/to/batch-output.jsonl
```

Repeat `build` and `ingest` for `question`, `answer`, and `positive`.
`--response-file` accepts a JSONL file or a directory of JSONL files and may be
repeated. Responses are reconciled by immutable `custom_id`, not line order. Missing,
duplicate, unexpected, errored, non-2xx, empty, or schema-invalid responses fail
closed.

For the managed OpenAI Batch lifecycle, build the stage first and then use:

```bash
python scripts/run_pipeline.py submit \
  --output-dir /path/to/positive_pair_labeling_run --stage citation \
  --acknowledge-external-api

python scripts/run_pipeline.py status \
  --output-dir /path/to/positive_pair_labeling_run --stage citation \
  --acknowledge-external-api

python scripts/run_pipeline.py download \
  --output-dir /path/to/positive_pair_labeling_run --stage citation \
  --acknowledge-external-api
```

Every network command requires explicit acknowledgement. Automatic SDK retries
are disabled so an ambiguous batch creation cannot trigger another submission.
Managed attempts,
submissions, downloads, validation failures, and retries are recorded as
immutable receipts. Use `retry --attempt N` only after the preceding downloaded
attempt failed validation, including batches containing only request errors;
use `resolve` only to bind a manually verified
batch after an ambiguous create call.

## 3. Finalize and verify

```bash
python scripts/run_pipeline.py finalize \
  --output-dir /path/to/positive_pair_labeling_run

python scripts/run_pipeline.py verify \
  --output-dir /path/to/positive_pair_labeling_run
```

The finalizer accepts JSON arrays, optionally wrapped in one Markdown code
fence. It strips surrounding whitespace from descriptive fields, verifies one
caption ID from each selected case, sorts the two caption IDs, and removes
duplicate caption pairs. Conflicting duplicate labels are resolved
deterministically by source pair index and then label text. Other malformed
outputs stop ingestion and must be corrected or retried before continuing.

The question prompt targets approximately three questions in each Level.
Validation requires all three ordered, nonempty sections with at least one
question per section. There is no automatic model-based response repair.
Replaying archived responses produces deterministic artifacts; fresh model
responses are not guaranteed to be identical, even with a dated model snapshot.

| Final artifact | Meaning |
|---|---|
| `08_labeled_positive_pairs.jsonl` | Validated labels, reasoning, and source `pair_index` |
| `08_positive_pairs.json` | Deduplicated caption-ID pairs |
| `08_text_pairs.json` | Short text labels aligned to the clean pairs |
| `08_pairs_by_bucket.json` | Outputs grouped by modality family |
| `08_stats.json` | Selection, validation, and deduplication counts |
| `run_manifest.json` | Model config, prompt hashes, handoff hashes, stage states, and artifact hashes |

## Version boundary

Pipeline 03 v6 accepts only a Pipeline 02 v6 handoff with schema version 4.
This contract proves that the binary human/animal screen covered the complete
unified case-pair population. Start both pipelines in new run directories;
do not edit a completed manifest or reuse a Pipeline 02 output directory as a
Pipeline 03 run.
