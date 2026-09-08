# Pipeline 01: unified PMC article and figure retrieval

Pipeline 01 builds one retrieval run from one canonical PMC-Patients CSV. It
selects eligible single-patient cases, retrieves their open-access PMC article
packages, extracts figures and captions, and joins those figure records to the
case-similarity graph.

The command line accepts exactly one `--input-csv`. Combining source releases
is deliberately outside this pipeline: if source tables must be reconciled,
that must be completed and reviewed before this workflow starts. Every later
stage consumes the one materialized patient table created at Stage 00.

## Requirements

Python 3.11 is recommended. From this directory:

```bash
python -m pip install -r requirements.txt
```

Run the deterministic offline tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Input contract

The canonical PMC-Patients CSV must contain at least:

| Column | Meaning |
|---|---|
| `patient_uid` | A case identifier such as `6011191-1`. A leading `PMC` and underscore separators are normalized. |
| `similar_patients` | A JSON or Python-literal mapping from patient UID to similarity score. |

Repeated identifiers within the canonical CSV are resolved deterministically:
the later row supplies metadata, while similarity targets are combined and a
later score replaces an earlier score for the same target. The materialization
report records every such occurrence. All additional columns are preserved for
retained cases.

Metadata fields remain strings, including leading zeros, empty fields, and
literal values such as `NA`.

Article multiplicity is calculated across the complete canonical table before
empty similarity mappings are filtered. Articles representing more than one
patient are excluded. Self-links and links to ineligible cases are removed.
An eligible case that is referenced only as a target is retained, even when
its own similarity mapping is empty, so every surviving relationship has both
endpoints in the retrieval cohort.

## Quick start

Run all stages:

```bash
python scripts/run_pipeline.py \
  --input-csv /path/to/PMC-Patients.csv \
  --output-dir /path/to/retrieval_run \
  --user-agent "my-pmc-retrieval/1.0 (contact: name@example.org)"
```

Use a fresh output directory for a clean run. `--limit N` is suitable for a
smoke test, but it limits article resolution and extraction only; Stage 01
still describes the full selected cohort.

## Output artifacts

All paths are relative to `--output-dir`.

| Stage | Artifact | Contents |
|---|---|---|
| 00 | `00_unified_patients.csv` | The sole canonical patient input consumed by Stage 01. |
| 00 | `00_unification_provenance.csv` | Source-row provenance for every materialized patient. |
| 00 | `00_unification_report.json` | Input hash, schema checks, normalization counts, and output hashes. |
| 01 | `01_unique_articles.csv` | Selected patient rows with canonical PMC IDs and filtered similarity lists. |
| 01 | `01_unique_article_ids.json` | Deterministically sorted PMC IDs to retrieve. |
| 01 | `01_stats.json` | Parsing, selection, relationship, and output counts. |
| 02 | `02_files_info.json` | Resolved open-access article-package records. |
| 02 | `02_failed_ids.json` | PMC IDs whose article package could not be resolved. |
| 02 | `02_retrieval_state.json` | Resume state bound to the selected PMC-ID population. |
| 02 | `02_stats.json` | Resolution configuration, counts, and hashes. |
| 03 | `new_data/` | Canonical JPEG images, caption text, and article NXML. |
| 03 | `03_images_captions.csv` | Figure manifest with run-relative paths. |
| 03 | `03_article_failures.json` | Article download or extraction failures. |
| 03 | `03_figure_failures.json` | Figure-level caption, image, or conversion failures. |
| 03 | `03_extraction_state.json` | Resume state and per-article output hashes. |
| 03 | `03_stats.json` | Extraction configuration, counts, and hashes. |
| 04 | `04_merged.csv` | Image-caption rows joined to patients and similarity lists. |
| 04 | `04_image_paths.json` | Portable, run-relative paths for the merged image population. |
| 04 | `04_handoff_report.json` | Join coverage, relationship counts, and asset validation. |
| all | `run_manifest.json` | Pipeline version, unified-data contract, configuration, status, counts, and artifact hashes. |

`new_data/` is the canonical asset-tree name used by the pipeline contract; it
contains the complete population produced by this run.

## Run individual stages

### Stages 00--01: materialize and select cases

```bash
python scripts/01_prepare_articles.py \
  --input-csv /path/to/PMC-Patients.csv \
  --output-dir /path/to/retrieval_run
```

This command materializes the canonical input and then selects the retrieval
cohort. Malformed, missing, non-mapping, and invalid-key similarity values are
reported in `00_unification_report.json` before the normalized table is passed
to Stage 01.

### Stage 02: resolve article packages

```bash
python scripts/02_retrieve_file_info.py \
  --output-dir /path/to/retrieval_run \
  --user-agent "my-pmc-retrieval/1.0 (contact: name@example.org)" \
  --retries 3 \
  --checkpoint-every 100
```

The resolver reads `01_unique_article_ids.json`, locates an open-access
article package for each PMC ID, and records failures explicitly.

### Stage 03: extract figures and captions

```bash
python scripts/03_extract_image_captions.py \
  --output-dir /path/to/retrieval_run \
  --retries 3 \
  --jpeg-quality 95 \
  --checkpoint-every 100
```

Archive paths are validated before extraction. The NXML article is selected
deterministically, figures are processed in document order, and a figure is
retained only when both a resolvable image and a nonempty caption exist. Source
images are decoded and written as RGB JPEG files; transparent pixels are
placed on white, and the first frame is used for a multi-frame image.
Caption paragraph and title boundaries are separated by spaces, while inline
formatting preserves words. Figure manifests retain the numeric document order.

The extraction stage enforces bounded archive, member, document, image, and
pixel sizes. A failure in one figure does not discard other valid figures from
the same article.

### Stage 04: merge and validate

```bash
python scripts/04_merge_dataset.py \
  --output-dir /path/to/retrieval_run
```

The merge normalizes patient identifiers, requires unique patient keys and
unique image/caption paths, and checks every retained JPEG, caption, and NXML
file. The standard all-stage run uses strict validation.

## Cached-source runs

The retrieval and extraction stages can use caller-supplied source caches:

- `--oa-responses-dir`: OA XML responses named `<ID>.xml` or `PMC<ID>.xml`;
- `--archives-dir`: article packages named as recorded in
  `02_files_info.json`.

Example:

```bash
python scripts/run_pipeline.py \
  --input-csv /path/to/PMC-Patients.csv \
  --output-dir /path/to/offline_run \
  --oa-responses-dir /path/to/oa_xml_cache \
  --archives-dir /path/to/article_package_cache
```

These options replace network access with exact copies of the same upstream
source responses. They do not accept downstream pair tables, prior exclusion
lists, or precomputed figure decisions.

## Checkpointing and resume

Stages 02 and 03 write deterministic checkpoints. Resume verifies the
canonical input hash, materialized Stage 00 and Stage 01 artifacts, selected
PMC IDs, extraction settings, package records, manifest rows, and completed
files before skipping work.

```bash
python scripts/run_pipeline.py \
  --input-csv /path/to/PMC-Patients.csv \
  --output-dir /path/to/retrieval_run \
  --resume
```

Changed inputs or completed extraction assets cause resume to fail. A newly
resolved article package may extend an otherwise valid run.

Version 3.1.0 corrects caption spacing, figure ordering, and article identity
selection. Runs created with 3.0.0 need a fresh output directory; their extraction
checkpoints cannot be reused. Preserve the canonical CSV and source caches for
repeatable retrieval, since remote OA responses and archives can change.

## Handoff to Pipeline 02

Pipeline 02 consumes the completed Pipeline 01 run directly:

```bash
python ../02_case_pair_selection_pipeline/scripts/run_pipeline.py prepare \
  --retrieval-run /path/to/retrieval_run \
  --output-dir /path/to/case_pair_run \
  --config ../02_case_pair_selection_pipeline/config.example.toml
```

The Pipeline 01 manifest declares that the run has one materialized patient
input, one merged figure table, and one complete asset tree. Pipeline 02
validates that contract and computes its own diagram decisions and binary
human/animal labels over the unified case-pair population.
