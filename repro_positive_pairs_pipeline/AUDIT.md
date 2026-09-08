# Pipeline audit — 2026-09-05

Reviewed the three active pipelines against the supplied PMC-PaIR thesis draft.
Historical outputs and batch requests were left unchanged. No API key was read
or used, and no LLM requests were sent.

## Changes

| Area | Correction |
|---|---|
| Patient materialization | Removed obsolete multi-source bookkeeping while preserving the single-source provenance schema and duplicate-row policy. Metadata such as leading-zero strings and literal `NA` survives cohort selection. |
| Figure extraction | Separated adjacent caption paragraphs, kept numeric figure order through the merged handoff, restricted article identity to its own front matter, and removed partial files after failed writes. |
| Retrieval resume | Pipeline 01 is now v3.1.0; extraction checks the saved version before reusing results. Pipeline 02 and the integration fixture accept this version. |
| Species filtering | Decisions use titles and abstracts; journal names remain provenance only. Filtering precedes case-description extraction. Ordinary medical animal terms no longer override a separately identified animal patient. |
| Citation matching | Conflicting identifiers cannot confirm a reference. An ambiguous match in one direction still permits verification in the other direction. |
| Article parsing | Shared the repeated NXML path, hash, and parser implementation across selection stages. Removed obsolete Pillow compatibility branches. |
| Model prompts | Question generation requests approximately three questions per level. Positive proposals require the same anatomical region and use the draft's stated allowable differences. Prompt version is 1.1.0. |
| Response ingestion | File and directory inputs use one normalizer, including directories supplied through the CLI. Removed duplicate hashing and redundant sharding checks. |
| Managed batches | Error-only completed batches can be downloaded, recorded as validation failures, and explicitly retried. SDK retries are disabled so batch creation cannot silently bypass the recorded submission state. |
| Documentation | Corrected stage order, environment name, caption-pair ordering, response normalization, and the distinction between archived-response replay and new model inference. |

The pair-selection and labeling schema remains v4 with pipeline versions
6.1.0. Their implementation fingerprints distinguish the revised code. Existing
completed runs require their original source and environment for verification;
do not rewrite their manifests. Start corrected retrieval runs in new directories.

## Validation

Used the existing offline suites, extending a few cases for the identified
regressions. Pipeline 01: 17 tests passed. Pipeline 02: 51 tests passed, with
focused follow-up checks after the final species changes. Pipeline 03: 43
distinct tests passed across configuration, batches, schemas, finalization,
offline replay, and the Pipeline 02 handoff.

A temporary end-to-end check exercised real Pipeline 01 retrieval from
synthetic article archives, Pipeline 02 selection, and all four Pipeline 03
build/ingest stages. Two independent runs produced byte-identical final pairs,
labels, text descriptions, statistics, and labeling manifests. Network sockets
were blocked; OCR and model responses were synthetic. The temporary check was
not added as another permanent test suite.

Checks ran on Python 3.11.2 with the installed packages, including pandas 1.5.3,
Pillow 9.4.0, and lxml 4.9.2. The pinned production environments, real OCR
runtime, and live Batch service were not validated. No dependencies were installed.

## Thesis points still requiring reconciliation

- The draft's 138,246 cases, 159,841 case relationships, 774 animal cases,
  86,316 retained image pairs, and 87,105 images were not regenerated. The
  historical `outputs/postprocessed/final_stats.json` reports 62,520 retained
  pairs for that artifact set. It is not evidence for the draft's final release
  count; identify and retain the exact release inputs, manifests, and responses.
- Figure extraction handles supported raster formats and selects one resolvable
  graphic per figure. “Regardless of their original file type” is broader than
  the implementation. The current source contract also requires selected
  single-patient UIDs to end in `-1`.
- Citation context includes header-qualified table rows and reliably expanded
  numeric citation ranges as well as inline-reference paragraphs. The draft
  currently describes only paragraphs.
- Response normalization trims whitespace and accepts an enclosing JSON code
  fence. Other malformed responses are rejected for correction and re-ingestion
  or explicit retry. The draft's statement that malformed responses “were
  adjusted” should specify the historical procedure used.
- Species screening remains a text heuristic. Explicit human-patient context
  takes precedence over background animal-model mentions; an animal-model title
  still identifies an animal study. New results should be counted from the saved
  classification ledger, including any adjudications.
- Image-level vocabulary mapping and BioMedCLIP training are outside these three
  pipelines. The draft describes 19 disease-or-etiology labels in the background
  but 20 plus `other` in the image-level methods. Its broad description of
  caption-derived labels also needs reconciliation with the more specific
  finding and fallback-diagnosis routes in those methods.

Exact model-label replay requires archived responses. Model snapshots and prompt
hashes record the requested inference configuration; they do not guarantee that
a new paid submission will reproduce the same answers.
