# Reproducible positive-pairs pipeline

This workflow constructs positive image pairs from one canonical PMC-Patients
CSV. Case links identify candidates; a retained image pair must be supported
by the two captions and their case context.

The workflow is divided into three versioned pipelines with separate run
directories and manifests:

| Pipeline | Directory | Main output |
|---|---|---|
| 01 — Article and figure retrieval | `01_data_retrieval_pipeline/` | One validated table of case-linked figures, captions, and article text |
| 02 — Evidence-backed case-pair selection | `02_case_pair_selection_pipeline/` | One eligible case-pair handoff plus complete diagram and human/animal records |
| 03 — Positive image-pair labeling | `03_positive_pair_labeling_pipeline/` | Validated and deduplicated model-labeled image pairs |

```text
One canonical PMC-Patients CSV
  → Pipeline 01: select cases and retrieve figures
  → Pipeline 02: compute diagram decisions
  → construct the complete unordered case-pair population
  → label every case and case pair as human or animal
  → extract case descriptions for human-pair endpoints
  → match citation evidence for human case pairs
  → write the schema-v4 eligible-pair handoff
  → Pipeline 03: validate and snapshot the handoff and screening ledger
  → run four text-only annotation stages
  → validate and deduplicate positive image pairs
```

Pipeline 01 accepts exactly one `--input-csv`. Any reconciliation needed to
create that canonical PMC-Patients table must occur before the pipeline run.

Use each pipeline's documented environment and a separate output directory.
Pipeline 01 produces `04_merged.csv` and `new_data/`; pass its whole run directory
to Pipeline 02's `prepare --retrieval-run`. Pass the resulting selection run
to Pipeline 03's `initialize --selection-run`. The current boundary is
Pipeline 01 v3.1.0 → Pipeline 02 v6.1.0/schema-v4 → Pipeline 03 v6.1.0.

Pipeline 02 computes its filters from the current Pipeline-01 population. Its
species screen writes exactly one `human` or `animal` label for every case and
every unordered case pair. A pair can proceed only when both endpoint cases
are labeled `human`. Malformed inputs stop the run instead of becoming another
classification state.

Pipeline 03 requires the Pipeline-02 v6/schema-v4 handoff and verifies the
complete human/animal ledger before creating model requests. It cannot
initialize from an exclusions-only table or from a partial population.

Each directory contains its input contract, commands, artifacts, and local
tests:

- [01: article and figure retrieval](01_data_retrieval_pipeline/README.md)
- [02: case-pair selection](02_case_pair_selection_pipeline/README.md)
- [03: image-pair labeling](03_positive_pair_labeling_pipeline/README.md)

For a run without network access, Pipeline 01 needs both cached OA XML responses
and cached article archives. Pipeline 02 uses local OCR and article text.
Pipeline 03's `initialize`, `build`, `ingest`, `finalize`, and `verify` commands
are local; `ingest` needs previously saved responses. Building requests does
not generate model answers. Managed Batch commands contact OpenAI and require
`--acknowledge-external-api`.

Reproducing model-derived labels requires the saved response files as well as
the source data, code, environment, and configuration. A dated model name alone
does not guarantee identical answers from a new submission. Preserve completed
run directories and their manifests. Pipeline 01 v3.1.0 requires a fresh run;
Pipelines 02 and 03 bind runs to their source and runtime fingerprints.

`new_batch_reqs/` and `outputs/` contain historical artifacts, outside the
three current run contracts. Image-level vocabulary mapping and BioMedCLIP
training are also outside these pipelines. See [the audit notes](AUDIT.md)
for changes and remaining differences from the supplied thesis draft.
