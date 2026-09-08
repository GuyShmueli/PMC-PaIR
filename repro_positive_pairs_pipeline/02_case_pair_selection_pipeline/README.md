# Pipeline 02: unified case-pair selection

Pipeline 02 consumes one completed Pipeline-01 run. It removes likely
diagrams, constructs unordered case pairs, labels the subject population as
human or animal, extracts case descriptions, matches citations, and publishes
the self-contained handoff used by Pipeline 03.

The workflow has no split-dataset mode. Diagram and human/animal screening are
recomputed from the complete unified input for every run.

## Required input

`--retrieval-run` must identify a completed Pipeline-01 version 3.1.0 run with
the unified dataset contract. It must contain:

- `run_manifest.json`;
- `00_unified_patients.csv`;
- `04_merged.csv`; and
- the canonical `new_data/` article and figure tree.

Pipeline 02 verifies this input before processing. A Pipeline-01 run that
reports retrieval failures is rejected by default. Use
`--allow-upstream-failures` only after reviewing the resulting coverage loss.

## Processing stages

1. **Diagram screening.** Every image is processed using the configured
   white-background and OCR rules.
2. **Case-pair construction.** Remaining figure records are grouped by case.
   Similar-case relations are converted into deterministic unordered pairs and
   deduplicated.
3. **Human/animal screening.** Every participating case receives one label:
   `human` or `animal`. A pair is labeled `human` only when both cases are
   human; otherwise it is labeled `animal`.
4. **Case-description extraction.** Case-related article sections are parsed
   only for cases belonging to a retained human pair. Repeated paragraphs are
   included once.
5. **Citation and eligibility checks.** Citation matching is applied only to
   human pairs. A pair is eligible when a citation is matched, both case
   descriptions are usable, and each case has at least one retained caption.
6. **Handoff.** Eligible cases, eligible pairs, and the complete species-label
   artifacts are bound into handoff schema version 4 for Pipeline 03.

Citation context is stored consistently in the backward-compatible
`citation_paragraphs` field. Values are literal semantic blocks from the citing
article: ordinary citation paragraphs or header-qualified table rows. Numeric
citation ranges and lists are expanded only through unique explicit
`<ref><label>` values. An empty array means that no reliable linked, ranged, or
structured-table context was found. The singular field name
`citation_paragraph` is not part of the handoff schema.

Citation evidence is matched by DOI, PMID, or PMCID, in that order, with an
optional guarded title fallback. A reference with conflicting identifiers is
not accepted. Canonical A-to-B direction is tried first; an ambiguous match in
that direction does not prevent accepting a verified B-to-A citation. A matched
bibliography entry is sufficient even when `citation_paragraphs` is empty.

## Human/animal rules

The `[species_filter]` configuration has one versioned production ruleset:

```toml
[species_filter]
ruleset = "human_animal_v1"
```

The ruleset uses article titles and abstracts to label explicit veterinary or
other non-human subjects and experimental animal/model studies as `animal`. Human animal-exposure and
zoonosis reports remain `human`. Medical uses of animal words—such as CAT
scan, canine tooth, FISH assay, and animal-derived reagents—also remain
`human` unless the article explicitly describes a non-human subject.
Journal titles are recorded as provenance and do not affect the label.
An animal-model study identified in the title is excluded; an animal-model
mention confined to the abstract does not override explicit human patient
context. These are text heuristics and cannot resolve every ambiguous report.

An unreadable, changed, unsafe, or malformed article is a preparation failure.
It stops the run and is never represented as a third population label.

Optional governed corrections can be supplied with
`--species-adjudications`. Each JSONL row has the exact schema:

```json
{"schema_version":1,"patient_uid":"12345-1","label":"human","reason":"Qualified review confirmed a human case."}
```

`label` must be `human` or `animal`. The automatic and final labels, together
with the reason for an override, remain in the case-classification artifact.

This is a transparent dataset filter, not a clinically validated species
classifier.

## Complete pair labels

`02_pair_species.jsonl` contains exactly one row for every row in
`01_pairs.jsonl`, in the same `pair_index` order. Each row records the pair,
the human/animal label of each endpoint, and the resulting pair label. The
verifier requires:

```text
human_pairs + animal_pairs = pairs_before_screen
```

This proves that the screen covered the complete unified case-pair population.

## Run

Create the environment and run from this directory:

```bash
conda env create -f environment.yml
conda activate repro-case-pair-selection

python scripts/run_pipeline.py prepare \
  --retrieval-run /path/to/completed_pipeline01_run \
  --config config.example.toml \
  --output-dir /path/to/new_pipeline02_run \
  --workers 8
```

To apply governed label corrections, add:

```bash
  --species-adjudications /path/to/species_adjudications.jsonl
```

Verify a completed run with:

```bash
python scripts/run_pipeline.py verify \
  --output-dir /path/to/completed_pipeline02_run
```

Output directories are published atomically and must be new or empty.

## Main artifacts

| Artifact | Purpose |
|---|---|
| `00_diagram_classifications.jsonl` | One diagram result per unified image record. |
| `00_diagram_blacklist.json` | Image paths excluded by the diagram screen. |
| `01_cases.jsonl` | Case records and retained captions. |
| `01_pairs.jsonl` | All deterministic unordered case pairs. |
| `02_case_summaries.jsonl` | Case descriptions and parse status for endpoints of human pairs. |
| `02_species_classifications.jsonl` | One human/animal label per participating case. |
| `02_pair_species.jsonl` | One human/animal label per constructed case pair. |
| `02_species_stats.json` | Complete case- and pair-level label counts. |
| `03_citations.jsonl` | Citation results for human pairs only. |
| `03_selection.json` | Eligible and selected case pairs. |
| `03_eligible_cases.jsonl` | Self-contained Pipeline-03 case input. |
| `03_eligible_pairs.jsonl` | Self-contained Pipeline-03 pair input. |
| `03_handoff_manifest.json` | Handoff schema, ordering contracts, and artifact metadata. |
| `run_manifest.json` | Run configuration, stages, and artifact provenance. |

The handoff declares
`population_filter = "complete_case_pair_human_animal_screen_required"` and
binds the three species-screen artifacts together with the eligible case and
pair records.

## Thesis alignment

The default diagram thresholds exclude images with a white-pixel ratio above
0.5, then exclude remaining images with more than 200 OCR characters. The
species screen runs before case-text extraction and citation verification.
Unordered case links remain candidates for Pipeline 03; they are not positive
image labels.

The citation extractor also handles linked tables and unambiguous numeric
reference ranges. The thesis's citation-context paragraph should mention these
in addition to ordinary paragraphs. The corrected species and citation rules
can change selection counts; the draft's 774 animal cases and final dataset
counts describe the historical run and have not been regenerated by this audit.

## Tests

```bash
PYTHONPATH=. pytest -q
```

The tests use synthetic articles and images. They cover unified-input
enforcement, diagram screening, binary human/animal rules, governed label
overrides, complete pair coverage, citation selection, handoff validation,
path safety, atomic failure, and tamper detection.

## Repair a legacy unified citation-context file

The standalone recovery command repairs only empty `extracted_citation`
values in the ad-hoc four-field unified JSON. It does not change the source
`03_citation_records.json`, any existing extracted context, pair identifiers,
or `citation_reason`. Run it once without `--apply` to validate and preview the
counts, then repeat with `--apply` for an atomic replacement and a complete
audit artifact:

```bash
python scripts/recover_case_pair_citations.py \
  --citation-records /path/to/outputs/03_citation_records.json \
  --case-pair-citations /path/to/outputs/case_pair_citations.json \
  --audit-output /path/to/outputs/case_pair_citations_recovery_audit.json \
  --workers 8

python scripts/recover_case_pair_citations.py \
  --citation-records /path/to/outputs/03_citation_records.json \
  --case-pair-citations /path/to/outputs/case_pair_citations.json \
  --audit-output /path/to/outputs/case_pair_citations_recovery_audit.json \
  --workers 8 \
  --apply
```

The command is idempotent for already populated rows. Unresolved rows remain
empty and are enumerated as `no_reliable_intext_marker_found` in the audit;
they are never filled from titles, abstracts, or generated text.
