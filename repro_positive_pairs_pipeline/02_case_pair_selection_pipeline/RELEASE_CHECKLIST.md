# Pipeline 02 release checklist

Before accepting a Pipeline-02 run:

- confirm the producer is Pipeline 01 version 3.1.0 with exactly one unified
  patient input;
- confirm the Pipeline-01 run and complete asset tree pass integrity checks;
- document any accepted upstream retrieval failures and resulting coverage
  loss;
- record the diagram-screen configuration and OCR runtime;
- confirm diagram classifications cover every unified figure record;
- record the `human_animal_v1` species ruleset;
- confirm `02_species_classifications.jsonl` covers every case in the
  constructed pair graph and contains only `human` or `animal` labels;
- confirm `02_pair_species.jsonl` has exactly one row for every
  `01_pairs.jsonl` row, with identical pair indices and keys;
- confirm human and animal pair counts sum to all constructed pairs;
- preserve and review any governed species-adjudication input;
- confirm citations were evaluated only for human pairs;
- run `python scripts/run_pipeline.py verify` successfully;
- confirm handoff schema version 4 binds the eligible cases, eligible pairs,
  and all three species-screen artifacts;
- evaluate the screen on explicit non-human subjects, animal-model studies,
  human exposure and zoonosis reports, and ambiguous medical terminology;
- review selected pairs for clinical relevance and labeling quality; and
- complete privacy, redistribution-rights, and project release review.

Do not interpret structural completion as clinical validation or
public-release approval.
