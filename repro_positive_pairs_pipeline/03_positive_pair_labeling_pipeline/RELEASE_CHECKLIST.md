# Release checklist

The pipeline deliberately does not choose a software license on behalf of the
repository owner. Before public release:

- confirm that the Pipeline 03 manifest binds one completed Pipeline 02 handoff,
  and archive both manifests plus the handoff tables;
- confirm that the imported handoff uses schema version 4 and that every
  eligible case is labeled `human` under the reported ruleset;
- confirm that Pipeline 03's handoff stage snapshots the hash-bound species
  classifications, complete pair labels, and statistics from a completed
  Pipeline 02 `02_human_animal_screen` stage;
- confirm that the pair-species table covers the complete constructed-pair
  population and that every selected pair maps exactly to a `human` row, and archive any
  governed adjudication input;
- add an owner-approved `LICENSE` and author-approved `CITATION.cff`;
- record source-article and figure licenses and redistribution decisions;
- run the complete offline test suite in a clean Python 3.11 environment;
- perform an explicitly approved, bounded live Batch smoke test;
- validate the selected model snapshots and prompt versions on adjudicated
  examples;
- archive the run manifest, raw batch outputs, and dependency lock used for the
  published result; and
- remove or document any failed/partial runs before distributing artifacts.
