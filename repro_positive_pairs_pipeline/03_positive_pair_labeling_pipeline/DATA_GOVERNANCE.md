# Data governance and external processing

This pipeline was designed for public PMC case-report material. That does not
make every source article freely redistributable: PMC articles retain their
article-specific licenses. Before publishing inputs, prompts, raw responses,
captions, images, or derived labels, review and record the license for every
source article and figure.

The direct scientific input is one hash-bound Pipeline 02 handoff containing
selected case descriptions, citation evidence, and caption groups, together
with the three canonical human/animal-screen artifacts. The complete
pair-species table records a binary label for every constructed relation in
the unified population. Pipeline 03 snapshots
these inputs in its own run and does not read the Level-1 asset tree. Retain
both pipeline manifests so the final labels remain traceable through the
selection run to the unified Level-1 source.

Pipeline 03 accepts only the Pipeline 02 v6 schema-v4 handoff. Every imported
case must carry the configured species ruleset and a final label of `human`,
and every imported pair must map exactly to a human row in the complete
pair-species table. The source manifest must bind the case classifications,
pair labels, and statistics artifacts to a completed human/animal-screen
stage. This records that the case passed the selected heuristic; it is not
proof that every retained article describes a human
patient. Preserve any governed adjudication input with the two pipeline runs
for population-scope audits.

The four model stages send the following text to OpenAI when `submit` is used:

- article titles and abstracts;
- citation paragraphs;
- extracted case-presentation text;
- figure captions; and
- model-generated questions and answers from preceding stages.

No image bytes are submitted by this implementation. The code reads the API
credential only through the OpenAI SDK (`OPENAI_API_KEY`) and never writes the
credential to a run artifact.

Do not use the live submission commands with identifiable, restricted, or
contractually controlled data until the responsible investigator has confirmed
the applicable consent, IRB/ethics, DUA, retention, regional-processing, and
provider-account requirements. `initialize`, offline `ingest`, `finalize`, and
`verify` do not contact OpenAI.

Model outputs are generated annotations, not clinical ground truth. A released
dataset should identify the exact model snapshot and prompts, disclose the
automated labeling procedure, include validation/error analysis, and state
that the result is for research rather than patient care.
