# Pipeline 02 data governance

## Unified population boundary

Pipeline 02 accepts one completed Pipeline-01 unified run. Diagram and
human/animal screening are computed over the complete population represented
by that run. Saved exclusion lists are not production inputs.

## Human/animal screening

Every case in the constructed pair graph receives one automatic label:
`human` or `animal`. Every pair also receives one label. A pair is `human` only
when both endpoint cases are human, and only human pairs can continue to
citation matching and Pipeline 03.

The ruleset is part of the immutable run specification. It is heuristic:
absence of an animal rule hit is not proof that an article reports a human
subject. Its behavior should be evaluated on explicit non-human cases, animal
experiments, human exposure and zoonosis reports, and ambiguous medical terms.

Optional adjudications are governed inputs. Each row identifies one current-run
case, assigns `human` or `animal`, and gives a nonempty reason. Both the
automatic and final labels remain auditable, and the adjudication file is
bound to the run.

Article integrity, path, hash, or XML parsing failures stop preparation. They
cannot be converted into a population label.

## Sensitive content

The compact species-classification artifact includes article titles and
journal names but excludes abstracts. Case descriptions, captions, citation
paragraphs, and article text may still contain sensitive clinical details.
Access and release must follow the project privacy review.

## Release interpretation

Diagram filtering, human/animal screening, citation matching, and later
model-assisted labeling are selection procedures, not clinical validation. A
completed technical run does not establish clinical correctness,
representativeness, or redistribution permission.

Before release, assess selected image pairs manually and confirm article- and
figure-level redistribution rights.
