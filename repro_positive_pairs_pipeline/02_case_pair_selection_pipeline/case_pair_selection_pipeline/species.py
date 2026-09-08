"""Deterministic binary human/animal article screening.

The versioned production rules label explicit non-human subjects and animal
experiments as ``animal``. Ambiguous medical uses and human exposure or
zoonosis reports remain ``human``. NXML integrity or parsing failures abort
the preparation run; they are never represented as a third population label.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from lxml import etree

from .articles import parse_nxml, resolve_nxml


HUMAN_ANIMAL_RULESET = "human_animal_v1"
DEFAULT_RULESET = HUMAN_ANIMAL_RULESET
RULESETS = (HUMAN_ANIMAL_RULESET,)

_PATIENT_UID_RE = re.compile(r"^(?P<article>\d+)-(?P<case>\d+)$")
_PMC_ID_RE = re.compile(r"^(?:PMC)?(?P<article>\d+)$", flags=re.IGNORECASE)

_SPECIES_PATTERNS: dict[str, re.Pattern[str]] = {
    "dog": re.compile(r"\b(?:dogs?|pupp(?:y|ies)|canines?)\b", re.I),
    "cat": re.compile(
        r"\b(?:cats?|kittens?|felines?|domestic\s+(?:short|long)hair)\b",
        re.I,
    ),
    "horse": re.compile(r"\b(?:horses?|equines?|pon(?:y|ies)|foals?)\b", re.I),
    "cattle": re.compile(r"\b(?:cattle|cows?|calves?|bovines?)\b", re.I),
    "sheep": re.compile(r"\b(?:sheep|lambs?|ovines?)\b", re.I),
    # "kid" is intentionally omitted because it commonly denotes a human
    # child in case-report titles and abstracts.
    "goat": re.compile(r"\b(?:goats?|caprines?)\b", re.I),
    "pig": re.compile(r"\b(?:pigs?|swine|porcines?|piglets?)\b", re.I),
    "rabbit": re.compile(r"\b(?:rabbits?|lagomorphs?)\b", re.I),
    "ferret": re.compile(r"\bferrets?\b", re.I),
    "bird": re.compile(
        r"\b(?:birds?|avians?|parrots?|parakeets?|chickens?|ducks?|geese)\b",
        re.I,
    ),
    "reptile": re.compile(
        r"\b(?:reptiles?|snakes?|lizards?|turtles?|tortoises?)\b", re.I
    ),
    "rodent": re.compile(
        r"\b(?:mice|mouse|rats?|murine|rodents?|hamsters?|guinea\s+pigs?)\b",
        re.I,
    ),
    "nonhuman_primate": re.compile(
        r"\b(?:non[- ]human\s+primates?|monkeys?|macaques?|baboons?|chimpanzees?)\b",
        re.I,
    ),
    "fish": re.compile(r"\b(?:fish|fishes|zebrafish)\b", re.I),
}
_BREED_PATTERN = re.compile(
    r"\b(?:labrador|retriever|bulldog|beagle|poodle|terrier|dachshund|"
    r"rottweiler|doberman|boxer|german\s+shepherd|shih\s+tzu|persian|"
    r"siamese|maine\s+coon)\b",
    re.I,
)
_VETERINARY_PATTERN = re.compile(
    r"\b(?:veterinar(?:y|ian)|veterinary\s+patient|animal\s+patient)\b",
    re.I,
)
_DIRECT_NONHUMAN_PATTERN = re.compile(
    r"\b(?:non[- ]human|nonhuman|animal\s+case\s+report|veterinary\s+case)\b",
    re.I,
)
_SUBJECT_MARKER = re.compile(
    r"\b(?:patient|case|present(?:ed|ing)?|underwent|had|was|were|showed|"
    r"diagnosed|treated|admitted|referred|suffering|with|year[- ]old|month[- ]old)\b",
    re.I,
)
_IN_SPECIES_CONTEXT = re.compile(
    r"\b(?:in|of|from)\s+(?:a|an|the|one|two|\d+)\b", re.I
)

_EXPOSURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "animal_bite": re.compile(
        r"\b(?:dog|cat|animal|monkey|rat|snake)\s+bites?\b", re.I
    ),
    "animal_attack": re.compile(r"\b(?:dog|cat|animal)\s+attack(?:s|ed)?\b", re.I),
    "cat_scratch": re.compile(r"\bcat[- ]scratch(?:es|ed|ing)?\b", re.I),
    "animal_contact": re.compile(
        r"\b(?:expos(?:ed|ure)|contact)\s+(?:to|with)\s+"
        r"(?:(?:a|an|the)\s+)?(?:animals?|dogs?|cats?|birds?|livestock|"
        r"horses?|rodents?|mice|rats?|monkeys?)\b",
        re.I,
    ),
    "animal_transmission": re.compile(
        r"\b(?:transmi(?:tted|ssion)|acquired)\s+(?:from|by)\s+"
        r"(?:animals?|dogs?|cats?|birds?|livestock|horses?|rodents?|mice|rats?)\b",
        re.I,
    ),
}
_ZOONOSIS_PATTERN = re.compile(
    r"\b(?:zoono(?:sis|ses|tic)|animal[- ]to[- ]human|pet\s+owner)\b", re.I
)
_ANIMAL_MODEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "animal_model": re.compile(
        r"\b(?:animal|murine|mouse|rat|canine|feline)\s+models?\b", re.I
    ),
    "experimental_animal": re.compile(
        r"\b(?:experimental(?:ly)?|laboratory)\s+"
        r"(?:animals?|mice|mouse|rats?|dogs?|cats?)\b",
        re.I,
    ),
    "experimental_subjects": re.compile(
        r"\b(?:mice|mouse|rats?|rabbits?|dogs?|cats?|pigs?|swine|monkeys?)\s+"
        r"(?:were|was)\s+(?:randomi[sz]ed|treated|injected|implanted|sacrificed|"
        r"euthani[sz]ed)\b",
        re.I,
    ),
}
_HUMAN_CONTEXT_PATTERN = re.compile(
    r"\b(?:human\s+(?:patient|case|subject)|man|woman|boy|girl)\b",
    re.I,
)
_AMBIGUOUS_MEDICAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "cat_scan": re.compile(
        r"\bcat\s+scan\b|\bcomputed\s+axial\s+tomography\b", re.I
    ),
    "canine_dental": re.compile(
        r"\bcanine\s+(?:fossa|space|eminence|tooth|teeth|root|roots|"
        r"retraction|guidance)\b|\b(?:maxillary|mandibular|impacted)\s+canine\b",
        re.I,
    ),
    "animal_adjective": re.compile(
        r"\b(?:canine|feline|equine|bovine|ovine|caprine|porcine|murine|avian)\b",
        re.I,
    ),
    "animal_association": re.compile(
        r"\b(?:canine|feline|equine|bovine|ovine|caprine|porcine|murine|avian)"
        r"[- ]associated\b",
        re.I,
    ),
    "dog_ear": re.compile(r"\bdog[- ]ears?\b", re.I),
    "laboratory_reagent": re.compile(
        r"\b(?:mouse|rabbit|goat)\s+(?:anti[- ]|antibody|serum)\b", re.I
    ),
    "fish_assay": re.compile(
        r"\bfluorescence\s+in\s+situ\s+hybridization\b|"
        r"\bfish[- ](?:analysis|assay|test(?:ing)?|probe|probes|study|studies|"
        r"positive|negative)\b",
        re.I,
    ),
}


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _field_species_hits(
    text: str,
    field: str,
    species_patterns: Mapping[str, re.Pattern[str]],
) -> tuple[list[str], bool]:
    reasons: list[str] = []
    explicit_subject = False
    medical_spans = [
        match.span()
        for name, pattern in _AMBIGUOUS_MEDICAL_PATTERNS.items()
        if name != "animal_adjective"
        for match in pattern.finditer(text)
    ]
    for name, pattern in species_patterns.items():
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        reasons.append(f"{field}:species:{name}")
        for match in matches:
            if any(start <= match.start() < end for start, end in medical_spans):
                continue
            before = text[max(0, match.start() - 35) : match.start()]
            after = text[match.end() : min(len(text), match.end() + 45)]
            if (
                _SUBJECT_MARKER.search(before)
                or _SUBJECT_MARKER.search(after)
                or _IN_SPECIES_CONTEXT.search(before)
            ):
                explicit_subject = True
    if explicit_subject:
        reasons.append(f"{field}:explicit_nonhuman_subject")
    return reasons, explicit_subject


def _field_context_hits(text: str, field: str) -> list[str]:
    reasons: list[str] = []
    if _VETERINARY_PATTERN.search(text):
        reasons.append(f"{field}:veterinary_context")
    if _DIRECT_NONHUMAN_PATTERN.search(text):
        reasons.append(f"{field}:direct_nonhuman_subject")
    if _ZOONOSIS_PATTERN.search(text):
        reasons.append(f"{field}:zoonosis")
    if _HUMAN_CONTEXT_PATTERN.search(text):
        reasons.append(f"{field}:human_patient_context")
    if _BREED_PATTERN.search(text):
        reasons.append(f"{field}:ambiguous:breed")
    for name, pattern in _EXPOSURE_PATTERNS.items():
        if pattern.search(text):
            reasons.append(f"{field}:exposure:{name}")
    for name, pattern in _ANIMAL_MODEL_PATTERNS.items():
        if pattern.search(text):
            reasons.append(f"{field}:model:{name}")
    for name, pattern in _AMBIGUOUS_MEDICAL_PATTERNS.items():
        if pattern.search(text):
            reasons.append(f"{field}:ambiguous:{name}")
    return reasons


def human_animal_v1(metadata: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return the binary subject-population label and matched audit rules."""

    reasons: list[str] = []
    explicit_subject = False
    for field in ("title", "abstract"):
        text = str(metadata.get(field) or "").lower()
        species_reasons, field_explicit = _field_species_hits(
            text,
            field,
            _SPECIES_PATTERNS,
        )
        reasons.extend(species_reasons)
        reasons.extend(_field_context_hits(text, field))
        explicit_subject = explicit_subject or field_explicit

    reasons = sorted(set(reasons))
    species_signal = any(":species:" in reason for reason in reasons)
    veterinary_signal = any("veterinary_context" in reason for reason in reasons)
    direct_nonhuman = any(
        reason.endswith(":direct_nonhuman_subject") for reason in reasons
    )
    human_context = any(
        token in reason
        for reason in reasons
        for token in (":exposure:", ":zoonosis", ":human_patient_context")
    )
    animal_model = any(":model:" in reason for reason in reasons)
    if any(reason.startswith("title:model:") for reason in reasons):
        label = "animal"
    elif direct_nonhuman:
        label = "animal"
    elif human_context:
        label = "human"
    elif animal_model or explicit_subject or (veterinary_signal and species_signal):
        label = "animal"
    else:
        label = "human"
    return label, reasons


def classify_metadata(
    metadata: Mapping[str, Any], ruleset: str = DEFAULT_RULESET
) -> dict[str, Any]:
    """Return the versioned automatic binary label for one metadata mapping."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if ruleset != HUMAN_ANIMAL_RULESET:
        raise ValueError(f"Unknown species ruleset: {ruleset!r}")
    label, reasons = human_animal_v1(metadata)
    return {
        "ruleset": ruleset,
        "automatic_label": label,
        "final_label": label,
        "matched_rules": sorted(set(reasons)),
    }


def matched_rule_vocabulary(ruleset: str) -> frozenset[str]:
    """Return every reason label that a versioned ruleset may emit."""

    if ruleset != HUMAN_ANIMAL_RULESET:
        raise ValueError(f"Unknown species ruleset: {ruleset!r}")
    species_names = _SPECIES_PATTERNS
    fields = ("title", "abstract")
    labels = {
        f"{field}:species:{name}"
        for field in fields
        for name in species_names
    }
    labels |= {
        f"{field}:{name}"
        for field in fields
        for name in (
            "veterinary_context",
            "explicit_nonhuman_subject",
            "direct_nonhuman_subject",
            "zoonosis",
            "human_patient_context",
        )
    }
    labels |= {
        f"{field}:exposure:{name}"
        for field in fields
        for name in _EXPOSURE_PATTERNS
    }
    labels |= {
        f"{field}:model:{name}"
        for field in fields
        for name in _ANIMAL_MODEL_PATTERNS
    }
    labels |= {
        f"{field}:ambiguous:{name}"
        for field in fields
        for name in (*_AMBIGUOUS_MEDICAL_PATTERNS, "breed")
    }
    return frozenset(labels)


def _uid_sort_key(uid: str) -> tuple[int, int, str]:
    match = _PATIENT_UID_RE.fullmatch(str(uid).strip())
    if not match:
        raise ValueError(f"Invalid canonical patient_uid: {uid!r}")
    canonical = f"{int(match.group('article'))}-{int(match.group('case'))}"
    if canonical != uid:
        raise ValueError(f"patient_uid is not canonical: {uid!r}")
    return int(match.group("article")), int(match.group("case")), uid


def _normalize_pmc_id(value: Any) -> str:
    match = _PMC_ID_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Invalid pmc_id: {value!r}")
    return str(int(match.group("article")))


def _local_name(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    return etree.QName(element).localname


def _element_text(element: etree._Element) -> str:
    return _normalize_ws(" ".join(element.itertext()))


def _first_descendant_text(root: etree._Element, local_name: str) -> str:
    for element in root.iter():
        if _local_name(element) == local_name:
            return _element_text(element)
    return ""


def _article_metadata(root: etree._Element) -> dict[str, str]:
    title = ""
    for title_group in root.iter():
        if _local_name(title_group) != "title-group":
            continue
        for element in title_group.iter():
            if _local_name(element) == "article-title":
                title = _element_text(element)
                break
        if title:
            break
    if not title:
        title = _first_descendant_text(root, "article-title")

    abstract = _normalize_ws(
        " ".join(
            _element_text(element)
            for element in root.iter()
            if _local_name(element) == "abstract"
        )
    )
    return {
        "title": title,
        "abstract": abstract,
        "journal_title": _first_descendant_text(root, "journal-title"),
    }


def _article_pmc_id(root: etree._Element) -> str | None:
    for element in root.iter():
        if _local_name(element) != "article-id":
            continue
        id_type = str(element.get("pub-id-type", "")).strip().casefold()
        if id_type not in {"pmc", "pmcid"}:
            continue
        value = _element_text(element)
        return _normalize_pmc_id(value) if value else None
    return None


def _base_case_record(case: dict[str, Any], ruleset: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "patient_uid": str(case.get("patient_uid", "")).strip(),
        "pmc_id": str(case.get("pmc_id", "")).strip(),
        "nxml_sha256": str(case.get("nxml_sha256", "")).strip().lower(),
        "ruleset": ruleset,
        "matched_rules": [],
        "title": "",
        "journal_title": "",
        "adjudication": None,
    }


def _classification_worker(
    task: tuple[dict[str, Any], str, str]
) -> dict[str, Any]:
    case, run_root, ruleset = task
    record = _base_case_record(case, ruleset)
    try:
        if case.get("schema_version") != 1:
            raise ValueError("case schema_version must be 1")
        uid = record["patient_uid"]
        _uid_sort_key(uid)
        pmc_id = _normalize_pmc_id(record["pmc_id"])
        if pmc_id != uid.split("-", 1)[0]:
            raise ValueError("case pmc_id does not match patient_uid")
        record["pmc_id"] = pmc_id

        nxml = resolve_nxml(case, run_root)
        root = parse_nxml(nxml)
        article_pmc_id = _article_pmc_id(root)
        if article_pmc_id and article_pmc_id != pmc_id:
            raise ValueError("NXML PMCID does not match the case record")

        metadata = _article_metadata(root)
        record.update(classify_metadata(metadata, ruleset))
        record["title"] = metadata["title"]
        record["journal_title"] = metadata["journal_title"]
        return record
    except etree.XMLSyntaxError:
        record["_preparation_error"] = "XMLSyntaxError: malformed NXML"
    except OSError:
        record["_preparation_error"] = "OSError: unable to read NXML"
    except ValueError as exc:
        record["_preparation_error"] = f"ValueError: {exc}"
    return record


def classify_cases(
    cases: list[dict[str, Any]],
    retrieval_run: str | Path,
    ruleset: str = DEFAULT_RULESET,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Return one deterministic binary species record per Pipeline-02 case.

    Any source-integrity or XML error aborts the whole operation. Such failures
    are preparation errors, not population labels.
    """

    if ruleset not in RULESETS:
        raise ValueError(f"Unknown species ruleset: {ruleset!r}")
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("workers must be at least 1")
    run_root = Path(retrieval_run).resolve()
    if not run_root.is_dir():
        raise ValueError(f"Retrieval run is not a directory: {retrieval_run}")
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("cases must be a list of case objects")

    ordered_cases = sorted(
        cases,
        key=lambda case: _uid_sort_key(str(case.get("patient_uid", "")).strip()),
    )
    uids = [str(case.get("patient_uid", "")).strip() for case in ordered_cases]
    if len(uids) != len(set(uids)):
        raise ValueError("cases contains duplicate patient_uid values")

    tasks = [(case, str(run_root), ruleset) for case in ordered_cases]
    if worker_count == 1 or len(tasks) < 2:
        records = [_classification_worker(task) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (worker_count * 4))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            records = list(
                executor.map(_classification_worker, tasks, chunksize=chunksize)
            )
    failures = [record for record in records if record.get("_preparation_error")]
    if failures:
        first = failures[0]
        raise ValueError(
            "Human/animal screening could not prepare every case: "
            f"failures={len(failures)}; first_patient_uid="
            f"{first.get('patient_uid')!r}; first_error="
            f"{first.get('_preparation_error')}"
        )
    return records


__all__ = [
    "DEFAULT_RULESET",
    "HUMAN_ANIMAL_RULESET",
    "RULESETS",
    "classify_cases",
    "classify_metadata",
    "human_animal_v1",
    "matched_rule_vocabulary",
]
