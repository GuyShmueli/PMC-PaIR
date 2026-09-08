from __future__ import annotations

import json
from typing import Any

from .io_utils import sha256_bytes


PROMPT_VERSION = "1.2.0"

CITATION_REASONING_PROMPT = """You are a medical citation analysis assistant. Compare a citing single-patient case report with the cited single-patient case report. Return exactly three concise plain-text fields:

Reason for Citation: <short phrase or sentence>
Possible Similarities: <brief mention>
Explanation: <one or two sentences>

Use the citation paragraphs when available. Otherwise use only the supplied titles and abstracts. Do not introduce facts absent from the supplied articles."""

QUESTION_GENERATION_PROMPT = """You are an experienced clinician specializing in medical imaging. Generate comparative, clinically relevant questions for Case A and Case B. Focus on diagnostic clues, anatomy, modality, pathology family, imaging findings, and meaningful clinical differences. Prefer direct yes/no questions when useful. Return only three nonempty sections, in this exact order, headed Level 1 (broad context), Level 2 (diagnostic category), and Level 3 (specific findings). Aim for approximately three questions per section, with every question ending in a question mark."""

ANSWER_GENERATION_PROMPT = """You are an experienced clinician specializing in medical imaging. Answer the supplied comparative questions using only Case A, Case B, and the citation analysis. Highlight diagnostic similarities and differences. Return only three nonempty answer sections, in this exact order, headed Level 1, Level 2, and Level 3. Do not add a preamble or closing commentary."""

POSITIVE_PAIR_PROMPT = """You are a medical-AI expert. Evaluate every inter-case caption pair, using the supplied patient-context questions and answers.

A pair is positive only when both captions describe the same modality family, the same pathology family, and the same anatomical region. Differences in view, plane, sequence, laterality, acuity, severity, or slice do not by themselves prevent a match. Do not collapse different pathophysiologies merely because images can look similar. Reject diagrams and illustrations.

Return valid JSON only: an array whose items have exactly these keys:
{"pair_id":["caption_id_from_case_a","caption_id_from_case_b"],"modality":"shared modality family","anatomy":"shared anatomy","diagnosis":"shared pathology family"}

Use caption IDs exactly as supplied. Return [] when there are no positive matches."""

DEVELOPER_PROMPTS = {
    "citation": CITATION_REASONING_PROMPT,
    "question": QUESTION_GENERATION_PROMPT,
    "answer": ANSWER_GENERATION_PROMPT,
    "positive": POSITIVE_PAIR_PROMPT,
}


def prompt_metadata() -> dict[str, Any]:
    return {
        "version": PROMPT_VERSION,
        "sha256": {
            stage: sha256_bytes(text.encode("utf-8"))
            for stage, text in DEVELOPER_PROMPTS.items()
        },
    }


def citation_user_prompt(record: dict[str, Any]) -> str:
    paragraphs = record.get("citation_paragraphs") or []
    citation_text = "\n".join(f"- {str(item).strip()}" for item in paragraphs if str(item).strip())
    if not citation_text:
        citation_text = "(No explicit citation paragraph was found.)"
    return (
        f"Citing Title: {record.get('citing_title', '')}\n"
        f"Citing Abstract: {record.get('citing_abstract', '')}\n"
        f"Citation Paragraphs:\n{citation_text}\n"
        f"Cited Title: {record.get('cited_title', '')}\n"
        f"Cited Abstract: {record.get('cited_abstract', '')}"
    )


def question_user_prompt(case_a: str, case_b: str, citation_text: str) -> str:
    return f"Case A:\n{case_a}\n\nCase B:\n{case_b}\n\nCitation analysis:\n{citation_text}"


def answer_user_prompt(
    case_a: str,
    case_b: str,
    citation_text: str,
    questions: str,
) -> str:
    return (
        f"Case A:\n{case_a}\n\nCase B:\n{case_b}\n\n"
        f"Citation analysis:\n{citation_text}\n\nQuestions:\n{questions}"
    )


def positive_user_prompt(
    case_a_captions: dict[str, str],
    case_b_captions: dict[str, str],
    questions: str,
    answers: str,
) -> str:
    return (
        f"PATIENT-CONTEXT QUESTIONS\n{questions}\n\n"
        f"PATIENT-CONTEXT ANSWERS\n{answers}\n\n"
        "CASE A CAPTIONS\n"
        + json.dumps(case_a_captions, ensure_ascii=False, sort_keys=True)
        + "\n\nCASE B CAPTIONS\n"
        + json.dumps(case_b_captions, ensure_ascii=False, sort_keys=True)
    )


def request_body(
    *,
    stage: str,
    model: str,
    reasoning_effort: str | None,
    user_prompt: str,
) -> dict[str, Any]:
    if stage not in DEVELOPER_PROMPTS:
        raise ValueError(f"Unknown model stage: {stage}")
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "developer", "content": DEVELOPER_PROMPTS[stage]},
            {"role": "user", "content": user_prompt},
        ],
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body
