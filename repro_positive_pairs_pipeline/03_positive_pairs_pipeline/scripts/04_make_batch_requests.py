from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from utils.artifacts import artifact_path, batch_stage_path
from utils.io_utils import load_json, require_no_missing, save_json, write_jsonl
from utils.prompts import (
    ANSWER_GENERATION_PROMPT,
    CITATION_REASONING_PROMPT,
    POSITIVE_PAIR_PROMPT,
    QUESTION_GENERATION_PROMPT,
    build_answer_user_prompt,
    build_chat_completion_request,
    build_citation_reasoning_user_prompt,
    build_positive_user_prompt,
    build_question_user_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenAI batch request JSONL files for the pipeline."
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    citation = subparsers.add_parser("citation", help="Build citation-reasoning requests.")
    citation.add_argument("--output-dir", required=True)
    citation.add_argument("--model", default="o4-mini")
    citation.add_argument("--reasoning-effort", default="medium")
    citation.add_argument("--limit", type=int, default=None)

    question = subparsers.add_parser("question", help="Build question-generation requests.")
    question.add_argument("--output-dir", required=True)
    question.add_argument("--model", default="o4-mini")
    question.add_argument("--reasoning-effort", default="low")
    question.add_argument("--limit", type=int, default=None)

    answer = subparsers.add_parser("answer", help="Build answer-generation requests.")
    answer.add_argument("--output-dir", required=True)
    answer.add_argument("--model", default="gpt-5-mini")
    answer.add_argument("--reasoning-effort", default=None)
    answer.add_argument("--limit", type=int, default=None)

    positive = subparsers.add_parser("positive", help="Build positive-pair classification requests.")
    positive.add_argument("--output-dir", required=True)
    positive.add_argument("--model", default="gpt-5-mini")
    positive.add_argument("--reasoning-effort", default=None)
    positive.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def _truncate_if_needed(values, limit):
    if limit is None:
        return values
    return values[:limit]


def build_citation_requests(output_dir: Path, *, model: str, reasoning_effort: str | None, limit: int | None) -> None:
    citation_records = load_json(artifact_path(output_dir, "citation_records_json", create_parent=False))
    citation_valid_indices = load_json(artifact_path(output_dir, "citation_valid_indices_json", create_parent=False))

    if len(citation_records) != len(citation_valid_indices):
        raise ValueError(
            f"Length mismatch: {len(citation_records)=} vs {len(citation_valid_indices)=}"
        )

    citation_records = _truncate_if_needed(citation_records, limit)
    citation_valid_indices = _truncate_if_needed(citation_valid_indices, limit)

    requests = []
    for idx, record in enumerate(citation_records, start=1):
        requests.append(
            build_chat_completion_request(
                custom_id=f"request-{idx}",
                model=model,
                developer_prompt=CITATION_REASONING_PROMPT,
                user_prompt=build_citation_reasoning_user_prompt(record),
                reasoning_effort=reasoning_effort,
            )
        )

    write_jsonl(requests, batch_stage_path(output_dir, "citation", "requests"))
    save_json(citation_valid_indices, batch_stage_path(output_dir, "citation", "request_indices"))

    print(f"Citation requests written: {len(requests)}")
    print(batch_stage_path(output_dir, "citation", "requests"))


def build_question_requests(output_dir: Path, *, model: str, reasoning_effort: str | None, limit: int | None) -> None:
    case_summary_pairs = load_json(artifact_path(output_dir, "case_summary_pairs_json", create_parent=False))
    common_indices = load_json(artifact_path(output_dir, "common_indices_json", create_parent=False))
    citation_valid_indices = load_json(artifact_path(output_dir, "citation_valid_indices_json", create_parent=False))
    citation_texts = load_json(batch_stage_path(output_dir, "citation", "texts", create_parent=False))

    require_no_missing(citation_texts, label="citation_texts")

    if len(citation_valid_indices) != len(citation_texts):
        raise ValueError(
            f"Length mismatch: {len(citation_valid_indices)=} vs {len(citation_texts)=}"
        )

    citation_text_by_pair_index = {
        pair_index: citation_text
        for pair_index, citation_text in zip(citation_valid_indices, citation_texts)
    }

    common_indices = _truncate_if_needed(common_indices, limit)

    requests = []
    for request_idx, pair_index in enumerate(common_indices, start=1):
        case_a, case_b = case_summary_pairs[pair_index]
        citation_reasoning = citation_text_by_pair_index[pair_index]
        requests.append(
            build_chat_completion_request(
                custom_id=f"request-{request_idx}",
                model=model,
                developer_prompt=QUESTION_GENERATION_PROMPT,
                user_prompt=build_question_user_prompt(case_a, case_b, citation_reasoning),
                reasoning_effort=reasoning_effort,
            )
        )

    write_jsonl(requests, batch_stage_path(output_dir, "question", "requests"))
    save_json(common_indices, batch_stage_path(output_dir, "question", "request_indices"))

    print(f"Question requests written: {len(requests)}")
    print(batch_stage_path(output_dir, "question", "requests"))


def build_answer_requests(output_dir: Path, *, model: str, reasoning_effort: str | None, limit: int | None) -> None:
    case_summary_pairs = load_json(artifact_path(output_dir, "case_summary_pairs_json", create_parent=False))
    common_indices = load_json(artifact_path(output_dir, "common_indices_json", create_parent=False))
    citation_valid_indices = load_json(artifact_path(output_dir, "citation_valid_indices_json", create_parent=False))
    citation_texts = load_json(batch_stage_path(output_dir, "citation", "texts", create_parent=False))
    question_texts = load_json(batch_stage_path(output_dir, "question", "texts", create_parent=False))

    require_no_missing(citation_texts, label="citation_texts")
    require_no_missing(question_texts, label="question_texts")

    if len(citation_valid_indices) != len(citation_texts):
        raise ValueError(
            f"Length mismatch: {len(citation_valid_indices)=} vs {len(citation_texts)=}"
        )
    if len(common_indices) != len(question_texts):
        raise ValueError(
            f"Length mismatch: {len(common_indices)=} vs {len(question_texts)=}"
        )

    citation_text_by_pair_index = {
        pair_index: citation_text
        for pair_index, citation_text in zip(citation_valid_indices, citation_texts)
    }

    common_indices = _truncate_if_needed(common_indices, limit)
    question_texts = _truncate_if_needed(question_texts, limit)

    requests = []
    for request_idx, (pair_index, questions) in enumerate(zip(common_indices, question_texts), start=1):
        case_a, case_b = case_summary_pairs[pair_index]
        citation_reasoning = citation_text_by_pair_index[pair_index]
        requests.append(
            build_chat_completion_request(
                custom_id=f"request-{request_idx}",
                model=model,
                developer_prompt=ANSWER_GENERATION_PROMPT,
                user_prompt=build_answer_user_prompt(case_a, case_b, citation_reasoning, questions),
                reasoning_effort=reasoning_effort,
            )
        )

    write_jsonl(requests, batch_stage_path(output_dir, "answer", "requests"))
    save_json(common_indices, batch_stage_path(output_dir, "answer", "request_indices"))

    print(f"Answer requests written: {len(requests)}")
    print(batch_stage_path(output_dir, "answer", "requests"))


def build_positive_requests(output_dir: Path, *, model: str, reasoning_effort: str | None, limit: int | None) -> None:
    pair_caption_dicts = load_json(artifact_path(output_dir, "pair_caption_dicts_json", create_parent=False))
    common_indices = load_json(artifact_path(output_dir, "common_indices_json", create_parent=False))
    question_texts = load_json(batch_stage_path(output_dir, "question", "texts", create_parent=False))
    answer_texts = load_json(batch_stage_path(output_dir, "answer", "texts", create_parent=False))

    require_no_missing(question_texts, label="question_texts")
    require_no_missing(answer_texts, label="answer_texts")

    if len(common_indices) != len(question_texts):
        raise ValueError(
            f"Length mismatch: {len(common_indices)=} vs {len(question_texts)=}"
        )
    if len(common_indices) != len(answer_texts):
        raise ValueError(
            f"Length mismatch: {len(common_indices)=} vs {len(answer_texts)=}"
        )

    positive_request_indices = []
    requests = []

    for pair_index, questions, answers in zip(common_indices, question_texts, answer_texts):
        pair_caption_group = pair_caption_dicts[pair_index]
        if not isinstance(pair_caption_group, list) or len(pair_caption_group) != 2:
            continue

        case_a_captions = pair_caption_group[0] or {}
        case_b_captions = pair_caption_group[1] or {}
        if not case_a_captions or not case_b_captions:
            continue

        positive_request_indices.append(pair_index)
        requests.append(
            build_chat_completion_request(
                custom_id=f"request-{len(requests) + 1}",
                model=model,
                developer_prompt=POSITIVE_PAIR_PROMPT,
                user_prompt=build_positive_user_prompt(
                    case_a_captions,
                    case_b_captions,
                    questions,
                    answers,
                ),
                reasoning_effort=reasoning_effort,
            )
        )

    if limit is not None:
        requests = requests[:limit]
        positive_request_indices = positive_request_indices[:limit]

    write_jsonl(requests, batch_stage_path(output_dir, "positive", "requests"))
    save_json(positive_request_indices, batch_stage_path(output_dir, "positive", "request_indices"))

    print(f"Positive requests written: {len(requests)}")
    print(batch_stage_path(output_dir, "positive", "requests"))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.stage == "citation":
        build_citation_requests(
            output_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
        )
    elif args.stage == "question":
        build_question_requests(
            output_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
        )
    elif args.stage == "answer":
        build_answer_requests(
            output_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
        )
    elif args.stage == "positive":
        build_positive_requests(
            output_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
        )
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
