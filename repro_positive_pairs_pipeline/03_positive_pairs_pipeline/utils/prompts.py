from __future__ import annotations

import json
from typing import Any


CITATION_REASONING_PROMPT = """You are a medical citation analysis assistant. You will read excerpts from a 'citing paper' referencing another 'cited paper,' each describing a single-patient case report. Your task is to produce four pieces of information:

1. Reason for Citation (a short phrase or sentence about why the authors mention the cited paper).
2. Possible Similarities (a brief mention of how these patients’ cases align, if at all).
3. Explanation (one to two sentences summarizing how the citation is used).

----
Follow this exact output format:
Reason for Citation: <short phrase or sentence>
Possible Similarities: <brief mention>
Explanation: <one to two sentences>
----
Read the following excerpt(s) and generate your final answer in the required format. Unify your response if multiple citations are present. If no citations are specified, base your response on the articles' titles and abstracts."""


# """You are a medical citation analysis assistant. You will read excerpts from a citing paper and a cited paper, each describing a single-patient case report.

# Your task is to produce exactly three fields in plain text:

# Reason for Citation: <short phrase or sentence>
# Possible Similarities: <brief mention>
# Explanation: <one to two sentences>

# Use the citation paragraphs when they are available. If no citation paragraphs are available, fall back to the titles and abstracts. Keep the answer concise and clinically focused."""


QUESTION_GENERATION_PROMPT = """You are an experienced clinician specializing in medical imaging. You will be given two cases and the reason for citation between the two (one case report cites the other), and asked to generate a set of comparative, clinically relevant questions that highlight both similarities and differences, helping differentiate potential diagnoses. Your questions should:

1. **Focus on Diagnostic Clues and Clinical Significance:**
   - Avoid trivial or purely technical details (like minor differences in image orientation).
   - Emphasize findings that inform diagnosis (e.g., presence of lesions, pattern of abnormalities, symptoms).

2. **Use a Comparative Framing:**
   - Explicitly compare Case A and Case B.
   - Aim for questions that probe both common ground and distinguishing features.
   - If possible, try asking mostly yes/no questions.

3. **Follow a Hierarchical Reasoning Approach:**
   - **Level 1 (Broad Context)**: Are both cases the same modality/organ system?
   - **Level 2 (General Diagnosis Category)**: Are they both infectious, both neoplastic, etc.?
   - **Level 3 (Specific Features & Findings)**: Detailed imaging findings (e.g., cavitation, consolidation, nodules), clinical presentation, organism type, etc.

4. **Consider Step-by-Step Reasoning:**
   - You may first silently analyze the diagnoses and imaging findings for each case.
   - Then generate questions that a clinician would naturally ask to tease out whether the two cases have the same underlying pathology or different pathologies.

5. **Provide a subtitle specifying the level (1/2/3) of the current set of questions.**"""

# """You are an experienced clinician specializing in medical imaging. You will be given two cases and the reason for citation between them, and you must generate comparative, clinically meaningful questions that help determine whether the cases share the same diagnosis or differ in important ways.

# Requirements:
# 1. Focus on diagnostic clues and clinical significance.
# 2. Frame the questions comparatively across Case A and Case B.
# 3. Start broad and move toward more specific findings.
# 4. Prefer yes/no questions when that makes sense.
# 5. Group the questions under Level 1, Level 2, and Level 3 subtitles.
# 6. Do not add commentary outside the question list."""


ANSWER_GENERATION_PROMPT = """You will be given case details for Case A, Case B, Reason for Citation (one case cites the other), along with a list of questions.
You are an experienced clinician specializing in medical imaging. Previously, a list of comparative, clinically oriented questions was generated for the two cases.
Now, your task is to answer those questions in a way that highlights the diagnostic similarities and differences between the two cases.
DO NOT provide any additional information or context outside of the answers to the questions."""

# """You will be given Case A, Case B, the citation analysis, and a list of comparative clinical questions.

# Answer the questions directly. Keep the answers tied to the case evidence. Do not add extra commentary outside the answers."""


POSITIVE_PAIR_PROMPT = """You are a medical-AI expert.

You will receive:
1. Patient-context questions and answers for a case pair.
2. A dictionary of caption_id -> caption for Case A.
3. A dictionary of caption_id -> caption for Case B.

Your task is to evaluate every inter-case caption pair (one caption from Case A and one caption from Case B) and return only the POSITIVE matches.

A pair is POSITIVE only if:
1. It is inter-case.
2. The two captions describe the same modality family (e.g., CT, MRI, X-ray, ultrasound), ignoring view/plane/sequence details.
3. The two captions describe the same pathology family within the same anatomy / organ system.
4. You ignore decorators such as acuity, severity, laterality, exact slice, or exact level.
5. You do NOT collapse different pathophysiologies just because the images could look visually similar.
6. You do NOT keep diagrams / illustrations.

Output rules:
- Output valid JSON only, with no markdown and no prose.
- Output a JSON array.
- Each item must have exactly these keys:
  {
    "pair_id": ["caption_id_from_case_a", "caption_id_from_case_b"],
    "reasoning": "<short explanation>",
    "modality": "<shared modality family>",
    "anatomy": "<shared anatomy or organ system>",
    "diagnosis": "<shared pathology family>"
  }
- Use caption IDs exactly as they appear in the provided dictionaries.
- If there are no positive matches, output []."""


def build_chat_completion_request(
    *,
    custom_id: str,
    model: str,
    developer_prompt: str,
    user_prompt: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def build_citation_reasoning_user_prompt(record: dict[str, Any]) -> str:
    citation_paragraphs = record.get("citation_paragraphs") or []
    if isinstance(citation_paragraphs, list):
        citation_text = "\n".join(
            f"- {paragraph}"
            for paragraph in citation_paragraphs
            if str(paragraph).strip()
        )
    else:
        citation_text = str(citation_paragraphs).strip()

    if not citation_text:
        citation_text = "(No explicit citation paragraphs found.)"

    return (
        f"Citing Title: {record.get('citing_title', '')}\n"
        f"Citing Abstract: {record.get('citing_abstract', '')}\n"
        f"Citation Paragraphs:\n{citation_text}\n"
        f"Cited Title: {record.get('cited_title', '')}\n"
        f"Cited Abstract: {record.get('cited_abstract', '')}"
    )


def build_question_user_prompt(
    case_a: str,
    case_b: str,
    citation_reasoning: str,
) -> str:
    return f"""Below are two cases and the citation analysis connecting them.

---
Case A:
{case_a}

Case B:
{case_b}

Citation analysis:
{citation_reasoning}
---

Now, **generate the list of comparative, clinically relevant questions** that a radiologist or physician might ask to determine whether these two cases share a diagnosis or differ in key features. 

Remember:
- Maintain clinical depth: mention imaging findings, pathophysiology, or hallmark symptoms.
- Emphasize similarities/differences that impact diagnosis.
- Keep the questions focused and direct, as if you are performing a differential diagnosis.
- Avoid vague or generic questions; ensure each question contributes to understanding diagnostic overlap or divergence.
- Start with broad-level questions (anatomy, broad diagnosis similarity), then proceed to more specific questions about the nature of the condition, causative agents, imaging signs, etc.
- Adhere to yes/no questions, if possible.
- Provide a subtitle specifying the level (1/2/3) of each set of questions.

Provide your final list of questions now."""


def build_answer_user_prompt(
    case_a: str,
    case_b: str,
    citation_reasoning: str,
    questions: str,
) -> str:
    return f"""Below are the two case descriptions, the citation analysis, and the comparative questions.

---
Case A:
{case_a}

Case B:
{case_b}

Citation analysis:
{citation_reasoning}

Questions:
{questions}
---

Provide the answers now."""


def build_positive_user_prompt(
    case_a_captions: dict[str, str],
    case_b_captions: dict[str, str],
    questions: str,
    answers: str,
) -> str:
    captions_a_json = json.dumps(case_a_captions, indent=2, ensure_ascii=False, sort_keys=True)
    captions_b_json = json.dumps(case_b_captions, indent=2, ensure_ascii=False, sort_keys=True)

    return f"""### PATIENT-CONTEXT QUERIES
{questions}

### PATIENT-CONTEXT ANSWERS
{answers}

### CASE A CAPTIONS
{captions_a_json}

### CASE B CAPTIONS
{captions_b_json}"""
