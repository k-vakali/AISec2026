#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    description: str
    prompt_text: str


PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    "label": PromptTemplate(
        prompt_id="label",
        description="Binary phishing classification only",
        prompt_text=(
            "You are analysing an email for phishing detection.\n\n"
            "Task:\n"
            "Decide whether the email is phishing or benign.\n\n"
            "Output format:\n"
            "LABEL: <0 or 1>\n\n"
            "Rules:\n"
            "- Use 1 for phishing.\n"
            "- Use 0 for benign.\n"
            "- Do not output anything else."
        ),
    ),

    "evidence": PromptTemplate(
        prompt_id="evidence",
        description="Binary classification with supporting evidence",
        prompt_text=(
            "You are analysing an email for phishing detection.\n\n"
            "Task:\n"
            "Decide whether the email is phishing or benign.\n"
            "Then give short evidence from the email.\n\n"
            "Output format:\n"
            "LABEL: <0 or 1>\n"
            "EVIDENCE:\n"
            "- <evidence point 1>\n"
            "- <evidence point 2>\n"
            "- <evidence point 3>\n\n"
            "Rules:\n"
            "- Use 1 for phishing.\n"
            "- Use 0 for benign.\n"
            "- Base the evidence only on the email content."
        ),
    ),

    "guided": PromptTemplate(
        prompt_id="guided",
        description="Guided phishing analysis prompt",
        prompt_text=(
            "You are analysing an email for phishing detection.\n\n"
            "Task:\n"
            "Decide whether the email is phishing or benign.\n\n"
            "When making your decision, pay attention to:\n"
            "- sender address and reply-to address\n"
            "- suspicious links or domains\n"
            "- urgent language or pressure tactics\n"
            "- requests for login, payment, verification, or personal information\n"
            "- spelling, grammar, or formatting issues\n"
            "- brand impersonation or mismatched identities\n"
            "- authentication information if present\n\n"
            "Output format:\n"
            "LABEL: <0 or 1>\n"
            "EVIDENCE:\n"
            "- <evidence point 1>\n"
            "- <evidence point 2>\n"
            "- <evidence point 3>\n"
            "SUMMARY: <one short sentence>\n\n"
            "Rules:\n"
            "- Use 1 for phishing.\n"
            "- Use 0 for benign.\n"
            "- Base your decision only on the email content."
        ),
    ),
}

def get_prompt_template(prompt_id: str) -> PromptTemplate:
    if prompt_id not in PROMPT_REGISTRY:
        known = ", ".join(sorted(PROMPT_REGISTRY))
        raise ValueError(
            f"Unsupported prompt_id: {prompt_id}\n"
            f"Known prompts: {known}"
        )
    return PROMPT_REGISTRY[prompt_id]


def list_prompt_ids() -> list[str]:
    return sorted(PROMPT_REGISTRY.keys())


def build_prompt(prompt_id: str, email_text: str) -> str:
    template = get_prompt_template(prompt_id)
    return (
        f"{template.prompt_text}\n\n"
        f"EMAIL TO ANALYSE:\n"
        f"{email_text}"
    )


def get_prompt_metadata(prompt_id: str) -> dict:
    template = get_prompt_template(prompt_id)
    return {
        "prompt_id": template.prompt_id,
        "prompt_description": template.description,
    }