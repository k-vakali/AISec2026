#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass

from model_adapters import BaseModelAdapter
from prompt_templates import build_prompt, get_prompt_metadata

@dataclass(frozen=True)
class ContextStats:
    prompt_id: str
    prompt_description: str
    prompt_tokens: int
    email_tokens: int
    input_tokens: int
    max_new_tokens: int
    total_requested_tokens: int
    context_limit: int
    fits: bool
    overflow_tokens: int


def count_text_tokens(adapter: BaseModelAdapter, text: str) -> int:
    """
    Count tokens for plain text only, without chat formatting.
    Useful for rough bookkeeping of raw email length.
    """
    encoded = adapter.tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return encoded["input_ids"].shape[1]


def analyse_context(
    adapter: BaseModelAdapter,
    prompt_id: str,
    email_text: str,
    max_new_tokens: int,
) -> ContextStats:
    """
    Compute token usage and context fit statistics for a single run.

    This function does not modify, truncate, or write anything.
    It only measures whether the requested run fits.
    """
    prompt_meta = get_prompt_metadata(prompt_id)
    full_prompt = build_prompt(prompt_id, email_text)

    prompt_body_only = build_prompt(prompt_id, "")

    prompt_tokens = adapter.count_tokens(prompt_body_only)
    email_tokens = count_text_tokens(adapter, email_text)
    input_tokens = adapter.count_tokens(full_prompt)

    total_requested_tokens = input_tokens + max_new_tokens
    context_limit = adapter.context_limit
    overflow_tokens = max(0, total_requested_tokens - context_limit)
    fits = overflow_tokens == 0

    return ContextStats(
        prompt_id=prompt_meta["prompt_id"],
        prompt_description=prompt_meta["prompt_description"],
        prompt_tokens=prompt_tokens,
        email_tokens=email_tokens,
        input_tokens=input_tokens,
        max_new_tokens=max_new_tokens,
        total_requested_tokens=total_requested_tokens,
        context_limit=context_limit,
        fits=fits,
        overflow_tokens=overflow_tokens,
    )


def context_stats_to_dict(stats: ContextStats) -> dict:
    return {
        "prompt_id": stats.prompt_id,
        "prompt_description": stats.prompt_description,
        "prompt_tokens": stats.prompt_tokens,
        "email_tokens": stats.email_tokens,
        "input_tokens": stats.input_tokens,
        "max_new_tokens": stats.max_new_tokens,
        "total_requested_tokens": stats.total_requested_tokens,
        "context_limit": stats.context_limit,
        "fits": stats.fits,
        "overflow_tokens": stats.overflow_tokens,
    }