#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
import re

from context_utils import analyse_context, context_stats_to_dict
from model_adapters import create_adapter, list_supported_models
from prompt_templates import build_prompt, list_prompt_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one model over parsed email folders p1 and p2."
    )

    parser.add_argument("--model", required=True, help="Model ID from model registry")

    parser.add_argument(
        "--parsed-root",
        required=True,
        type=Path,
        help="Root directory containing parser subfolders, e.g. outputs/P1 and outputs/P2",
    )

    parser.add_argument(
        "--parsers",
        nargs="+",
        default=["P1", "P2"],
        help="Parser subfolders to run, e.g. P1 P2",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path",
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="One or more prompt IDs",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate",
    )

    parser.add_argument(
        "--allow-over-context",
        action="store_true",
        help="Attempt generation even when the prompt exceeds the model context window",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    supported_models = set(list_supported_models())
    if args.model not in supported_models:
        raise ValueError(
            f"Unsupported model: {args.model}\n"
            f"Supported models: {', '.join(sorted(supported_models))}"
        )

    supported_prompts = set(list_prompt_ids())
    bad_prompts = [p for p in args.prompts if p not in supported_prompts]
    if bad_prompts:
        raise ValueError(
            f"Unsupported prompt IDs: {', '.join(bad_prompts)}\n"
            f"Supported prompts: {', '.join(sorted(supported_prompts))}"
        )

    args.parsed_root = args.parsed_root.expanduser().resolve()

    if not args.parsed_root.exists():
        raise ValueError(f"Parsed root does not exist: {args.parsed_root}")

    if not args.parsed_root.is_dir():
        raise ValueError(f"Parsed root is not a directory: {args.parsed_root}")

    for parser_id in args.parsers:
        parser_dir = args.parsed_root / parser_id

        if not parser_dir.exists():
            raise ValueError(f"Parser folder does not exist: {parser_dir}")

        if not parser_dir.is_dir():
            raise ValueError(f"Parser path is not a directory: {parser_dir}")


def build_output_path(args: argparse.Namespace) -> Path:
    # model_safe = args.model.replace("/", "_")
    # prompts_part = "-".join(args.prompts)
    # parsers_part = "-".join(args.parsers)

    # filename = (
    #     f"{model_safe}_{parsers_part}_{prompts_part}_"
    #     f"t{args.max_new_tokens}.csv"
    # )
    input_name = args.parsed_root.name

    if input_name.endswith("_POut"):
        run_name = input_name.removesuffix("_POut") + "_LLMOut"
    else:
        run_name = input_name + "_LLMOut"

    model_short = (
        "llama"
        if "Llama" in args.model
        else "qwen"
    )

    filename = f"{run_name}_{model_short}.csv"


    return Path("results") / filename


def find_email_files(parsed_root: Path, parser_id: str) -> list[Path]:
    parser_dir = parsed_root / parser_id
    return sorted(parser_dir.glob("*.txt"))





def parse_binary_prediction(raw_output: str) -> tuple[bool, int | None]:
    """
    Robustly parse a binary phishing label from model output.

    Accepts examples:
    - 0
    - 1
    - LABEL: 0
    - LABEL: 1
    - LABEL: <0>
    - LABEL: <1>
    - LABEL = 1
    - Label - 0
    - classification: 1
    """

    text = raw_output.strip()

    if not text:
        return False, None

    # Exact single-token output
    if text in {"0", "1"}:
        return True, int(text)

    # Prefer explicit label/classification lines
    for line in text.splitlines():
        clean = line.strip()

        match = re.search(
            r"\b(label|classification|class|prediction)\b\s*[:=\-]?\s*[<\[\(\{]?\s*([01])\s*[>\]\)\}]?",
            clean,
            flags=re.IGNORECASE,
        )

        if match:
            return True, int(match.group(2))

    # Fallback: if the first non-empty line contains a single 0 or 1
    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        "",
    )

    match = re.fullmatch(r"[<\[\(\{]?\s*([01])\s*[>\]\)\}]?", first_line)
    if match:
        return True, int(match.group(1))

    return False, None

def build_row_base(
    email_path: Path,
    parser_id: str,
    model: str,
    prompt_id: str,
) -> dict:
    return {
        "email_id": email_path.stem,
        "email_path": str(email_path),
        "parser_id": parser_id,
        "model_name": model,
        "prompt_id": prompt_id,
    }


def main() -> None:
    args = parse_args()

    try:
        validate_args(args)
    except Exception as e:
        print(f"Argument error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output_csv is None:
        args.output_csv = build_output_path(args)

    args.output_csv = args.output_csv.expanduser().resolve()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    adapter = create_adapter(args.model)
    adapter.load()
    model_meta = adapter.get_metadata()

    print(f"Output will be written to: {args.output_csv}")

    fieldnames = [
        "email_id",
       # "email_path",
        "parser_id",
        "model_name",
       # "family",
        "context_limit",
        "prompt_id",
       # "prompt_description",
       # "prompt_tokens",
       # "email_tokens",
        "input_tokens",
      #  "max_new_tokens",
      #  "total_requested_tokens",
        "fits",
        "overflow_tokens",
        "status",
      #  "runtime_seconds",
        "raw_output",
        "parse_ok",
        "pred_label",
    ]

    row_count = 0

    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore",
        )
        writer.writeheader()

        for parser_id in args.parsers:
            files = find_email_files(args.parsed_root, parser_id)

            if not files:
                print(f"[WARN] No .txt files found in {args.parsed_root / parser_id}")
                continue

            for email_path in files:
                email_text = email_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )

                for prompt_id in args.prompts:
                    base = build_row_base(
                        email_path=email_path,
                        parser_id=parser_id,
                        model=args.model,
                        prompt_id=prompt_id,
                    )

                    stats = analyse_context(
                        adapter=adapter,
                        prompt_id=prompt_id,
                        email_text=email_text,
                        max_new_tokens=args.max_new_tokens,
                    )

                    row = {
                        **base,
                        "family": model_meta["family"],
                        "context_limit": model_meta["context_limit"],
                        **context_stats_to_dict(stats),
                        "status": "",
                        "runtime_seconds": "",
                        "raw_output": "",
                        "parse_ok": False,
                        "pred_label": "",
                    }

                    if not stats.fits and not args.allow_over_context:
                        row["status"] = "skipped_over_context"

                        writer.writerow(row)
                        row_count += 1

                        print(
                            f"[SKIP] {parser_id} | {email_path.name} | {prompt_id} | "
                            f"overflow_tokens={stats.overflow_tokens}"
                        )

                        continue

                    full_prompt = build_prompt(prompt_id, email_text)

                    start = time.perf_counter()

                    try:
                        raw_output = adapter.generate(
                            prompt=full_prompt,
                            max_new_tokens=args.max_new_tokens,
                        )

                        runtime_seconds = round(time.perf_counter() - start, 3)
                        parse_ok, pred_label = parse_binary_prediction(raw_output)

                        row["status"] = "ok" if stats.fits else "ok_over_context"
                        row["runtime_seconds"] = runtime_seconds
                        row["raw_output"] = raw_output
                        row["parse_ok"] = parse_ok
                        row["pred_label"] = pred_label if pred_label is not None else ""

                        writer.writerow(row)
                        row_count += 1

                        print(
                            f"[OK] {parser_id} | {email_path.name} | {prompt_id} | "
                            f"fits={stats.fits} | parse_ok={parse_ok}"
                        )

                    except Exception as e:
                        runtime_seconds = round(time.perf_counter() - start, 3)

                        row["status"] = (
                            f"generation_error:{type(e).__name__}"
                            if stats.fits
                            else f"generation_error_over_context:{type(e).__name__}"
                        )

                        row["runtime_seconds"] = runtime_seconds
                        row["raw_output"] = str(e)

                        writer.writerow(row)
                        row_count += 1

                        print(
                            f"[ERR] {parser_id} | {email_path.name} | {prompt_id} | "
                            f"fits={stats.fits} | {type(e).__name__}: {e}"
                        )

    print(f"Done. Wrote {row_count} rows to {args.output_csv}")


if __name__ == "__main__":
    main()