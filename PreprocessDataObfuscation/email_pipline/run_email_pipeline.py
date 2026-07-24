import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from extract_html_from_eml import extract_html_from_eml
from css_detect import detect_invisible_html
from disruption_type import classify_jsonl_file


def hidden_type_counts(classified_rows):
    css_hidden = 0
    html_comment = 0
    mso_comment = 0

    for row in classified_rows:
        row_type = row.get("type")
        strategy = row.get("embedding_strategy")

        if strategy == "html_comment":
            html_comment += 1
        elif strategy == "mso_conditional_comment":
            mso_comment += 1
        elif row_type == "html_comment":
            html_comment += 1
        elif row_type == "conditional_comment":
            mso_comment += 1
        else:
            css_hidden += 1

    return {
        "css_hidden_fragment_count": css_hidden,
        "html_comment_count": html_comment,
        "mso_conditional_comment_count": mso_comment,
        "has_css_hidden_text": css_hidden > 0,
        "has_html_comments": html_comment > 0,
        "has_mso_conditional_comments": mso_comment > 0,
    }

def reason_flags(hidden_rows):
    reasons = []

    for row in hidden_rows:
        reasons.extend(row.get("hidden_reasons", []))

    joined = " | ".join(reasons).lower()

    return {
        "css_display_none": "display:none" in joined,
        "css_visibility_hidden": "visibility:hidden" in joined or "visibility:collapse" in joined,
        "css_color_transparent": "color:transparent" in joined,
        "css_same_color": "same_color" in joined,
        "css_low_contrast": "low_contrast" in joined,
        "css_opacity_zero": "opacity" in joined,
        "css_position_offset": "position_offset" in joined,
        "css_clipping": "clipping" in joined,
        "css_filter": "filter" in joined,
        "css_tiny_font": "font too small" in joined,
    }


def strategy_flags(counts):
    return {
        "strategy_add_paragraph": counts.get("add_paragraph", 0) > 0,
        "strategy_insert_word": counts.get("insert_word", 0) > 0,
        "strategy_disrupt_word": counts.get("disrupt_word", 0) > 0,
        #"strategy_paragraph_disruption": counts.get("paragraph_disruption", 0) > 0,        
        "strategy_html_comment": counts.get("html_comment", 0) > 0,
        "strategy_mso_conditional_comment": counts.get("mso_conditional_comment", 0) > 0,
        "has_hidden_whitespace_only": counts.get("hidden_whitespace_only", 0) > 0
    }

def strategy_counts(counts):
    return {
        "count_add_paragraph": counts.get("add_paragraph", 0),
        "count_insert_word": counts.get("insert_word", 0),
        "count_disrupt_word": counts.get("disrupt_word", 0),
       # "count_paragraph_disruption": counts.get("paragraph_disruption", 0),
        "count_html_comment": counts.get("html_comment", 0),
        "count_mso_conditional_comment": counts.get("mso_conditional_comment", 0),
        "hidden_whitespace_only_count": counts.get("hidden_whitespace_only", 0),
    }

def reason_counts(hidden_rows):
    counts = {
        "count_css_display_none": 0,
        "count_css_visibility_hidden": 0,
        "count_css_color_transparent": 0,
        "count_css_same_color": 0,
        "count_css_low_contrast": 0,
        "count_css_opacity_zero": 0,
        "count_css_position_offset": 0,
        "count_css_clipping": 0,
        "count_css_filter": 0,
        "count_css_tiny_font": 0,
    }

    for row in hidden_rows:
        joined = " | ".join(row.get("hidden_reasons", [])).lower()

        if "display:none" in joined:
            counts["count_css_display_none"] += 1

        if "visibility:hidden" in joined or "visibility:collapse" in joined:
            counts["count_css_visibility_hidden"] += 1

        if "color:transparent" in joined:
            counts["count_css_color_transparent"] += 1

        if "same_color" in joined:
            counts["count_css_same_color"] += 1

        if "low_contrast" in joined:
            counts["count_css_low_contrast"] += 1

        if "opacity" in joined:
            counts["count_css_opacity_zero"] += 1

        if "position_offset" in joined:
            counts["count_css_position_offset"] += 1

        if "clipping" in joined:
            counts["count_css_clipping"] += 1

        if "filter" in joined:
            counts["count_css_filter"] += 1

        if "font too small" in joined:
            counts["count_css_tiny_font"] += 1

    return counts


def process_email(eml_path, input_dir, output_dir):
    rel_path = eml_path.relative_to(input_dir)
    safe_stem = "_".join(rel_path.with_suffix("").parts)

    html_dir = output_dir / "extracted_html"
    css_json_dir = output_dir / "css_json"
    classified_dir = output_dir / "classified_json"

    html_path = html_dir / f"{safe_stem}.html"
    css_json_path = css_json_dir / f"{safe_stem}.jsonl"
    classified_path = classified_dir / f"{safe_stem}_classified.jsonl"

    html_dir.mkdir(parents=True, exist_ok=True)
    css_json_dir.mkdir(parents=True, exist_ok=True)
    classified_dir.mkdir(parents=True, exist_ok=True)

    has_html = extract_html_from_eml(
        str(eml_path),
        output_dir=str(html_dir)
    )

    # Important:
    # Your extractor currently writes basename.html.
    # So we locate the file it produced.
    produced_html_path = html_dir / f"{eml_path.stem}.html"

    if produced_html_path.exists() and produced_html_path != html_path:

        if html_path.exists():
            html_path.unlink()

    produced_html_path.replace(html_path)
    if not has_html or not html_path.exists():
        return {
            "email_file": str(rel_path),
            "has_html": False,
            "css_display_none": False,
            "css_visibility_hidden": False,
            "css_color_transparent": False,
            "css_same_color": False,
            "css_low_contrast": False,
            "css_opacity_zero": False,
            "css_position_offset": False,
            "css_clipping": False,
            "css_filter": False,
            "css_tiny_font": False,
            "strategy_add_paragraph": False,
            "strategy_insert_word": False,
            "strategy_disrupt_word": False,
            "strategy_html_comment": False,
            "strategy_mso_conditional_comment": False,
            "hidden_fragment_count": 0,
            "visible_fragment_count": 0,
            "total_fragment_count": 0,
            "error": None,
        }

    css_rows, file_visible = detect_invisible_html(
        str(html_path),
        str(css_json_path)
    )

    classified_rows, class_summary = classify_jsonl_file(
        str(css_json_path),
        str(classified_path)
    )

    counts = class_summary["counts"]
    stats = class_summary["stats"]

    type_counts = hidden_type_counts(classified_rows)

    summary = {
        "email_file": str(rel_path),
        "has_html": True,
        **reason_flags(classified_rows),
        #**reason_counts(classified_rows),
        **strategy_flags(counts),
        #**strategy_counts(counts),
        **type_counts,
        "hidden_fragment_count": stats["hidden_fragments"],
        "visible_fragment_count": stats["visible_fragments"],
        "total_fragment_count": stats["total_fragments"],
        "dominant_strategy": (
            "no_hidden_text"
            if stats["hidden_fragments"] == 0
            else max(counts, key=counts.get)
        ),
        "error": None,
    }

    return summary


def run_pipeline(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Latest file (always overwritten)
    latest_summary_path = output_dir / "pipeline_summary.jsonl"

    # Historical file (never overwritten)
    history_dir = output_dir / "history"
    history_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    history_summary_path = (
        history_dir / f"pipeline_summary_{timestamp}.jsonl"
    )

    # if input_path.is_file():
    #     if input_path.suffix.lower() != ".eml":
    #         raise ValueError(f"Input file must be .eml: {input_path}")

    #     eml_files = [input_path]
    #     input_base = input_path.parent

    # else:
    #     eml_files = sorted(input_path.rglob("*.eml"))
    #     input_base = input_path

    EMAIL_SUFFIXES = {".eml", ".txt"}

    if input_path.is_file():
        if input_path.suffix.lower() not in EMAIL_SUFFIXES:
            raise ValueError(f"Input file must be .eml or .txt: {input_path}")

        eml_files = [input_path]
        input_base = input_path.parent

    else:
        eml_files = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in EMAIL_SUFFIXES
        )
        input_base = input_path


    with open(latest_summary_path, "w", encoding="utf-8") as latest_file, \
         open(history_summary_path, "w", encoding="utf-8") as history_file:

        for eml_path in tqdm(eml_files, desc="Processing emails"):
            try:
                summary = process_email(eml_path, input_base, output_dir)
            except Exception as e:
                summary = {
                    "email_file": str(eml_path.relative_to(input_base)),
                    "has_html": False,
                    "hidden_fragment_count": 0,
                    "visible_fragment_count": 0,
                    "total_fragment_count": 0,
                    "error": str(e),
                }

            line = json.dumps(summary, ensure_ascii=False) + "\n"

            latest_file.write(line)
            history_file.write(line)




def main():
    parser = argparse.ArgumentParser(
        description="Run phishing email invisible-text pipeline."
    )

    parser.add_argument(
        "input_path",
        help="Specific .eml file or folder of .eml files to process"
    )

    parser.add_argument(
        "--output-root",
        default="pipeline_outputs",
        help="Top-level output folder"
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_root = Path(args.output_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    output_dir = output_root / f"{input_path.stem}_out"

    run_pipeline(input_path, output_dir)


if __name__ == "__main__":
    main()