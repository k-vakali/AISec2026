import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


def calculate_stats(summary_file):
    total_emails = 0

    css_counts = Counter()
    strategy_counts = Counter()

    css_email_files = defaultdict(list)
    strategy_email_files = defaultdict(list)

    emails_with_any_css = 0
    emails_with_any_hidden = 0

    css_fields = [
        "css_display_none",
        "css_visibility_hidden",
        "css_color_transparent",
        "css_same_color",
        "css_low_contrast",
        "css_opacity_zero",
        "css_position_offset",
        "css_clipping",
        "css_filter",
        "css_tiny_font",
    ]

    strategy_fields = [
        "strategy_add_paragraph",
        "strategy_insert_word",
        "strategy_disrupt_word",
        "strategy_html_comment",
        "strategy_mso_conditional_comment",
    ]

    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            email_file = row.get("email_file")
            total_emails += 1

            has_css = False
            has_hidden = False

            # ---------------- CSS tricks ----------------

            for field in css_fields:
                if row.get(field, False):
                    css_counts[field] += 1
                    css_email_files[field].append(email_file)
                    has_css = True

            if has_css:
                emails_with_any_css += 1

            # ---------------- hidden-text techniques / strategies ----------------

            for field in strategy_fields:
                if row.get(field, False):
                    strategy_counts[field] += 1
                    strategy_email_files[field].append(email_file)
                    has_hidden = True

            if has_hidden:
                emails_with_any_hidden += 1

    # ---------------- output ----------------

    print("\n========== OVERALL STATS ==========\n")

    print(f"Total emails: {total_emails}")

    print("\n--- Emails with concealment ---")

    print(
        f"Any CSS trick: "
        f"{emails_with_any_css} "
        f"({emails_with_any_css / total_emails:.2%})"
    )

    print(
        f"Any hidden-text technique (inc comments): "
        f"{emails_with_any_hidden} "
        f"({emails_with_any_hidden / total_emails:.2%})"
    )

    print("\n--- CSS technique counts ---")

    for field in css_fields:
        v = css_counts[field]
        if v:
            print(f"{field}: {v} ({v / total_emails:.2%})")

    print("\n--- CSS technique email lists ---")

    for field in css_fields:
        emails = css_email_files[field]

        if emails:
            print(f"\n{field}: {len(emails)} emails")

            for email_file in emails:
                print(f"  {email_file}")

    print("\n--- Concealment subtype counts ---")

    for field in strategy_fields:
        v = strategy_counts[field]
        if v:
            print(f"{field}: {v} ({v / total_emails:.2%})")

    print("\n--- Concealment subtype email lists ---")

    for field in strategy_fields:
        emails = strategy_email_files[field]

        if emails:
            print(f"\n{field}: {len(emails)} emails")

            for email_file in emails:
                print(f"  {email_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate concealment statistics from pipeline summary."
    )

    parser.add_argument(
        "summary_file",
        help="Path to pipeline_summary.jsonl",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_file)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}"
        )

    calculate_stats(summary_path)