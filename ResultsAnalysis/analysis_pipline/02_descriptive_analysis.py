
"""
02_descriptive_analysis.py

Fast descriptive analysis for repeated-measures LLM phishing detection results.

This script does NOT make inferential claims. It identifies observed patterns
worth testing later with clustered bootstrap and GEE.

Outputs:
- main-effect recall summaries
- all two-way summaries
- selected three-way summaries
- full 12-configuration ranking
- observed pairwise differences
- subgroup effect ranges
- automatically flagged patterns
- end-to-end and parseable-only summaries

Usage:
    python 02_descriptive_analysis.py validated_cleaned_data.csv \\
        --output-dir 02_descriptive_out

The input may be either:
1. the original merged CSV, or
2. validated_cleaned_data.csv produced by 01_validate_data.py

Dependencies:
    pip install pandas numpy
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_ORIGINAL_COLUMNS = {
    "email_id",
    "parser_id",
    "model_name",
    "prompt_id",
    "status",
    "parse_ok",
    "pred_label",
    "year",
    "dataset",
    "obfuscation",
    "ground_truth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate descriptive summaries for phishing results."
    )
    parser.add_argument("csv_file", type=Path, help="Input CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("02_descriptive_out"),
        help="Output directory",
    )
    parser.add_argument(
        "--flag-threshold",
        type=float,
        default=5.0,
        help=(
            "Absolute percentage-point threshold used to flag observed "
            "differences worth later testing"
        ),
    )
    return parser.parse_args()


def normalise_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(mapping)
        .astype("boolean")
    )


def short_model_name(value: object) -> str:
    text = str(value).strip()
    lower = text.lower()

    if "llama" in lower:
        return "Llama"
    if "qwen" in lower:
        return "Qwen"

    return text.split("/")[-1]


def prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()

    # Case 1: validated cleaned input from Script 1.
    cleaned_columns = {
        "base_email_id",
        "model",
        "parser",
        "prompt",
        "year_num",
        "dataset_clean",
        "obfuscation_clean",
        "detected_end_to_end",
    }

    if cleaned_columns.issubset(df.columns):
        df["model"] = df["model"].astype("string")
        df["parser"] = (
            df["parser"].astype("string").str.strip().str.upper()
        )
        df["prompt"] = (
            df["prompt"].astype("string").str.strip().str.lower()
        )
        df["dataset_clean"] = (
            df["dataset_clean"].astype("string").str.strip().str.lower()
        )
        df["obfuscation_clean"] = (
            df["obfuscation_clean"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        df["year_num"] = pd.to_numeric(
            df["year_num"], errors="coerce"
        ).astype("Int64")
        df["detected"] = pd.to_numeric(
            df["detected_end_to_end"], errors="coerce"
        )

        if "parseable" in df.columns:
            df["parseable_bool"] = normalise_bool(df["parseable"])
        else:
            df["parseable_bool"] = True

        if "correct_reconstructed" in df.columns:
            df["correct_parseable"] = pd.to_numeric(
                df["correct_reconstructed"], errors="coerce"
            )
        elif "correct_parseable" in df.columns:
            df["correct_parseable"] = pd.to_numeric(
                df["correct_parseable"], errors="coerce"
            )
        else:
            df["correct_parseable"] = np.where(
                df["parseable_bool"],
                df["detected"],
                np.nan,
            )

    # Case 2: original merged CSV.
    else:
        missing = REQUIRED_ORIGINAL_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                "Input is neither the validated cleaned file nor the "
                "original expected CSV. Missing columns: "
                + ", ".join(sorted(missing))
            )

        df["base_email_id"] = (
            df["email_id"]
            .astype("string")
            .str.replace(r"_P(?:1|2)$", "", regex=True)
        )
        df["model"] = df["model_name"].map(short_model_name)
        df["parser"] = (
            df["parser_id"].astype("string").str.strip().str.upper()
        )
        df["prompt"] = (
            df["prompt_id"].astype("string").str.strip().str.lower()
        )
        df["dataset_clean"] = (
            df["dataset"].astype("string").str.strip().str.lower()
        )
        df["obfuscation_clean"] = (
            df["obfuscation"].astype("string").str.strip().str.lower()
        )
        df["year_num"] = pd.to_numeric(
            df["year"], errors="coerce"
        ).astype("Int64")

        status = (
            df["status"].astype("string").str.strip().str.lower()
        )
        parse_ok = normalise_bool(df["parse_ok"])
        pred = pd.to_numeric(df["pred_label"], errors="coerce")
        truth = pd.to_numeric(df["ground_truth"], errors="coerce")

        df["parseable_bool"] = (
            (status == "ok")
            & (parse_ok == True)
            & pred.isin([0, 1])
        )

        df["detected"] = (
            (truth == 1)
            & (status == "ok")
            & (parse_ok == True)
            & (pred == 1)
        ).astype(int)

        df["correct_parseable"] = np.where(
            df["parseable_bool"],
            (pred == truth).astype(int),
            np.nan,
        )

    df["configuration"] = (
        df["model"].astype(str)
        + " | "
        + df["prompt"].astype(str)
        + " | "
        + df["parser"].astype(str)
    )

    df["collection"] = (
        df["dataset_clean"].astype(str)
        + "_"
        + df["year_num"].astype(str)
    )

    return df


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """
    Wilson interval for a single observed proportion.

    Important: when rows are repeated observations of the same email,
    this interval is descriptive only and not the final inferential CI.
    """
    if total <= 0:
        return np.nan, np.nan

    z = 1.959963984540054
    p = successes / total
    denominator = 1 + (z**2 / total)
    centre = p + (z**2 / (2 * total))
    adjustment = z * np.sqrt(
        (p * (1 - p) / total)
        + (z**2 / (4 * total**2))
    )

    low = (centre - adjustment) / denominator
    high = (centre + adjustment) / denominator
    return float(low), float(high)


def summarise(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    outcome: str = "detected",
) -> pd.DataFrame:
    result = (
        df.groupby(list(group_cols), dropna=False, observed=True)
        .agg(
            successes=(outcome, "sum"),
            observations=(outcome, "count"),
            recall=(outcome, "mean"),
            unique_emails=("base_email_id", "nunique"),
        )
        .reset_index()
    )

    intervals = [
        wilson_interval(int(row.successes), int(row.observations))
        for row in result.itertuples()
    ]

    result["wilson_low"] = [x[0] for x in intervals]
    result["wilson_high"] = [x[1] for x in intervals]

    for col in ["recall", "wilson_low", "wilson_high"]:
        result[f"{col}_pct"] = 100 * result[col]

    return result


def pairwise_observed_differences(
    summary: pd.DataFrame,
    factor: str,
    context_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    context_cols = list(context_cols or [])
    rows = []

    if context_cols:
        grouped = summary.groupby(
            context_cols,
            dropna=False,
            observed=True,
        )
    else:
        grouped = [((), summary)]

    for context_values, group in grouped:
        if not isinstance(context_values, tuple):
            context_values = (context_values,)

        levels = sorted(group[factor].dropna().unique().tolist())

        for level_a, level_b in itertools.combinations(levels, 2):
            row_a = group[group[factor] == level_a]
            row_b = group[group[factor] == level_b]

            if len(row_a) != 1 or len(row_b) != 1:
                continue

            recall_a = float(row_a["recall"].iloc[0])
            recall_b = float(row_b["recall"].iloc[0])
            difference = recall_a - recall_b

            record = {
                "factor": factor,
                "level_a": level_a,
                "level_b": level_b,
                "recall_a": recall_a,
                "recall_b": recall_b,
                "recall_a_pct": 100 * recall_a,
                "recall_b_pct": 100 * recall_b,
                "difference": difference,
                "difference_percentage_points": 100 * difference,
                "absolute_difference_percentage_points": (
                    100 * abs(difference)
                ),
                "relative_change_vs_b_pct": (
                    100 * difference / recall_b
                    if recall_b != 0
                    else np.nan
                ),
            }

            for column, value in zip(context_cols, context_values):
                record[column] = value

            rows.append(record)

    return pd.DataFrame(rows)


def factor_range_table(
    summary: pd.DataFrame,
    factor: str,
    context_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    context_cols = list(context_cols or [])

    if context_cols:
        result = (
            summary.groupby(context_cols, dropna=False, observed=True)
            .agg(
                minimum_recall=("recall", "min"),
                maximum_recall=("recall", "max"),
                recall_range=("recall", lambda x: x.max() - x.min()),
                minimum_level=(
                    factor,
                    lambda x: summary.loc[x.index, "recall"].idxmin(),
                ),
            )
            .reset_index()
        )

        # Replace index-based minimum/maximum level calculation safely.
        records = []

        for context_values, group in summary.groupby(
            context_cols,
            dropna=False,
            observed=True,
        ):
            if not isinstance(context_values, tuple):
                context_values = (context_values,)

            min_row = group.loc[group["recall"].idxmin()]
            max_row = group.loc[group["recall"].idxmax()]

            record = {
                "factor": factor,
                "minimum_level": min_row[factor],
                "maximum_level": max_row[factor],
                "minimum_recall": min_row["recall"],
                "maximum_recall": max_row["recall"],
                "recall_range": (
                    max_row["recall"] - min_row["recall"]
                ),
            }

            for column, value in zip(context_cols, context_values):
                record[column] = value

            records.append(record)

        result = pd.DataFrame(records)

    else:
        min_row = summary.loc[summary["recall"].idxmin()]
        max_row = summary.loc[summary["recall"].idxmax()]

        result = pd.DataFrame(
            [
                {
                    "factor": factor,
                    "minimum_level": min_row[factor],
                    "maximum_level": max_row[factor],
                    "minimum_recall": min_row["recall"],
                    "maximum_recall": max_row["recall"],
                    "recall_range": (
                        max_row["recall"] - min_row["recall"]
                    ),
                }
            ]
        )

    result["minimum_recall_pct"] = 100 * result["minimum_recall"]
    result["maximum_recall_pct"] = 100 * result["maximum_recall"]
    result["recall_range_percentage_points"] = (
        100 * result["recall_range"]
    )

    return result


def generate_all_tables(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    main_factors = [
        "model",
        "prompt",
        "parser",
        "year_num",
        "dataset_clean",
        "obfuscation_clean",
    ]

    # Main effects.
    for factor in main_factors:
        name = f"main_{factor}"
        tables[name] = summarise(df, [factor])
        tables[name].to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

    # All two-way tables.
    for factor_a, factor_b in itertools.combinations(
        main_factors,
        2,
    ):
        name = f"two_way_{factor_a}_X_{factor_b}"
        tables[name] = summarise(
            df,
            [factor_a, factor_b],
        )
        tables[name].to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

    # Selected three-way tables.
    three_way_sets = [
        ["model", "prompt", "parser"],
        ["model", "parser", "obfuscation_clean"],
        ["model", "prompt", "year_num"],
        ["prompt", "parser", "obfuscation_clean"],
        ["model", "year_num", "dataset_clean"],
        ["parser", "year_num", "dataset_clean"],
    ]

    for factors in three_way_sets:
        name = "three_way_" + "_X_".join(factors)
        tables[name] = summarise(df, factors)
        tables[name].to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

    # Complete configuration ranking.
    config = summarise(
        df,
        ["model", "prompt", "parser"],
    ).sort_values(
        ["recall", "model", "prompt", "parser"],
        ascending=[False, True, True, True],
    )

    config.insert(
        0,
        "observed_rank",
        np.arange(1, len(config) + 1),
    )

    config.to_csv(
        output_dir / "configuration_ranking.csv",
        index=False,
    )
    tables["configuration_ranking"] = config

    # Parseable-only sensitivity summaries.
    parseable = df[df["parseable_bool"] == True].copy()

    if not parseable.empty:
        for factor in main_factors:
            table = summarise(
                parseable,
                [factor],
                outcome="correct_parseable",
            )
            table.to_csv(
                output_dir / f"parseable_only_main_{factor}.csv",
                index=False,
            )

        parseable_config = summarise(
            parseable,
            ["model", "prompt", "parser"],
            outcome="correct_parseable",
        ).sort_values(
            "recall",
            ascending=False,
        )

        parseable_config.to_csv(
            output_dir / "parseable_only_configuration_ranking.csv",
            index=False,
        )

    return tables


def generate_observed_contrasts(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    main_factors = [
        "model",
        "prompt",
        "parser",
        "year_num",
        "dataset_clean",
        "obfuscation_clean",
    ]

    overall_contrasts = []
    subgroup_contrasts = []
    range_tables = []

    # Overall contrasts.
    for factor in main_factors:
        summary = summarise(df, [factor])
        contrast = pairwise_observed_differences(
            summary,
            factor,
        )
        contrast["analysis_scope"] = "overall"
        overall_contrasts.append(contrast)

        range_table = factor_range_table(
            summary,
            factor,
        )
        range_table["analysis_scope"] = "overall"
        range_tables.append(range_table)

    # Simple effects within every other factor.
    for target_factor, moderator_factor in itertools.permutations(
        main_factors,
        2,
    ):
        summary = summarise(
            df,
            [moderator_factor, target_factor],
        )

        contrasts = pairwise_observed_differences(
            summary,
            target_factor,
            context_cols=[moderator_factor],
        )

        if not contrasts.empty:
            contrasts["analysis_scope"] = (
                f"{target_factor}_within_{moderator_factor}"
            )
            contrasts["moderator_factor"] = moderator_factor
            subgroup_contrasts.append(contrasts)

        ranges = factor_range_table(
            summary,
            target_factor,
            context_cols=[moderator_factor],
        )

        if not ranges.empty:
            ranges["analysis_scope"] = (
                f"{target_factor}_within_{moderator_factor}"
            )
            ranges["moderator_factor"] = moderator_factor
            range_tables.append(ranges)

    overall_df = (
        pd.concat(overall_contrasts, ignore_index=True)
        if overall_contrasts
        else pd.DataFrame()
    )

    subgroup_df = (
        pd.concat(subgroup_contrasts, ignore_index=True, sort=False)
        if subgroup_contrasts
        else pd.DataFrame()
    )

    ranges_df = (
        pd.concat(range_tables, ignore_index=True, sort=False)
        if range_tables
        else pd.DataFrame()
    )

    overall_df.to_csv(
        output_dir / "observed_overall_contrasts.csv",
        index=False,
    )

    subgroup_df.to_csv(
        output_dir / "observed_subgroup_contrasts.csv",
        index=False,
    )

    ranges_df.to_csv(
        output_dir / "observed_effect_ranges.csv",
        index=False,
    )

    return overall_df, subgroup_df


def flag_patterns(
    overall: pd.DataFrame,
    subgroup: pd.DataFrame,
    threshold_pp: float,
    output_dir: Path,
) -> pd.DataFrame:
    frames = []

    if not overall.empty:
        temp = overall.copy()
        temp["flag_reason"] = np.where(
            temp["absolute_difference_percentage_points"] >= threshold_pp,
            f"absolute observed difference >= {threshold_pp:.1f} pp",
            "",
        )
        frames.append(temp)

    if not subgroup.empty:
        temp = subgroup.copy()
        temp["flag_reason"] = np.where(
            temp["absolute_difference_percentage_points"] >= threshold_pp,
            f"absolute observed subgroup difference >= {threshold_pp:.1f} pp",
            "",
        )
        frames.append(temp)

    combined = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame()
    )

    if combined.empty:
        combined.to_csv(
            output_dir / "patterns_worth_testing.csv",
            index=False,
        )
        return combined

    flagged = combined[
        combined["absolute_difference_percentage_points"]
        >= threshold_pp
    ].copy()

    flagged = flagged.sort_values(
        "absolute_difference_percentage_points",
        ascending=False,
    )

    flagged.to_csv(
        output_dir / "patterns_worth_testing.csv",
        index=False,
    )

    return flagged


def write_summary_report(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    overall: pd.DataFrame,
    subgroup: pd.DataFrame,
    flagged: pd.DataFrame,
    output_dir: Path,
    threshold_pp: float,
) -> None:
    lines: list[str] = []

    lines.append("DESCRIPTIVE ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Rows: {len(df):,}")
    lines.append(
        f"Unique underlying emails: {df['base_email_id'].nunique():,}"
    )
    lines.append("")

    lines.append("IMPORTANT")
    lines.append("-" * 70)
    lines.append(
        "These are observed descriptive differences. They are not yet "
        "cluster-adjusted statistical tests."
    )
    lines.append(
        "Wilson intervals in the CSV files are descriptive only when "
        "summaries pool repeated rows from the same email."
    )
    lines.append("")

    lines.append("MAIN EFFECTS")
    lines.append("-" * 70)

    for key in [
        "main_model",
        "main_prompt",
        "main_parser",
        "main_year_num",
        "main_dataset_clean",
        "main_obfuscation_clean",
    ]:
        table = tables[key]
        factor = key.replace("main_", "")

        lines.append("")
        lines.append(factor.upper())

        display_cols = [
            factor,
            "successes",
            "observations",
            "recall_pct",
        ]

        lines.append(
            table[display_cols].to_string(
                index=False,
                formatters={
                    "recall_pct": lambda x: f"{x:.2f}",
                },
            )
        )

    lines.append("")
    lines.append("TOP CONFIGURATIONS")
    lines.append("-" * 70)

    config = tables["configuration_ranking"].head(12)

    lines.append(
        config[
            [
                "observed_rank",
                "model",
                "prompt",
                "parser",
                "successes",
                "observations",
                "recall_pct",
            ]
        ].to_string(
            index=False,
            formatters={
                "recall_pct": lambda x: f"{x:.2f}",
            },
        )
    )

    lines.append("")
    lines.append(
        f"OBSERVED PATTERNS FLAGGED AT >= {threshold_pp:.1f} PP"
    )
    lines.append("-" * 70)

    if flagged.empty:
        lines.append("No observed contrasts exceeded the threshold.")
    else:
        top = flagged.head(30)

        columns = [
            "analysis_scope",
            "factor",
            "level_a",
            "level_b",
            "difference_percentage_points",
        ]

        optional_columns = [
            col
            for col in [
                "model",
                "prompt",
                "parser",
                "year_num",
                "dataset_clean",
                "obfuscation_clean",
            ]
            if col in top.columns
        ]

        columns.extend(optional_columns)

        lines.append(
            top[columns].to_string(
                index=False,
                formatters={
                    "difference_percentage_points": (
                        lambda x: f"{x:.2f}"
                    ),
                },
            )
        )

    lines.append("")
    lines.append("NEXT STEP")
    lines.append("-" * 70)
    lines.append(
        "Use patterns_worth_testing.csv to choose effects for targeted "
        "McNemar, GEE, and email-cluster bootstrap analyses."
    )
    lines.append(
        "Do not select findings using p-values at this stage; select them "
        "using substantive size, consistency, and relevance."
    )

    (output_dir / "descriptive_summary.txt").write_text(
        "\\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {args.csv_file}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = prepare_data(args.csv_file)

    print("Generating descriptive tables...")
    tables = generate_all_tables(df, args.output_dir)

    print("Calculating observed contrasts and ranges...")
    overall, subgroup = generate_observed_contrasts(
        df,
        args.output_dir,
    )

    print("Flagging patterns worth later testing...")
    flagged = flag_patterns(
        overall,
        subgroup,
        threshold_pp=args.flag_threshold,
        output_dir=args.output_dir,
    )

    write_summary_report(
        df,
        tables,
        overall,
        subgroup,
        flagged,
        args.output_dir,
        args.flag_threshold,
    )

    print("")
    print("=" * 70)
    print("DESCRIPTIVE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Input: {args.csv_file}")
    print(f"Output: {args.output_dir.resolve()}")
    print("")
    print("Open these files first:")
    print("  descriptive_summary.txt")
    print("  configuration_ranking.csv")
    print("  patterns_worth_testing.csv")
    print("  observed_effect_ranges.csv")
    print("")
    print(
        "Reminder: these are descriptive results, not final "
        "cluster-adjusted statistical tests."
    )


if __name__ == "__main__":
    main()
