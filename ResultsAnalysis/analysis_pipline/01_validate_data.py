
"""
01_validate_data.py

Validates the repeated-measures structure of an LLM phishing detection CSV.

Checks:
- required columns
- parser-independent base email IDs
- duplicate rows
- duplicate experimental cells
- expected rows per email
- missing model/prompt/parser combinations
- consistency of year, dataset, obfuscation, and ground truth
- year × dataset balance
- parsing/status failures
- prediction and correctness consistency

Usage:
    python 01_validate_data.py all_results_merged.csv --output-dir validation_out

Dependencies:
    pip install pandas numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
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
        description="Validate repeated-measures phishing experiment data."
    )
    parser.add_argument("csv_file", type=Path, help="Input CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_out"),
        help="Output directory",
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


def load_and_clean(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    df = df.copy()

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
    df["status_clean"] = (
        df["status"].astype("string").str.strip().str.lower()
    )

    df["parse_ok_bool"] = normalise_bool(df["parse_ok"])
    df["pred_label_num"] = pd.to_numeric(
        df["pred_label"], errors="coerce"
    )
    df["ground_truth_num"] = pd.to_numeric(
        df["ground_truth"], errors="coerce"
    )
    df["year_num"] = pd.to_numeric(
        df["year"], errors="coerce"
    ).astype("Int64")

    if "fits" in df.columns:
        df["fits_bool"] = normalise_bool(df["fits"])
    else:
        df["fits_bool"] = pd.Series(
            pd.NA, index=df.index, dtype="boolean"
        )

    if "input_tokens" in df.columns:
        df["input_tokens_num"] = pd.to_numeric(
            df["input_tokens"], errors="coerce"
        )

    if "overflow_tokens" in df.columns:
        df["overflow_tokens_num"] = pd.to_numeric(
            df["overflow_tokens"], errors="coerce"
        )

    df["parseable"] = (
        (df["status_clean"] == "ok")
        & (df["parse_ok_bool"] == True)
        & df["pred_label_num"].isin([0, 1])
    )

    df["detected_end_to_end"] = (
        (df["ground_truth_num"] == 1)
        & (df["status_clean"] == "ok")
        & (df["parse_ok_bool"] == True)
        & (df["pred_label_num"] == 1)
    ).astype(int)

    df["correct_reconstructed"] = np.where(
        df["pred_label_num"].isin([0, 1])
        & df["ground_truth_num"].isin([0, 1]),
        (
            df["pred_label_num"] == df["ground_truth_num"]
        ).astype(int),
        np.nan,
    )

    if "correct" in df.columns:
        df["correct_original_bool"] = normalise_bool(df["correct"])
    else:
        df["correct_original_bool"] = pd.Series(
            pd.NA, index=df.index, dtype="boolean"
        )

    return df


def validate(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = []

    report.append("PHISHING EXPERIMENT DATA VALIDATION")
    report.append("=" * 60)
    report.append(f"Rows: {len(df):,}")
    report.append(
        f"Unique underlying emails: {df['base_email_id'].nunique():,}"
    )
    report.append(f"Models: {df['model'].nunique()}")
    report.append(f"Prompts: {df['prompt'].nunique()}")
    report.append(f"Parsers: {df['parser'].nunique()}")
    report.append("")

    model_levels = sorted(df["model"].dropna().unique())
    prompt_levels = sorted(df["prompt"].dropna().unique())
    parser_levels = sorted(df["parser"].dropna().unique())

    report.append(f"Model levels: {model_levels}")
    report.append(f"Prompt levels: {prompt_levels}")
    report.append(f"Parser levels: {parser_levels}")
    report.append("")

    expected_rows_per_email = (
        len(model_levels) * len(prompt_levels) * len(parser_levels)
    )

    report.append(
        "Expected rows per email from observed factorial design: "
        f"{expected_rows_per_email}"
    )

    exact_duplicate_count = int(df.duplicated().sum())
    report.append(f"Exact duplicate rows: {exact_duplicate_count}")

    cell_columns = [
        "base_email_id",
        "model",
        "prompt",
        "parser",
    ]

    duplicate_cell_mask = df.duplicated(
        subset=cell_columns,
        keep=False,
    )

    report.append(
        "Rows in duplicate email/model/prompt/parser cells: "
        f"{int(duplicate_cell_mask.sum())}"
    )

    if duplicate_cell_mask.any():
        df.loc[duplicate_cell_mask].sort_values(
            cell_columns
        ).to_csv(
            output_dir / "duplicate_experimental_cells.csv",
            index=False,
        )

    rows_per_email = (
        df.groupby("base_email_id")
        .size()
        .rename("rows")
        .reset_index()
    )

    rows_per_email["expected_rows"] = expected_rows_per_email
    rows_per_email["complete"] = (
        rows_per_email["rows"] == expected_rows_per_email
    )

    rows_per_email.to_csv(
        output_dir / "rows_per_email.csv",
        index=False,
    )

    report.append("")
    report.append("Rows-per-email distribution:")

    for row_count, frequency in (
        rows_per_email["rows"].value_counts().sort_index().items()
    ):
        report.append(f"  {row_count} rows: {frequency} emails")

    incomplete_email_ids = rows_per_email.loc[
        ~rows_per_email["complete"],
        "base_email_id",
    ].tolist()

    report.append(
        f"Incomplete emails: {len(incomplete_email_ids)}"
    )

    expected_cells = {
        (model, prompt, parser)
        for model in model_levels
        for prompt in prompt_levels
        for parser in parser_levels
    }

    missing_cells = []

    for email_id in incomplete_email_ids:
        group = df[df["base_email_id"] == email_id]

        observed_cells = set(
            zip(
                group["model"],
                group["prompt"],
                group["parser"],
            )
        )

        for model, prompt, parser in sorted(
            expected_cells - observed_cells
        ):
            missing_cells.append(
                {
                    "base_email_id": email_id,
                    "model": model,
                    "prompt": prompt,
                    "parser": parser,
                }
            )

    if missing_cells:
        pd.DataFrame(missing_cells).to_csv(
            output_dir / "missing_experimental_cells.csv",
            index=False,
        )

    report.append("")
    report.append("EMAIL-LEVEL METADATA CONSISTENCY")
    report.append("-" * 60)

    metadata_columns = [
        "year_num",
        "dataset_clean",
        "obfuscation_clean",
        "ground_truth_num",
    ]

    inconsistencies = []

    for column in metadata_columns:
        counts = (
            df.groupby("base_email_id")[column]
            .nunique(dropna=False)
        )

        inconsistent_ids = counts[counts > 1].index.tolist()

        report.append(
            f"Inconsistent {column}: {len(inconsistent_ids)} emails"
        )

        for email_id in inconsistent_ids:
            values = (
                df.loc[
                    df["base_email_id"] == email_id,
                    column,
                ]
                .drop_duplicates()
                .astype(str)
                .tolist()
            )

            inconsistencies.append(
                {
                    "base_email_id": email_id,
                    "column": column,
                    "values": " | ".join(values),
                }
            )

    if inconsistencies:
        pd.DataFrame(inconsistencies).to_csv(
            output_dir / "metadata_inconsistencies.csv",
            index=False,
        )

    email_metadata = (
        df[
            [
                "base_email_id",
                "year_num",
                "dataset_clean",
                "obfuscation_clean",
            ]
        ]
        .drop_duplicates("base_email_id")
    )

    year_dataset_table = pd.crosstab(
        email_metadata["year_num"],
        email_metadata["dataset_clean"],
        margins=True,
    )

    year_dataset_table.to_csv(
        output_dir / "year_by_dataset_unique_emails.csv"
    )

    report.append("")
    report.append("YEAR × DATASET COUNTS BASED ON UNIQUE EMAILS")
    report.append("-" * 60)
    report.append(year_dataset_table.to_string())

    occupied = year_dataset_table.drop(
        index="All", errors="ignore"
    ).drop(
        columns="All", errors="ignore"
    )

    if occupied.shape[0] > 1 and occupied.shape[1] > 1:
        fully_crossed = bool((occupied > 0).all().all())
    else:
        fully_crossed = False

    report.append("")
    report.append(
        f"Year and dataset fully crossed: {fully_crossed}"
    )

    if not fully_crossed:
        report.append(
            "WARNING: year and dataset may be partially or fully confounded."
        )

    report.append("")
    report.append("OUTCOME AND FAILURE CHECKS")
    report.append("-" * 60)

    all_positive = bool(
        df["ground_truth_num"].dropna().eq(1).all()
    )
    report.append(
        f"All non-missing ground-truth labels equal 1: {all_positive}"
    )

    report.append(
        f"Missing predicted labels: {int(df['pred_label_num'].isna().sum())}"
    )
    report.append(
        f"Non-OK status rows: "
        f"{int((df['status_clean'] != 'ok').sum())}"
    )
    report.append(
        f"parse_ok=False rows: "
        f"{int((df['parse_ok_bool'] == False).sum())}"
    )
    report.append(
        f"Unparseable end-to-end rows: "
        f"{int((~df['parseable']).sum())}"
    )

    if "fits_bool" in df.columns:
        report.append(
            f"fits=False rows: "
            f"{int((df['fits_bool'] == False).sum())}"
        )

    if "overflow_tokens_num" in df.columns:
        report.append(
            f"Rows with overflow_tokens > 0: "
            f"{int((df['overflow_tokens_num'] > 0).sum())}"
        )

    status_table = pd.crosstab(
        df["status_clean"],
        df["parse_ok_bool"],
        dropna=False,
        margins=True,
    )

    status_table.to_csv(
        output_dir / "status_by_parse_ok.csv"
    )

    report.append("")
    report.append("Status × parse_ok:")
    report.append(status_table.to_string())

    if "correct" in df.columns:
        comparable = (
            df["correct_original_bool"].notna()
            & pd.Series(df["correct_reconstructed"]).notna()
        )

        mismatch_mask = (
            df.loc[
                comparable,
                "correct_original_bool",
            ].astype(int)
            != df.loc[
                comparable,
                "correct_reconstructed",
            ].astype(int)
        )

        mismatch_count = int(mismatch_mask.sum())

        report.append("")
        report.append(
            "Original correct versus reconstructed correctness mismatches: "
            f"{mismatch_count}"
        )

        if mismatch_count:
            mismatch_rows = df.loc[comparable].loc[mismatch_mask]
            mismatch_rows.to_csv(
                output_dir / "correctness_mismatches.csv",
                index=False,
            )

    # Save a compact cleaned sample, not the whole transformed dataset.
    sample_columns = [
        "email_id",
        "base_email_id",
        "parser",
        "model",
        "prompt",
        "year_num",
        "dataset_clean",
        "obfuscation_clean",
        "status_clean",
        "parse_ok_bool",
        "pred_label_num",
        "ground_truth_num",
        "parseable",
        "detected_end_to_end",
    ]

    available_sample_columns = [
        column
        for column in sample_columns
        if column in df.columns
    ]

    df[available_sample_columns].head(50).to_csv(
        output_dir / "cleaned_sample_first_50_rows.csv",
        index=False,
    )

    df.to_csv(
        output_dir / "validated_cleaned_data.csv",
        index=False,
    )

    report.append("")
    report.append("VALIDATION DECISION GUIDE")
    report.append("-" * 60)
    report.append(
        "Proceed if duplicate cells = 0, metadata inconsistencies = 0, "
        "and missing cells are understood."
    )
    report.append(
        "Do not estimate separate year and dataset effects if they are "
        "not sufficiently crossed."
    )
    report.append(
        "Use base_email_id as the cluster ID in later GEE/bootstrap analyses."
    )
    report.append(
        "Use detected_end_to_end as the primary recall outcome."
    )

    (output_dir / "validation_report.txt").write_text(
        "\\n".join(report),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {args.csv_file}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading data...")
    df = load_and_clean(args.csv_file)

    print("Running validation checks...")
    validate(df, args.output_dir)

    print("")
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"Input: {args.csv_file}")
    print(f"Output: {args.output_dir.resolve()}")
    print(
        f"Unique underlying emails: "
        f"{df['base_email_id'].nunique():,}"
    )
    print(f"Rows: {len(df):,}")
    print("")
    print("Open these files first:")
    print("  validation_report.txt")
    print("  cleaned_sample_first_50_rows.csv")
    print("  rows_per_email.csv")



    print("\nparse_ok values")
    print(df["parse_ok"].value_counts(dropna=False))

    print("\nRows where parse_ok != True")
    print(
        df[df["parse_ok"] != True][
            [
                "email_id",
                "model_name",
                "prompt_id",
                "parser_id",
                "parse_ok",
                "pred_label",
            ]
        ]
    )






if __name__ == "__main__":
    main()

