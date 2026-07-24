
"""
04_build_paper_tables.py

Builds complete paper-ready result tables from the outputs of the previous
analysis scripts.

Expected inputs
---------------
01_validation_out/
    validation_report.txt
    validated_cleaned_data.csv

02_descriptive_out/
    configuration_ranking.csv
    main_model.csv
    main_prompt.csv
    main_parser.csv
    main_year_num.csv
    main_dataset_clean.csv
    main_obfuscation_clean.csv
    two_way_model_X_prompt.csv
    two_way_model_X_parser.csv
    two_way_parser_X_obfuscation_clean.csv
    two_way_parser_X_dataset_clean.csv
    two_way_year_num_X_dataset_clean.csv

03_inferential_out/
    gee_all_coefficients.csv
    gee_model_metadata.csv
    mcnemar_all_valid_tests.csv
    between_group_all_valid_tests.csv

Outputs
-------
paper_tables_out/
    table_overall_recall.csv
    table_overall_recall.tex
    table_configuration_ranking.csv
    table_configuration_ranking.tex
    table_model_prompt.csv
    table_model_prompt.tex
    table_model_parser.csv
    table_model_parser.tex
    table_parser_obfuscation.csv
    table_parser_obfuscation.tex
    table_parser_dataset.csv
    table_parser_dataset.tex
    table_dataset_year.csv
    table_dataset_year.tex
    table_gee_model_selection.csv
    table_gee_model_selection.tex
    table_gee_main_effects.csv
    table_gee_main_effects.tex
    table_gee_interactions.csv
    table_gee_interactions.tex
    table_mcnemar_selected.csv
    table_mcnemar_selected.tex
    results_numbers_summary.txt

Usage
-----
python 04_build_paper_tables.py \
    --validation-dir 01_validation_out \
    --descriptive-dir 02_descriptive_out \
    --inferential-dir 03_inferential_out \
    --output-dir 04_paper_tables_out

Dependencies
------------
pip install pandas numpy
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-ready tables from previous analysis outputs."
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("01_validation_out"),
    )
    parser.add_argument(
        "--descriptive-dir",
        type=Path,
        default=Path("02_descriptive_out"),
    )
    parser.add_argument(
        "--inferential-dir",
        type=Path,
        default=Path("03_inferential_out"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("04_paper_tables_out"),
    )
    parser.add_argument(
        "--interaction-model",
        default="combined_key_interactions",
        help="GEE model specification used for final coefficient tables",
    )
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def format_p(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def format_or(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def format_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return ""
    return f"{float(low):.2f}--{float(high):.2f}"


def escape_latex(text: object) -> str:
    text = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def save_csv_and_tex(
    df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    caption: str,
    label: str,
    column_format: str | None = None,
) -> None:
    df.to_csv(csv_path, index=False)

    tex_df = df.copy()
    tex_df.columns = [escape_latex(c) for c in tex_df.columns]

    tex = tex_df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format=column_format,
        position="t",
    )
    tex_path.write_text(tex, encoding="utf-8")


def read_main_table(path: Path, factor: str, display_factor: str) -> pd.DataFrame:
    df = pd.read_csv(require(path))
    result = pd.DataFrame({
        "Factor": display_factor,
        "Level": df[factor].astype(str),
        "Recall (%)": df["recall_pct"].round(2),
        "Successes": df["successes"].astype(int),
        "Observations": df["observations"].astype(int),
    })
    return result


def build_overall_recall(descriptive_dir: Path, output_dir: Path) -> pd.DataFrame:
    tables = [
        read_main_table(
            descriptive_dir / "main_model.csv",
            "model",
            "Model",
        ),
        read_main_table(
            descriptive_dir / "main_prompt.csv",
            "prompt",
            "Prompt",
        ),
        read_main_table(
            descriptive_dir / "main_parser.csv",
            "parser",
            "Parser",
        ),
        read_main_table(
            descriptive_dir / "main_year_num.csv",
            "year_num",
            "Collection year",
        ),
        read_main_table(
            descriptive_dir / "main_dataset_clean.csv",
            "dataset_clean",
            "Dataset",
        ),
        read_main_table(
            descriptive_dir / "main_obfuscation_clean.csv",
            "obfuscation_clean",
            "Obfuscation",
        ),
    ]

    result = pd.concat(tables, ignore_index=True)
    result["Level"] = result["Level"].replace({
        "evidence": "Evidence",
        "guided": "Guided",
        "label": "Label",
        "nazario": "Nazario",
        "spam_assassin": "SpamAssassin",
        "no": "No",
        "yes": "Yes",
    })

    save_csv_and_tex(
        result,
        output_dir / "table_overall_recall.csv",
        output_dir / "table_overall_recall.tex",
        "Overall phishing detection recall across experimental factors.",
        "tab:overall_recall",
        "llrrr",
    )
    return result


def pivot_recall(
    path: Path,
    index: str,
    columns: str,
    rename_index: dict | None = None,
    rename_columns: dict | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(require(path))
    table = df.pivot(
        index=index,
        columns=columns,
        values="recall_pct",
    )
    if rename_index:
        table = table.rename(index=rename_index)
    if rename_columns:
        table = table.rename(columns=rename_columns)
    return table.round(2)


def build_condition_tables(descriptive_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    outputs = {}

    model_prompt = pivot_recall(
        descriptive_dir / "two_way_model_X_prompt.csv",
        "model",
        "prompt",
        rename_columns={
            "evidence": "Evidence",
            "guided": "Guided",
            "label": "Label",
        },
    ).reset_index().rename(columns={"model": "Model"})
    save_csv_and_tex(
        model_prompt,
        output_dir / "table_model_prompt.csv",
        output_dir / "table_model_prompt.tex",
        "Recall (\\%) by model and prompting strategy.",
        "tab:model_prompt",
    )
    outputs["model_prompt"] = model_prompt

    model_parser = pivot_recall(
        descriptive_dir / "two_way_model_X_parser.csv",
        "model",
        "parser",
    )
    model_parser["Difference (P1-P2, pp)"] = (
        model_parser["P1"] - model_parser["P2"]
    ).round(2)
    model_parser = model_parser.reset_index().rename(columns={"model": "Model"})
    save_csv_and_tex(
        model_parser,
        output_dir / "table_model_parser.csv",
        output_dir / "table_model_parser.tex",
        "Recall (\\%) by model and parser implementation.",
        "tab:model_parser",
    )
    outputs["model_parser"] = model_parser

    parser_obf = pivot_recall(
        descriptive_dir / "two_way_parser_X_obfuscation_clean.csv",
        "parser",
        "obfuscation_clean",
        rename_columns={"no": "No obfuscation", "yes": "Obfuscation"},
    )
    parser_obf["Difference (Yes-No, pp)"] = (
        parser_obf["Obfuscation"] - parser_obf["No obfuscation"]
    ).round(2)
    parser_obf = parser_obf.reset_index().rename(columns={"parser": "Parser"})
    save_csv_and_tex(
        parser_obf,
        output_dir / "table_parser_obfuscation.csv",
        output_dir / "table_parser_obfuscation.tex",
        "Recall (\\%) by parser and email obfuscation.",
        "tab:parser_obfuscation",
    )
    outputs["parser_obfuscation"] = parser_obf

    parser_dataset = pivot_recall(
        descriptive_dir / "two_way_parser_X_dataset_clean.csv",
        "parser",
        "dataset_clean",
        rename_columns={
            "nazario": "Nazario",
            "spam_assassin": "SpamAssassin",
        },
    )
    parser_dataset["Difference (Nazario-SpamAssassin, pp)"] = (
        parser_dataset["Nazario"] - parser_dataset["SpamAssassin"]
    ).round(2)
    parser_dataset = parser_dataset.reset_index().rename(columns={"parser": "Parser"})
    save_csv_and_tex(
        parser_dataset,
        output_dir / "table_parser_dataset.csv",
        output_dir / "table_parser_dataset.tex",
        "Recall (\\%) by parser and dataset.",
        "tab:parser_dataset",
    )
    outputs["parser_dataset"] = parser_dataset

    dataset_year = pivot_recall(
        descriptive_dir / "two_way_year_num_X_dataset_clean.csv",
        "dataset_clean",
        "year_num",
        rename_index={
            "nazario": "Nazario",
            "spam_assassin": "SpamAssassin",
        },
    )
    year_columns = list(dataset_year.columns)
    if 2023 in year_columns and 2025 in year_columns:
        dataset_year["Difference (2025-2023, pp)"] = (
            dataset_year[2025] - dataset_year[2023]
        ).round(2)
    elif "2023" in year_columns and "2025" in year_columns:
        dataset_year["Difference (2025-2023, pp)"] = (
            dataset_year["2025"] - dataset_year["2023"]
        ).round(2)
    dataset_year = dataset_year.reset_index().rename(
        columns={"dataset_clean": "Dataset", 2023: "2023", 2025: "2025"}
    )
    save_csv_and_tex(
        dataset_year,
        output_dir / "table_dataset_year.csv",
        output_dir / "table_dataset_year.tex",
        "Recall (\\%) by dataset and collection year.",
        "tab:dataset_year",
    )
    outputs["dataset_year"] = dataset_year

    return outputs


def build_configuration_ranking(descriptive_dir: Path, output_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(require(descriptive_dir / "configuration_ranking.csv"))
    result = df[
        [
            "observed_rank",
            "model",
            "prompt",
            "parser",
            "successes",
            "observations",
            "recall_pct",
        ]
    ].copy()
    result.columns = [
        "Rank",
        "Model",
        "Prompt",
        "Parser",
        "Successes",
        "Observations",
        "Recall (%)",
    ]
    result["Prompt"] = result["Prompt"].str.title()
    result["Recall (%)"] = result["Recall (%)"].round(2)

    save_csv_and_tex(
        result,
        output_dir / "table_configuration_ranking.csv",
        output_dir / "table_configuration_ranking.tex",
        "Ranking of all evaluated model--prompt--parser configurations.",
        "tab:configuration_rankings",
    )
    return result


def clean_term(term: str) -> str:
    mappings = [
        (r".*model.*\[T\.Qwen\]$", "Qwen"),
        (r".*prompt.*\[T\.guided\]$", "Guided prompt"),
        (r".*prompt.*\[T\.label\]$", "Label prompt"),
        (r".*parser.*\[T\.P2\]$", "Parser P2"),
        (r".*year_num.*\[T\.2025\]$", "Collection year 2025"),
        (r".*dataset_clean.*\[T\.spam_assassin\]$", "SpamAssassin"),
        (r".*obfuscation_clean.*\[T\.yes\]$", "Obfuscated email"),
    ]
    for pattern, label in mappings:
        if re.match(pattern, term):
            return label
    return term


def clean_interaction_term(term: str) -> str:
    parts = term.split(":")
    cleaned = [clean_term(part) for part in parts]
    return " × ".join(cleaned)


def build_gee_model_selection(
    inferential_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    df = pd.read_csv(require(inferential_dir / "gee_model_metadata.csv"))
    result = df[
        [
            "model_specification",
            "converged",
            "n_rows",
            "n_clusters",
            "qic",
            "qicu",
        ]
    ].copy()

    result["Model"] = (
        result["model_specification"]
        .str.replace("screen_", "", regex=False)
        .str.replace("_X_", " × ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    result.loc[
        result["model_specification"] == "combined_key_interactions",
        "Model",
    ] = "Combined key interactions"
    result.loc[
        result["model_specification"] == "main_effects",
        "Model",
    ] = "Main effects"

    result = result[
        ["Model", "converged", "n_rows", "n_clusters", "qic", "qicu"]
    ]
    result.columns = [
        "Model",
        "Converged",
        "Rows",
        "Clusters",
        "QIC",
        "QICu",
    ]
    result["QIC"] = result["QIC"].round(2)
    result["QICu"] = result["QICu"].round(2)
    result = result.sort_values("QIC")

    save_csv_and_tex(
        result,
        output_dir / "table_gee_model_selection.csv",
        output_dir / "table_gee_model_selection.tex",
        "Comparison of candidate GEE models. Lower QIC indicates improved fit.",
        "tab:qic_models",
    )
    return result


def build_gee_tables(
    inferential_dir: Path,
    output_dir: Path,
    final_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(require(inferential_dir / "gee_all_coefficients.csv"))
    final = df[df["model_specification"] == final_model].copy()

    if final.empty:
        raise ValueError(
            f"No coefficient rows found for model specification: {final_model}"
        )

    non_intercept = final[final["term"] != "Intercept"].copy()

    main = non_intercept[~non_intercept["term"].str.contains(":", regex=False)].copy()
    main["Variable"] = main["term"].map(clean_term)
    main["OR"] = main["odds_ratio"].map(format_or)
    main["95% CI"] = [
        format_ci(low, high)
        for low, high in zip(
            main["ci_low_odds_ratio"],
            main["ci_high_odds_ratio"],
        )
    ]
    main["Raw p"] = main["p_value"].map(format_p)
    main["Holm p"] = main["p_holm"].map(format_p)
    main["FDR p"] = main["p_fdr_bh"].map(format_p)
    main_result = main[
        ["Variable", "OR", "95% CI", "Raw p", "Holm p", "FDR p"]
    ]

    save_csv_and_tex(
        main_result,
        output_dir / "table_gee_main_effects.csv",
        output_dir / "table_gee_main_effects.tex",
        "Main effects from the final GEE model. Reference levels were Llama, Evidence, P1, 2023, Nazario, and non-obfuscated email.",
        "tab:gee_main",
    )

    inter = non_intercept[non_intercept["term"].str.contains(":", regex=False)].copy()
    inter["Interaction"] = inter["term"].map(clean_interaction_term)
    inter["OR"] = inter["odds_ratio"].map(format_or)
    inter["95% CI"] = [
        format_ci(low, high)
        for low, high in zip(
            inter["ci_low_odds_ratio"],
            inter["ci_high_odds_ratio"],
        )
    ]
    inter["Raw p"] = inter["p_value"].map(format_p)
    inter["Holm p"] = inter["p_holm"].map(format_p)
    inter["FDR p"] = inter["p_fdr_bh"].map(format_p)
    inter["Raw p numeric"] = inter["p_value"]
    inter["Holm p numeric"] = inter["p_holm"]
    inter["FDR p numeric"] = inter["p_fdr_bh"]

    inter_result = inter[
        [
            "Interaction",
            "OR",
            "95% CI",
            "Raw p",
            "Holm p",
            "FDR p",
            "Raw p numeric",
            "Holm p numeric",
            "FDR p numeric",
        ]
    ].sort_values("Raw p numeric")

    save_csv_and_tex(
        inter_result,
        output_dir / "table_gee_interactions.csv",
        output_dir / "table_gee_interactions.tex",
        "Interaction effects from the final GEE model.",
        "tab:gee_interactions",
    )

    notable = inter_result[
        (inter_result["Holm p numeric"] < 0.05)
        | (inter_result["FDR p numeric"] < 0.05)
        | (inter_result["Raw p numeric"] < 0.05)
    ].copy()

    save_csv_and_tex(
        notable,
        output_dir / "table_gee_interactions_notable.csv",
        output_dir / "table_gee_interactions_notable.tex",
        "Interaction effects with raw or multiplicity-adjusted $p<0.05$.",
        "tab:gee_interactions_notable",
    )

    return main_result, inter_result


def build_selected_mcnemar(
    inferential_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    df = pd.read_csv(require(inferential_dir / "mcnemar_all_valid_tests.csv"))

    # Include all non-subgroup tests, which are condition-specific valid tests.
    selected = df[df["subgroup_factor"] == "all"].copy()

    selected["Comparison"] = (
        selected["level_a"].astype(str)
        + " vs "
        + selected["level_b"].astype(str)
    )
    selected["Condition"] = ""

    for col, label in [
        ("fixed_model", "model"),
        ("fixed_prompt", "prompt"),
        ("fixed_parser", "parser"),
    ]:
        if col in selected.columns:
            selected["Condition"] += np.where(
                selected[col].notna(),
                label + "=" + selected[col].astype(str) + "; ",
                "",
            )

    selected["Condition"] = selected["Condition"].str.rstrip("; ")
    selected["Recall A (%)"] = selected["recall_a_pct"].round(2)
    selected["Recall B (%)"] = selected["recall_b_pct"].round(2)
    selected["Difference (pp)"] = selected[
        "difference_percentage_points"
    ].round(2)
    selected["Discordant A/B"] = (
        selected["a_correct_b_wrong"].astype(int).astype(str)
        + "/"
        + selected["a_wrong_b_correct"].astype(int).astype(str)
    )
    selected["Raw p"] = selected["p_value"].map(format_p)
    selected["Holm p"] = selected["p_holm"].map(format_p)
    selected["FDR p"] = selected["p_fdr_bh"].map(format_p)

    result = selected[
        [
            "varying_factor",
            "Comparison",
            "Condition",
            "n_pairs",
            "Recall A (%)",
            "Recall B (%)",
            "Difference (pp)",
            "Discordant A/B",
            "Raw p",
            "Holm p",
            "FDR p",
        ]
    ].copy()

    result.columns = [
        "Factor",
        "Comparison",
        "Fixed condition",
        "Pairs",
        "Recall A (%)",
        "Recall B (%)",
        "Difference (pp)",
        "Discordant A/B",
        "Raw p",
        "Holm p",
        "FDR p",
    ]

    result = result.sort_values(["Factor", "Raw p"])

    save_csv_and_tex(
        result,
        output_dir / "table_mcnemar_selected.csv",
        output_dir / "table_mcnemar_selected.tex",
        "Valid condition-specific McNemar comparisons.",
        "tab:mcnemar_selected",
    )
    return result


def write_summary(
    overall: pd.DataFrame,
    ranking: pd.DataFrame,
    model_selection: pd.DataFrame,
    gee_main: pd.DataFrame,
    gee_interactions: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = []
    lines.append("RESULTS NUMBERS SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append("OVERALL RECALL")
    lines.append(overall.to_string(index=False))
    lines.append("")
    lines.append("TOP AND BOTTOM CONFIGURATIONS")
    lines.append(ranking.head(3).to_string(index=False))
    lines.append("")
    lines.append(ranking.tail(3).to_string(index=False))
    lines.append("")
    lines.append("BEST GEE MODEL")
    lines.append(model_selection.head(1).to_string(index=False))
    lines.append("")
    lines.append("FINAL GEE MAIN EFFECTS")
    lines.append(gee_main.to_string(index=False))
    lines.append("")
    lines.append("FINAL GEE INTERACTIONS")
    lines.append(gee_interactions.to_string(index=False))
    lines.append("")
    lines.append("INTERPRETATION NOTE")
    lines.append(
        "With interactions in the model, main-effect coefficients are "
        "conditional on the reference levels. Do not describe them as "
        "unconditional overall effects."
    )

    (output_dir / "results_numbers_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Building overall recall table...")
    overall = build_overall_recall(
        args.descriptive_dir,
        args.output_dir,
    )

    print("Building condition tables...")
    build_condition_tables(
        args.descriptive_dir,
        args.output_dir,
    )

    print("Building configuration ranking...")
    ranking = build_configuration_ranking(
        args.descriptive_dir,
        args.output_dir,
    )

    print("Building GEE model-selection table...")
    model_selection = build_gee_model_selection(
        args.inferential_dir,
        args.output_dir,
    )

    print("Building exact GEE coefficient tables...")
    gee_main, gee_interactions = build_gee_tables(
        args.inferential_dir,
        args.output_dir,
        args.interaction_model,
    )

    print("Building condition-specific McNemar table...")
    build_selected_mcnemar(
        args.inferential_dir,
        args.output_dir,
    )

    write_summary(
        overall,
        ranking,
        model_selection,
        gee_main,
        gee_interactions,
        args.output_dir,
    )

    print("")
    print("=" * 72)
    print("PAPER TABLE EXTRACTION COMPLETE")
    print("=" * 72)
    print(f"Output directory: {args.output_dir.resolve()}")
    print("")
    print("Open these files first:")
    print("  results_numbers_summary.txt")
    print("  table_gee_main_effects.tex")
    print("  table_gee_interactions.tex")
    print("  table_gee_model_selection.tex")
    print("  table_configuration_ranking.tex")
    print("  table_overall_recall.tex")


if __name__ == "__main__":
    main()
