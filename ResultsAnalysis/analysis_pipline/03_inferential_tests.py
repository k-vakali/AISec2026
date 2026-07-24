"""
03_inferential_tests.py

Inferential analysis for repeated-measures LLM phishing detection results.

Statistical design
------------------
- The underlying email is the clustering/matching unit.
- McNemar tests are used only for paired within-email comparisons where
  each email contributes exactly one pair.
- Chi-square/Fisher tests are used only for between-email factors within a
  fixed model × prompt × parser configuration, so each email contributes
  exactly one binary outcome.
- Logistic GEE models are clustered by base_email_id to account for all
  repeated observations across model, prompt, and parser.

Analyses
--------
1. Exhaustive valid McNemar tests:
   - model comparisons within fixed prompt × parser
   - prompt comparisons within fixed model × parser
   - parser comparisons within fixed model × prompt
   - all of the above repeated within year, dataset, and obfuscation groups

2. Valid chi-square/Fisher tests:
   - year comparisons within each fixed model × prompt × parser
   - dataset comparisons within each fixed model × prompt × parser
   - obfuscation comparisons within each fixed model × prompt × parser
   - simple effects such as year within dataset, dataset within year,
     and obfuscation within year/dataset

3. Logistic GEE:
   - main-effects model
   - one-interaction-at-a-time screening models
   - combined theory-driven interaction model
   - robust SEs clustered by email
   - Holm and Benjamini-Hochberg correction

Usage
-----
python 03_inferential_tests.py validated_cleaned_data.csv \
    --output-dir 03_inferential_out

Dependencies
------------
pip install pandas numpy scipy statsmodels patsy
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------
# Arguments and loading
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired tests, between-email tests, and clustered GEE."
    )
    parser.add_argument("csv_file", type=Path, help="Input CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("03_inferential_out"),
        help="Output directory",
    )
    parser.add_argument(
        "--minimum-group-size",
        type=int,
        default=10,
        help="Minimum emails required in each between-email group",
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
        df["model"] = df["model"].astype("string").str.strip()
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
    else:
        required = {
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

        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                "Missing required columns: " + ", ".join(sorted(missing))
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

        status = df["status"].astype("string").str.strip().str.lower()
        parse_ok = normalise_bool(df["parse_ok"])
        pred = pd.to_numeric(df["pred_label"], errors="coerce")
        truth = pd.to_numeric(df["ground_truth"], errors="coerce")

        df["detected"] = (
            (truth == 1)
            & (status == "ok")
            & (parse_ok == True)
            & (pred == 1)
        ).astype(int)

    df = df.dropna(
        subset=[
            "base_email_id",
            "model",
            "prompt",
            "parser",
            "year_num",
            "dataset_clean",
            "obfuscation_clean",
            "detected",
        ]
    ).copy()

    df["detected"] = df["detected"].astype(int)
    df["year_num"] = df["year_num"].astype(int)

    return df


# ---------------------------------------------------------------------
# Multiple-testing correction
# ---------------------------------------------------------------------

def adjust_pvalues(
    df: pd.DataFrame,
    p_column: str = "p_value",
    family_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply Holm and BH corrections within test families.
    """
    result = df.copy()
    result["p_holm"] = np.nan
    result["p_fdr_bh"] = np.nan

    if result.empty:
        return result

    if family_columns:
        groups = result.groupby(
            family_columns,
            dropna=False,
            observed=True,
        )
    else:
        groups = [("all", result)]

    for _, group in groups:
        valid = group[p_column].notna()
        indexes = group.index[valid]
        pvalues = group.loc[indexes, p_column].astype(float)

        if len(pvalues) == 0:
            continue

        result.loc[indexes, "p_holm"] = multipletests(
            pvalues,
            method="holm",
        )[1]
        result.loc[indexes, "p_fdr_bh"] = multipletests(
            pvalues,
            method="fdr_bh",
        )[1]

    return result


# ---------------------------------------------------------------------
# McNemar tests
# ---------------------------------------------------------------------

def exact_mcnemar_result(
    paired: pd.DataFrame,
    level_a: object,
    level_b: object,
) -> dict:
    a_correct_b_wrong = int(
        ((paired[level_a] == 1) & (paired[level_b] == 0)).sum()
    )
    a_wrong_b_correct = int(
        ((paired[level_a] == 0) & (paired[level_b] == 1)).sum()
    )
    both_correct = int(
        ((paired[level_a] == 1) & (paired[level_b] == 1)).sum()
    )
    both_wrong = int(
        ((paired[level_a] == 0) & (paired[level_b] == 0)).sum()
    )

    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant == 0:
        p_value = 1.0
    else:
        test = mcnemar(
            [
                [both_correct, a_correct_b_wrong],
                [a_wrong_b_correct, both_wrong],
            ],
            exact=True,
        )
        p_value = float(test.pvalue)

    recall_a = float(paired[level_a].mean())
    recall_b = float(paired[level_b].mean())

    # Matched-pair odds ratio. Infinite/zero values are retained explicitly.
    if a_wrong_b_correct == 0 and a_correct_b_wrong > 0:
        matched_odds_ratio = np.inf
    elif a_correct_b_wrong == 0 and a_wrong_b_correct > 0:
        matched_odds_ratio = 0.0
    elif a_correct_b_wrong == 0 and a_wrong_b_correct == 0:
        matched_odds_ratio = np.nan
    else:
        matched_odds_ratio = (
            a_correct_b_wrong / a_wrong_b_correct
        )

    return {
        "n_pairs": len(paired),
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "recall_a": recall_a,
        "recall_b": recall_b,
        "recall_a_pct": 100 * recall_a,
        "recall_b_pct": 100 * recall_b,
        "difference": recall_a - recall_b,
        "difference_percentage_points": 100 * (recall_a - recall_b),
        "matched_odds_ratio": matched_odds_ratio,
        "p_value": p_value,
    }


def run_mcnemar_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    repeated_factors = ["model", "prompt", "parser"]
    subgroup_factors = [
        None,
        "year_num",
        "dataset_clean",
        "obfuscation_clean",
    ]

    for varying_factor in repeated_factors:
        fixed_factors = [
            factor for factor in repeated_factors
            if factor != varying_factor
        ]

        varying_levels = sorted(
            df[varying_factor].dropna().unique().tolist()
        )

        fixed_level_lists = [
            sorted(df[factor].dropna().unique().tolist())
            for factor in fixed_factors
        ]

        for fixed_values_tuple in itertools.product(*fixed_level_lists):
            fixed_values = dict(zip(fixed_factors, fixed_values_tuple))

            for subgroup_factor in subgroup_factors:
                if subgroup_factor is None:
                    subgroup_levels = [None]
                else:
                    subgroup_levels = sorted(
                        df[subgroup_factor].dropna().unique().tolist()
                    )

                for subgroup_level in subgroup_levels:
                    data = df.copy()

                    for key, value in fixed_values.items():
                        data = data[data[key] == value]

                    if subgroup_factor is not None:
                        data = data[
                            data[subgroup_factor] == subgroup_level
                        ]

                    for level_a, level_b in itertools.combinations(
                        varying_levels,
                        2,
                    ):
                        pair_data = data[
                            data[varying_factor].isin([level_a, level_b])
                        ]

                        paired = pair_data.pivot_table(
                            index="base_email_id",
                            columns=varying_factor,
                            values="detected",
                            aggfunc="first",
                        )

                        if (
                            level_a not in paired.columns
                            or level_b not in paired.columns
                        ):
                            continue

                        paired = paired.dropna(
                            subset=[level_a, level_b]
                        )

                        if paired.empty:
                            continue

                        result = exact_mcnemar_result(
                            paired,
                            level_a,
                            level_b,
                        )

                        row = {
                            "test_family": (
                                f"mcnemar_{varying_factor}"
                            ),
                            "varying_factor": varying_factor,
                            "level_a": level_a,
                            "level_b": level_b,
                            "subgroup_factor": (
                                subgroup_factor
                                if subgroup_factor is not None
                                else "all"
                            ),
                            "subgroup_level": (
                                subgroup_level
                                if subgroup_factor is not None
                                else "all"
                            ),
                        }

                        for key, value in fixed_values.items():
                            row[f"fixed_{key}"] = value

                        row.update(result)
                        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = adjust_pvalues(
            result,
            p_column="p_value",
            family_columns=[
                "test_family",
                "subgroup_factor",
            ],
        )

    return result


# ---------------------------------------------------------------------
# Chi-square / Fisher tests
# ---------------------------------------------------------------------

def binary_group_test(
    data: pd.DataFrame,
    group_factor: str,
    level_a: object,
    level_b: object,
    minimum_group_size: int,
) -> dict | None:
    """
    Compare two independent email groups.

    Handles degenerate 2x2 tables safely. A table is degenerate when every
    observation has the same outcome, for example both groups have 100%
    recall. In that case there is no outcome variation to test and p=1.
    """
    subset = data[data[group_factor].isin([level_a, level_b])].copy()

    counts = pd.crosstab(
        subset[group_factor],
        subset["detected"],
    ).reindex(
        index=[level_a, level_b],
        columns=[0, 1],
        fill_value=0,
    )

    n_a = int(counts.loc[level_a].sum())
    n_b = int(counts.loc[level_b].sum())

    if n_a < minimum_group_size or n_b < minimum_group_size:
        return None

    failures_a = int(counts.loc[level_a, 0])
    successes_a = int(counts.loc[level_a, 1])
    failures_b = int(counts.loc[level_b, 0])
    successes_b = int(counts.loc[level_b, 1])

    recall_a = successes_a / n_a if n_a else np.nan
    recall_b = successes_b / n_b if n_b else np.nan

    table = counts.values.astype(int)
    column_totals = table.sum(axis=0)
    row_totals = table.sum(axis=1)

    chi2 = np.nan
    dof = np.nan
    minimum_expected_count = np.nan
    odds_ratio = np.nan

    # If one outcome column is entirely zero, all observations have the
    # same result. There is no evidence of a group difference.
    if (column_totals == 0).any():
        p_value = 1.0
        test_used = "No outcome variation"
    elif (row_totals == 0).any():
        # This should not occur after the minimum group-size check, but keep
        # it safe in case of malformed input.
        return None
    else:
        # Fisher's exact test is valid for sparse 2x2 tables and zeros.
        odds_ratio, fisher_p = fisher_exact(
            table,
            alternative="two-sided",
        )

        try:
            chi2, chi_p, dof, expected = chi2_contingency(
                table,
                correction=False,
            )
            minimum_expected_count = float(expected.min())

            if (expected < 5).any():
                p_value = float(fisher_p)
                test_used = "Fisher exact"
            else:
                p_value = float(chi_p)
                test_used = "Pearson chi-square"
        except ValueError:
            # Defensive fallback for any remaining degenerate table.
            p_value = float(fisher_p)
            test_used = "Fisher exact fallback"

    return {
        "n_a": n_a,
        "n_b": n_b,
        "successes_a": successes_a,
        "successes_b": successes_b,
        "failures_a": failures_a,
        "failures_b": failures_b,
        "recall_a": recall_a,
        "recall_b": recall_b,
        "recall_a_pct": 100 * recall_a,
        "recall_b_pct": 100 * recall_b,
        "difference": recall_a - recall_b,
        "difference_percentage_points": 100 * (recall_a - recall_b),
        "odds_ratio": odds_ratio,
        "chi_square": chi2,
        "degrees_of_freedom": dof,
        "minimum_expected_count": minimum_expected_count,
        "test_used": test_used,
        "p_value": float(p_value),
    }


def run_between_email_tests(
    df: pd.DataFrame,
    minimum_group_size: int,
) -> pd.DataFrame:
    """
    Valid only within a fixed model × prompt × parser configuration,
    giving one outcome per email.
    """
    rows: list[dict] = []

    fixed_experimental = ["model", "prompt", "parser"]
    between_factors = [
        "year_num",
        "dataset_clean",
        "obfuscation_clean",
    ]

    fixed_level_lists = [
        sorted(df[f].dropna().unique().tolist())
        for f in fixed_experimental
    ]

    for fixed_tuple in itertools.product(*fixed_level_lists):
        fixed_values = dict(zip(fixed_experimental, fixed_tuple))
        config_data = df.copy()

        for key, value in fixed_values.items():
            config_data = config_data[config_data[key] == value]

        # Main between-email comparisons within configuration.
        for varying_factor in between_factors:
            levels = sorted(
                config_data[varying_factor]
                .dropna()
                .unique()
                .tolist()
            )

            for level_a, level_b in itertools.combinations(levels, 2):
                test = binary_group_test(
                    config_data,
                    varying_factor,
                    level_a,
                    level_b,
                    minimum_group_size,
                )

                if test is None:
                    continue

                row = {
                    "test_family": (
                        f"between_{varying_factor}"
                    ),
                    "varying_factor": varying_factor,
                    "level_a": level_a,
                    "level_b": level_b,
                    "conditioning_factor": "none",
                    "conditioning_level": "all",
                }
                row.update(
                    {
                        f"fixed_{key}": value
                        for key, value in fixed_values.items()
                    }
                )
                row.update(test)
                rows.append(row)

        # Simple effects among between-email factors.
        # Example: year within dataset, dataset within year,
        # obfuscation within year, etc.
        for varying_factor, conditioning_factor in itertools.permutations(
            between_factors,
            2,
        ):
            varying_levels = sorted(
                config_data[varying_factor]
                .dropna()
                .unique()
                .tolist()
            )
            conditioning_levels = sorted(
                config_data[conditioning_factor]
                .dropna()
                .unique()
                .tolist()
            )

            for conditioning_level in conditioning_levels:
                subgroup = config_data[
                    config_data[conditioning_factor]
                    == conditioning_level
                ]

                for level_a, level_b in itertools.combinations(
                    varying_levels,
                    2,
                ):
                    test = binary_group_test(
                        subgroup,
                        varying_factor,
                        level_a,
                        level_b,
                        minimum_group_size,
                    )

                    if test is None:
                        continue

                    row = {
                        "test_family": (
                            f"between_{varying_factor}_within_"
                            f"{conditioning_factor}"
                        ),
                        "varying_factor": varying_factor,
                        "level_a": level_a,
                        "level_b": level_b,
                        "conditioning_factor": conditioning_factor,
                        "conditioning_level": conditioning_level,
                    }
                    row.update(
                        {
                            f"fixed_{key}": value
                            for key, value in fixed_values.items()
                        }
                    )
                    row.update(test)
                    rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result = adjust_pvalues(
            result,
            p_column="p_value",
            family_columns=["test_family"],
        )

    return result


# ---------------------------------------------------------------------
# Logistic GEE
# ---------------------------------------------------------------------

def make_reference_term(
    column: str,
    preferred_reference: object,
    observed_levels: Iterable[object],
) -> str:
    levels = list(observed_levels)

    if preferred_reference in levels:
        return (
            f"C({column}, Treatment(reference={preferred_reference!r}))"
        )

    return f"C({column})"


def fit_gee_model(
    df: pd.DataFrame,
    formula: str,
):
    model = smf.gee(
        formula=formula,
        groups="base_email_id",
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit(maxiter=300)


def extract_gee(
    result,
    specification: str,
    formula: str,
) -> pd.DataFrame:
    conf = result.conf_int()

    return pd.DataFrame(
        {
            "model_specification": specification,
            "formula": formula,
            "term": result.params.index,
            "coefficient_log_odds": result.params.values,
            "standard_error": result.bse.values,
            "z_statistic": result.tvalues.values,
            "p_value": result.pvalues.values,
            "odds_ratio": np.exp(result.params.values),
            "ci_low_log_odds": conf[0].values,
            "ci_high_log_odds": conf[1].values,
            "ci_low_odds_ratio": np.exp(conf[0].values),
            "ci_high_odds_ratio": np.exp(conf[1].values),
        }
    )


def run_gee_models(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_term = make_reference_term(
        "model",
        "Llama",
        df["model"].unique(),
    )
    prompt_term = make_reference_term(
        "prompt",
        "evidence",
        df["prompt"].unique(),
    )
    parser_term = make_reference_term(
        "parser",
        "P1",
        df["parser"].unique(),
    )
    year_term = make_reference_term(
        "year_num",
        2023,
        df["year_num"].unique(),
    )
    dataset_term = make_reference_term(
        "dataset_clean",
        "nazario",
        df["dataset_clean"].unique(),
    )
    obf_term = make_reference_term(
        "obfuscation_clean",
        "no",
        df["obfuscation_clean"].unique(),
    )

    main_terms = [
        model_term,
        prompt_term,
        parser_term,
        year_term,
        dataset_term,
        obf_term,
    ]

    main_formula = "detected ~ " + " + ".join(main_terms)

    interaction_pairs = {
        "model_X_prompt": (model_term, prompt_term),
        "model_X_parser": (model_term, parser_term),
        "model_X_year": (model_term, year_term),
        "model_X_dataset": (model_term, dataset_term),
        "model_X_obfuscation": (model_term, obf_term),
        "prompt_X_parser": (prompt_term, parser_term),
        "prompt_X_year": (prompt_term, year_term),
        "prompt_X_dataset": (prompt_term, dataset_term),
        "prompt_X_obfuscation": (prompt_term, obf_term),
        "parser_X_year": (parser_term, year_term),
        "parser_X_dataset": (parser_term, dataset_term),
        "parser_X_obfuscation": (parser_term, obf_term),
        "year_X_dataset": (year_term, dataset_term),
        "year_X_obfuscation": (year_term, obf_term),
        "dataset_X_obfuscation": (dataset_term, obf_term),
    }

    formulas: dict[str, str] = {
        "main_effects": main_formula,
    }

    # One interaction at a time, retaining all main effects.
    for name, (term_a, term_b) in interaction_pairs.items():
        formulas[f"screen_{name}"] = (
            main_formula + f" + {term_a}:{term_b}"
        )

    # Combined theory/data-driven model based on the descriptive findings.
    combined_interactions = [
        f"{model_term}:{prompt_term}",
        f"{model_term}:{parser_term}",
        f"{prompt_term}:{parser_term}",
        f"{parser_term}:{obf_term}",
        f"{parser_term}:{dataset_term}",
        f"{year_term}:{dataset_term}",
        f"{prompt_term}:{year_term}",
    ]

    formulas["combined_key_interactions"] = (
        main_formula + " + " + " + ".join(combined_interactions)
    )

    coefficient_frames = []
    metadata_rows = []

    for specification, formula in formulas.items():
        try:
            result = fit_gee_model(df, formula)

            coefficient_frame = extract_gee(
                result,
                specification,
                formula,
            )
            coefficient_frames.append(coefficient_frame)

            qic = np.nan
            qicu = np.nan
            try:
                qic_result = result.qic()
                if isinstance(qic_result, tuple):
                    qic, qicu = qic_result
                else:
                    qic = qic_result
            except Exception:
                pass

            metadata_rows.append(
                {
                    "model_specification": specification,
                    "formula": formula,
                    "converged": True,
                    "n_rows": int(result.nobs),
                    "n_clusters": int(
                        df["base_email_id"].nunique()
                    ),
                    "qic": qic,
                    "qicu": qicu,
                    "error": "",
                }
            )

            (
                output_dir
                / f"gee_{specification}_summary.txt"
            ).write_text(
                result.summary().as_text(),
                encoding="utf-8",
            )

        except Exception as exc:
            metadata_rows.append(
                {
                    "model_specification": specification,
                    "formula": formula,
                    "converged": False,
                    "n_rows": len(df),
                    "n_clusters": int(
                        df["base_email_id"].nunique()
                    ),
                    "qic": np.nan,
                    "qicu": np.nan,
                    "error": repr(exc),
                }
            )

    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame()
    )

    if not coefficients.empty:
        coefficients = adjust_pvalues(
            coefficients,
            p_column="p_value",
            family_columns=["model_specification"],
        )

    metadata = pd.DataFrame(metadata_rows)

    (output_dir / "gee_formulas.json").write_text(
        json.dumps(formulas, indent=2),
        encoding="utf-8",
    )

    return coefficients, metadata


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def make_notable_results(
    mcnemar_df: pd.DataFrame,
    between_df: pd.DataFrame,
    gee_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not mcnemar_df.empty:
        notable_mcnemar = mcnemar_df[
            (mcnemar_df["p_holm"] < 0.05)
            | (
                mcnemar_df["difference_percentage_points"]
                .abs()
                .ge(5)
            )
        ].sort_values(
            ["p_holm", "difference_percentage_points"],
            ascending=[True, False],
        )

        notable_mcnemar.to_csv(
            output_dir / "mcnemar_notable_results.csv",
            index=False,
        )

    if not between_df.empty:
        notable_between = between_df[
            (between_df["p_holm"] < 0.05)
            | (
                between_df["difference_percentage_points"]
                .abs()
                .ge(5)
            )
        ].sort_values(
            ["p_holm", "difference_percentage_points"],
            ascending=[True, False],
        )

        notable_between.to_csv(
            output_dir / "between_group_notable_results.csv",
            index=False,
        )

    if not gee_df.empty:
        interaction_mask = gee_df["term"].str.contains(
            ":",
            regex=False,
            na=False,
        )

        notable_gee = gee_df[
            interaction_mask
            & (
                (gee_df["p_holm"] < 0.05)
                | (gee_df["p_fdr_bh"] < 0.05)
                | (gee_df["p_value"] < 0.05)
            )
        ].sort_values(
            ["model_specification", "p_value"]
        )

        notable_gee.to_csv(
            output_dir / "gee_notable_interactions.csv",
            index=False,
        )


def write_summary(
    df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
    between_df: pd.DataFrame,
    gee_df: pd.DataFrame,
    gee_metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines: list[str] = []

    lines.append("INFERENTIAL ANALYSIS SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Rows: {len(df):,}")
    lines.append(
        f"Unique emails/clusters: {df['base_email_id'].nunique():,}"
    )
    lines.append("")

    lines.append("METHODOLOGICAL RULES")
    lines.append("-" * 72)
    lines.append(
        "McNemar tests compare model, prompt, or parser only within fixed "
        "conditions so each email contributes exactly one matched pair."
    )
    lines.append(
        "Chi-square/Fisher tests compare year, dataset, or obfuscation only "
        "within a fixed model × prompt × parser configuration."
    )
    lines.append(
        "Overall adjusted associations and interactions are assessed using "
        "logistic GEE clustered by underlying email."
    )
    lines.append("")

    lines.append("TEST COUNTS")
    lines.append("-" * 72)
    lines.append(f"McNemar tests: {len(mcnemar_df):,}")
    lines.append(f"Between-email tests: {len(between_df):,}")
    lines.append(
        f"GEE coefficient rows: {len(gee_df):,}"
    )
    lines.append("")

    if not mcnemar_df.empty:
        lines.append("TOP HOLM-ADJUSTED MCNEMAR RESULTS")
        lines.append("-" * 72)

        top = mcnemar_df.sort_values("p_holm").head(20)

        display_columns = [
            "varying_factor",
            "level_a",
            "level_b",
            "subgroup_factor",
            "subgroup_level",
            "difference_percentage_points",
            "a_correct_b_wrong",
            "a_wrong_b_correct",
            "p_value",
            "p_holm",
        ]

        fixed_columns = [
            col for col in [
                "fixed_model",
                "fixed_prompt",
                "fixed_parser",
            ]
            if col in top.columns
        ]

        lines.append(
            top[display_columns + fixed_columns].to_string(
                index=False,
                formatters={
                    "difference_percentage_points": (
                        lambda x: f"{x:.2f}"
                    ),
                    "p_value": lambda x: f"{x:.4g}",
                    "p_holm": lambda x: f"{x:.4g}",
                },
            )
        )
        lines.append("")

    if not between_df.empty:
        lines.append("TOP HOLM-ADJUSTED BETWEEN-EMAIL RESULTS")
        lines.append("-" * 72)

        top = between_df.sort_values("p_holm").head(20)

        display_columns = [
            "varying_factor",
            "level_a",
            "level_b",
            "conditioning_factor",
            "conditioning_level",
            "difference_percentage_points",
            "test_used",
            "p_value",
            "p_holm",
            "fixed_model",
            "fixed_prompt",
            "fixed_parser",
        ]

        lines.append(
            top[display_columns].to_string(
                index=False,
                formatters={
                    "difference_percentage_points": (
                        lambda x: f"{x:.2f}"
                    ),
                    "p_value": lambda x: f"{x:.4g}",
                    "p_holm": lambda x: f"{x:.4g}",
                },
            )
        )
        lines.append("")

    lines.append("GEE MODEL STATUS")
    lines.append("-" * 72)
    lines.append(
        gee_metadata[
            [
                "model_specification",
                "converged",
                "n_rows",
                "n_clusters",
                "qic",
                "error",
            ]
        ].to_string(index=False)
    )
    lines.append("")

    lines.append("INTERPRETATION WARNING")
    lines.append("-" * 72)
    lines.append(
        "Significance of a simple effect in one subgroup and non-significance "
        "in another subgroup does not prove an interaction."
    )
    lines.append(
        "Use GEE interaction terms, and later cluster-bootstrap "
        "difference-in-differences, for interaction claims."
    )
    lines.append(
        "Chi-square/Fisher results are configuration-specific and should not "
        "be described as overall year, dataset, or obfuscation effects."
    )

    (output_dir / "inferential_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.csv_file.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {args.csv_file}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = prepare_data(args.csv_file)

    print("Running valid paired McNemar tests...")
    mcnemar_df = run_mcnemar_tests(df)
    mcnemar_df.to_csv(
        args.output_dir / "mcnemar_all_valid_tests.csv",
        index=False,
    )

    print("Running valid chi-square/Fisher tests...")
    between_df = run_between_email_tests(
        df,
        minimum_group_size=args.minimum_group_size,
    )
    between_df.to_csv(
        args.output_dir / "between_group_all_valid_tests.csv",
        index=False,
    )

    print("Fitting clustered logistic GEE models...")
    gee_df, gee_metadata = run_gee_models(
        df,
        args.output_dir,
    )

    gee_df.to_csv(
        args.output_dir / "gee_all_coefficients.csv",
        index=False,
    )
    gee_metadata.to_csv(
        args.output_dir / "gee_model_metadata.csv",
        index=False,
    )

    make_notable_results(
        mcnemar_df,
        between_df,
        gee_df,
        args.output_dir,
    )

    write_summary(
        df,
        mcnemar_df,
        between_df,
        gee_df,
        gee_metadata,
        args.output_dir,
    )

    print("")
    print("=" * 72)
    print("INFERENTIAL ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Input: {args.csv_file}")
    print(f"Output: {args.output_dir.resolve()}")
    print("")
    print("Open these files first:")
    print("  inferential_summary.txt")
    print("  mcnemar_notable_results.csv")
    print("  between_group_notable_results.csv")
    print("  gee_model_metadata.csv")
    print("  gee_notable_interactions.csv")
    print("")
    print(
        "The next step after reviewing these results is a targeted "
        "email-cluster bootstrap for effect-size confidence intervals."
    )


if __name__ == "__main__":
    main()